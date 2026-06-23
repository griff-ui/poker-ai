"""
Abstracted MCCFR for NLHE — performance-optimized with precomputed buckets.

Key optimization: bucket lookups for flop/turn require Monte Carlo sampling.
We precompute all street buckets for both players at deal-time, before traversal.
This means each unique (hole, board) pair is computed exactly ONCE per iteration
regardless of how many nodes visit that bucket during the tree traversal.

Information set key: f"{player}|b{bucket}|s{street}|{history}"
Bet abstraction: fold(f), check/call(c), bet33(r1), bet67(r2), pot(r3)
"""

import random, time, json, os
from collections import defaultdict
from treys import Card, Evaluator

from abstraction.card_abstraction import CardAbstraction
from abstraction.equity import remaining_deck, ALL_CARDS

evaluator = Evaluator()

MAX_RAISES  = 2
BET_NAMES   = ['r1', 'r2', 'r3']     # 33%, 67%, 100% pot
BET_FRACS   = {'r1': 0.33, 'r2': 0.67, 'r3': 1.00}
ALL_ACTIONS = ['f', 'c'] + BET_NAMES


def valid_actions(n_raises: int) -> list[str]:
    return ['f', 'c'] if n_raises >= MAX_RAISES else ALL_ACTIONS


def is_bet(a: str) -> bool:
    return a in BET_NAMES


def count_raises(hist_parts: list[str]) -> int:
    return sum(1 for p in hist_parts if is_bet(p))


def street_over(hist_parts: list[str]) -> bool:
    if not hist_parts:
        return False
    if hist_parts[-1] == 'f':
        return True
    if hist_parts[-1] == 'c':
        if any(is_bet(p) for p in hist_parts[:-1]):
            return True
        if len(hist_parts) == 2 and hist_parts[0] == 'c' and hist_parts[1] == 'c':
            return True
    return False


# ─────────────────────────────────────────────────────────────
# Pot tracker
# ─────────────────────────────────────────────────────────────

SMALL_BLIND = 1.0
BIG_BLIND   = 2.0


class Pot:
    __slots__ = ('contrib', 'pot')

    def __init__(self):
        self.contrib = [SMALL_BLIND, BIG_BLIND]
        self.pot     = SMALL_BLIND + BIG_BLIND

    def copy(self):
        p = Pot.__new__(Pot)
        p.contrib = self.contrib.copy()
        p.pot     = self.pot
        return p

    def call(self, player: int):
        amt = max(self.contrib[1 - player] - self.contrib[player], 0.0)
        self.contrib[player] += amt
        self.pot += amt

    def bet(self, player: int, frac: float):
        MAX_STACK = 200.0   # Starting stack in big blinds
        amt_call = max(self.contrib[1 - player] - self.contrib[player], 0.0)
        # Cap call at remaining stack
        amt_call = min(amt_call, MAX_STACK - self.contrib[player])
        self.contrib[player] += amt_call
        self.pot += amt_call
        raise_amt = max(1.0, self.pot * frac)
        # Cap raise at remaining stack
        raise_amt = min(raise_amt, MAX_STACK - self.contrib[player])
        self.contrib[player] += raise_amt
        self.pot += raise_amt

    def winnings(self, winner: int) -> float:
        return self.contrib[1 - winner]

    def loss(self, loser: int) -> float:
        return -self.contrib[loser]


# ─────────────────────────────────────────────────────────────
# Precomputed hand context (computed ONCE per iteration)
# ─────────────────────────────────────────────────────────────

class HandContext:
    """
    All information about a single dealt hand, pre-computed before traversal.
    Bucket lookups happen here (Monte Carlo is called at most once per street per player).
    """
    def __init__(self, hole0: list[int], hole1: list[int], abstraction: CardAbstraction):
        self.hole0 = hole0
        self.hole1 = hole1
        self.abstraction = abstraction

        # Deal full 5 community cards upfront (we'll reveal them street by street)
        deck = remaining_deck(hole0 + hole1)
        board5 = random.sample(deck, 5)
        self.flop  = board5[:3]
        self.turn  = board5[3:4]
        self.river = board5[4:5]

        # Precompute buckets for each player at each street
        # This is the expensive part — done once, not on every node visit
        self._buckets: dict[tuple[int,int], int] = {}  # (player, street) → bucket
        self._precompute_buckets()

    def board(self, street: int) -> list[int]:
        if street == 0: return []
        if street == 1: return self.flop
        if street == 2: return self.flop + self.turn
        return self.flop + self.turn + self.river

    def bucket(self, player: int, street: int) -> int:
        return self._buckets[(player, street)]

    def _precompute_buckets(self):
        from abstraction.equity import dual_equity_histogram, emd
        import numpy as np

        # Preflop: use scalar equity (fast, no sampling needed)
        for player in range(2):
            hole = self.hole0 if player == 0 else self.hole1
            self._buckets[(player, 0)] = self.abstraction._preflop_bucket(hole)

        # Flop + Turn: use dual histogram (compute both players together)
        for street, board in [(1, self.flop), (2, self.flop + self.turn)]:
            n_s = 80 if street == 1 else 100
            h0, h1 = dual_equity_histogram(self.hole0, self.hole1, board, n_samples=n_s)
            centroids = (self.abstraction._flop_centroids
                         if street == 1 else self.abstraction._turn_centroids)
            for player, hist in [(0, h0), (1, h1)]:
                dists = [emd(hist, c) for c in centroids]
                self._buckets[(player, street)] = int(np.argmin(dists))

        # River: exact strength percentile (fast, no sampling)
        river_board = self.flop + self.turn + self.river
        for player in range(2):
            hole = self.hole0 if player == 0 else self.hole1
            self._buckets[(player, 3)] = self.abstraction._river_bucket(hole, river_board)

    def showdown(self, traverser: int, pot: Pot) -> float:
        my_hole  = self.hole0 if traverser == 0 else self.hole1
        opp_hole = self.hole1 if traverser == 0 else self.hole0
        board    = self.board(3)
        my  = evaluator.evaluate(board, my_hole)
        opp = evaluator.evaluate(board, opp_hole)
        if my < opp:   return pot.winnings(traverser)
        elif my > opp: return pot.loss(traverser)
        else:          return 0.0


# ─────────────────────────────────────────────────────────────
# MCCFR Node
# ─────────────────────────────────────────────────────────────

class AbstrNode:
    __slots__ = ('regrets', 'strat_sum')

    def __init__(self):
        self.regrets   = defaultdict(float)
        self.strat_sum = defaultdict(float)

    def strategy(self, actions: list[str]) -> dict[str, float]:
        pos = {a: max(self.regrets[a], 0.0) for a in actions}
        t   = sum(pos.values())
        return {a: pos[a] / t for a in actions} if t > 0 else {a: 1/len(actions) for a in actions}

    def avg_strategy(self, actions: list[str]) -> dict[str, float]:
        t = sum(self.strat_sum[a] for a in actions)
        return {a: self.strat_sum[a] / t for a in actions} if t > 0 else {a: 1/len(actions) for a in actions}

    def sample(self, strat: dict) -> str:
        ks = list(strat.keys())
        return random.choices(ks, weights=[strat[k] for k in ks])[0]


# ─────────────────────────────────────────────────────────────
# Abstracted MCCFR Solver
# ─────────────────────────────────────────────────────────────

class AbstractedMCCFR:

    def __init__(self, abstraction: CardAbstraction):
        self.abstraction = abstraction
        self.nodes: dict[str, AbstrNode] = {}
        self.iterations = 0

    def _node(self, key: str) -> AbstrNode:
        if key not in self.nodes:
            self.nodes[key] = AbstrNode()
        return self.nodes[key]

    def _key(self, player: int, bucket: int, street: int, hist_str: str) -> str:
        return f"{player}|b{bucket}|s{street}|{hist_str}"

    def _apply_action(self, a: str, player: int, pot: Pot) -> Pot:
        new_pot = pot.copy()
        if a == 'c':
            new_pot.call(player)
        elif is_bet(a):
            new_pot.bet(player, BET_FRACS[a])
        return new_pot

    def traverse(
        self,
        ctx: HandContext,
        street: int,
        hist_parts: list[str],
        pot: Pot,
        traverser: int,
    ) -> float:
        player = len(hist_parts) % 2

        # ── Fold terminal ──────────────────────────────────────
        if hist_parts and hist_parts[-1] == 'f':
            folder = (len(hist_parts) - 1) % 2
            return pot.loss(traverser) if folder == traverser else pot.winnings(traverser)

        # ── Street over ────────────────────────────────────────
        if street_over(hist_parts):
            if street == 3:
                return ctx.showdown(traverser, pot)
            return self.traverse(ctx, street + 1, [], pot, traverser)

        # ── Decision node ──────────────────────────────────────
        n_raises = count_raises(hist_parts)
        actions  = valid_actions(n_raises)

        bucket = ctx.bucket(player, street)
        key    = self._key(player, bucket, street, '_'.join(hist_parts))
        node   = self._node(key)
        strat  = node.strategy(actions)

        if player == traverser:
            vals = {}
            ev   = 0.0
            for a in actions:
                new_pot = self._apply_action(a, player, pot)
                v = self.traverse(ctx, street, hist_parts + [a], new_pot, traverser)
                vals[a] = v
                ev += strat[a] * v
            for a in actions:
                node.regrets[a]   += vals[a] - ev
                node.strat_sum[a] += strat[a]
            return ev
        else:
            a = node.sample(strat)
            for act in actions:
                node.strat_sum[act] += strat[act]
            new_pot = self._apply_action(a, player, pot)
            return self.traverse(ctx, street, hist_parts + [a], new_pot, traverser)

    # ── Training ──────────────────────────────────────────────

    def train(self, iterations: int = 10_000, checkpoint_every: int = 1000):
        t0 = time.time()
        total_ev = 0.0
        checkpoints = []

        for i in range(iterations):
            sample       = random.sample(ALL_CARDS, 4)
            hole0, hole1 = sample[:2], sample[2:]

            # HandContext precomputes ALL street buckets (the expensive part)
            ctx       = HandContext(hole0, hole1, self.abstraction)
            pot       = Pot()
            traverser = i % 2

            ev = self.traverse(ctx, 0, [], pot, traverser)
            total_ev += ev if traverser == 0 else -ev

            if (i + 1) % checkpoint_every == 0:
                avg     = total_ev / (i + 1)
                elapsed = time.time() - t0
                print(f"  Iter {i+1:6d} | Avg EV(P0): {avg:+.4f} | "
                      f"Nodes: {len(self.nodes):,} | {elapsed:.1f}s")
                checkpoints.append({'iter': i+1, 'ev': avg, 'nodes': len(self.nodes)})

        self.iterations = iterations
        return checkpoints

    # ── Output ────────────────────────────────────────────────

    def get_strategy_summary(self, top_n: int = 30) -> list[dict]:
        results = []
        for key, node in self.nodes.items():
            parts   = key.split('|')
            player, bucket, street_s, hist = parts[0], parts[1], parts[2], parts[3]
            hist_parts = [p for p in hist.split('_') if p]
            n_r    = count_raises(hist_parts)
            actions = valid_actions(n_r)
            avg    = node.avg_strategy(actions)
            results.append({
                'info_set': key,
                'player': player, 'bucket': bucket, 'street': street_s,
                'history': hist,
                **{a: round(avg.get(a, 0.0), 4) for a in ALL_ACTIONS},
                'visits': sum(node.strat_sum.values()),
            })
        results.sort(key=lambda x: -x['visits'])
        return results[:top_n]

    def save_strategy(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        data = {'iterations': self.iterations, 'strategy': self.get_strategy_summary(200)}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[Solver] Strategy saved → {path}")
