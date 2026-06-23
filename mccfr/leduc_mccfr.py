"""
Monte Carlo CFR (External Sampling) for Leduc Hold'em.
"""
import random
from collections import defaultdict
from utils.game_utils import LEDUC_DECK, leduc_winner

ANTE = 1
BET_SIZES = {0: 2, 1: 4}
MAX_RAISES = 2


def is_round_over(cur):
    if not cur: return False
    if cur[-1] == 'f': return True
    if cur[-1] == 'c':
        if 'r' in cur: return True
        if cur == 'cc': return True
    return False


class MCCFRNode:
    def __init__(self):
        self.regrets = defaultdict(float)
        self.strat_sum = defaultdict(float)

    def strategy(self, actions):
        pos = {a: max(self.regrets[a], 0.0) for a in actions}
        t = sum(pos.values())
        return {a: pos[a]/t for a in actions} if t > 0 else {a: 1/len(actions) for a in actions}

    def avg_strategy(self, actions):
        t = sum(self.strat_sum[a] for a in actions)
        return {a: self.strat_sum[a]/t for a in actions} if t > 0 else {a: 1/len(actions) for a in actions}

    def sample(self, strat):
        ks, vs = zip(*strat.items())
        return random.choices(list(ks), weights=list(vs))[0]


class LeducMCCFR:
    def __init__(self):
        self.nodes = {}
        self.iterations = 0

    def _node(self, key):
        if key not in self.nodes:
            self.nodes[key] = MCCFRNode()
        return self.nodes[key]

    def _compute_bets(self, rounds):
        bets = [float(ANTE), float(ANTE)]
        for r_idx, rnd in enumerate(rounds):
            for j, a in enumerate(rnd):
                p = j % 2
                if a == 'r':
                    bets[p] += BET_SIZES[r_idx]
                elif a == 'c' and 'r' in rnd[:j]:
                    bets[p] += max(bets[1-p] - bets[p], 0)
        return bets

    def _traverse(self, p0, p1, board, history, traverser):
        rounds = history.split('|')
        r_idx = len(rounds) - 1
        cur = rounds[-1]
        player = len(cur) % 2

        if cur and cur[-1] == 'f':
            bets = self._compute_bets(rounds)
            folder = (len(cur) - 1) % 2
            if folder == traverser:
                return -bets[traverser]
            return bets[1 - traverser]

        if is_round_over(cur):
            if r_idx == 0:
                return self._traverse(p0, p1, board, history + '|', traverser)
            else:
                bets = self._compute_bets(rounds)
                w = leduc_winner(p0, p1, board)
                if w == traverser: return bets[1 - traverser]
                elif w == (1 - traverser): return -bets[traverser]
                return 0.0

        rc = cur.count('r')
        actions = ['f', 'c'] if rc >= MAX_RAISES else ['f', 'c', 'r']

        hole = p0 if player == 0 else p1
        vis_board = board if r_idx == 1 else None
        key = f"{player}|{hole}|{vis_board or '?'}|{cur}"
        node = self._node(key)
        strat = node.strategy(actions)

        rounds_list = list(rounds)

        if player == traverser:
            vals = {}
            ev = 0.0
            for a in actions:
                rounds_list[-1] = cur + a
                nh = '|'.join(rounds_list)
                rounds_list[-1] = cur
                v = self._traverse(p0, p1, board, nh, traverser)
                vals[a] = v
                ev += strat[a] * v
            for a in actions:
                node.regrets[a] += vals[a] - ev
                node.strat_sum[a] += strat[a]
            return ev
        else:
            a = node.sample(strat)
            for act in actions:
                node.strat_sum[act] += strat[act]
            rounds_list[-1] = cur + a
            nh = '|'.join(rounds_list)
            return self._traverse(p0, p1, board, nh, traverser)

    def train(self, iterations=10_000):
        deck = LEDUC_DECK.copy()
        total = 0.0
        checkpoints = []
        for i in range(iterations):
            random.shuffle(deck)
            traverser = i % 2
            val = self._traverse(deck[0], deck[1], deck[2], '', traverser)
            total += val if traverser == 0 else -val
            if (i + 1) % 1000 == 0:
                avg = total / (i + 1)
                checkpoints.append(avg)
                print(f"  Iter {i+1:6d} | Avg EV(P0): {avg:+.4f} | Nodes: {len(self.nodes)}")
        self.iterations = iterations
        return checkpoints

    def get_strategy_summary(self, top_n=20):
        results = []
        for key, node in self.nodes.items():
            rc = key.split('|')[3].count('r')
            actions = ['f', 'c'] if rc >= MAX_RAISES else ['f', 'c', 'r']
            avg = node.avg_strategy(actions)
            results.append({
                'info_set': key,
                'fold':  round(avg.get('f', 0), 4),
                'call':  round(avg.get('c', 0), 4),
                'raise': round(avg.get('r', 0), 4),
                'visits': sum(node.strat_sum.values()),
            })
        results.sort(key=lambda x: -x['visits'])
        return results[:top_n]
