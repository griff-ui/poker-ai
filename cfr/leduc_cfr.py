"""
Vanilla CFR for Leduc Hold'em with correct terminal detection.
"""
import random
from utils.game_utils import LEDUC_DECK, leduc_winner

ANTE = 1
BET_SIZES = {0: 2, 1: 4}
MAX_RAISES = 2


def is_round_over(cur):
    if not cur: return False
    if cur[-1] == 'f': return True
    if cur[-1] == 'c':
        if 'r' in cur: return True  # call after a bet
        if cur == 'cc': return True  # check-check
    return False


class CFRNode:
    def __init__(self, n):
        self.n = n
        self.regrets = [0.0] * n
        self.strat_sum = [0.0] * n

    def strategy(self, reach):
        pos = [max(r, 0.0) for r in self.regrets]
        t = sum(pos)
        s = [p / t for p in pos] if t > 0 else [1.0 / self.n] * self.n
        for i in range(self.n):
            self.strat_sum[i] += reach * s[i]
        return s

    def avg_strategy(self):
        t = sum(self.strat_sum)
        return [x / t for x in self.strat_sum] if t > 0 else [1.0 / self.n] * self.n


class LeducCFR:
    def __init__(self):
        self.nodes = {}
        self.iterations = 0

    def _node(self, key, n):
        if key not in self.nodes:
            self.nodes[key] = CFRNode(n)
        return self.nodes[key]

    def _compute_bets(self, rounds):
        bets = [float(ANTE), float(ANTE)]
        for r_idx, rnd in enumerate(rounds):
            for j, a in enumerate(rnd):
                p = j % 2
                if a == 'r':
                    bets[p] += BET_SIZES[r_idx]
                elif a == 'c' and 'r' in rnd[:j]:
                    diff = max(bets[1-p] - bets[p], 0)
                    bets[p] += diff
        return bets

    def _cfr(self, p0, p1, board, history, p0r, p1r):
        rounds = history.split('|')
        r_idx = len(rounds) - 1
        cur = rounds[-1]
        player = len(cur) % 2

        # Fold terminal
        if cur and cur[-1] == 'f':
            bets = self._compute_bets(rounds)
            folder = (len(cur) - 1) % 2
            return -bets[0] if folder == 0 else bets[1]

        # Round over
        if is_round_over(cur):
            if r_idx == 0:
                return self._cfr(p0, p1, board, history + '|', p0r, p1r)
            else:
                bets = self._compute_bets(rounds)
                w = leduc_winner(p0, p1, board)
                if w == 0:   return bets[1]
                elif w == 1: return -bets[0]
                else:        return 0.0

        # Decision node
        rc = cur.count('r')
        actions = ['f', 'c'] if rc >= MAX_RAISES else ['f', 'c', 'r']
        n = len(actions)

        hole = p0 if player == 0 else p1
        vis_board = board if r_idx == 1 else None
        key = f"{player}|{hole}|{vis_board or '?'}|{cur}"
        node = self._node(key, n)
        strat = node.strategy(p0r if player == 0 else p1r)

        act_vals = []
        node_val = 0.0

        for i, a in enumerate(actions):
            rounds_copy = list(rounds)
            rounds_copy[-1] = cur + a
            nh = '|'.join(rounds_copy)
            new_p0r = p0r * strat[i] if player == 0 else p0r
            new_p1r = p1r * strat[i] if player == 1 else p1r
            v = self._cfr(p0, p1, board, nh, new_p0r, new_p1r)
            av = v if player == 0 else -v
            act_vals.append(av)
            node_val += strat[i] * av

        opp_reach = p1r if player == 0 else p0r
        for i in range(n):
            node.regrets[i] += opp_reach * (act_vals[i] - node_val)

        return node_val if player == 0 else -node_val

    def train(self, iterations=10_000):
        deck = LEDUC_DECK.copy()
        total = 0.0
        checkpoints = []
        for i in range(iterations):
            random.shuffle(deck)
            val = self._cfr(deck[0], deck[1], deck[2], '', 1.0, 1.0)
            total += val
            if (i + 1) % 1000 == 0:
                avg = total / (i + 1)
                checkpoints.append(avg)
                print(f"  Iter {i+1:6d} | Avg EV(P0): {avg:+.4f} | Nodes: {len(self.nodes)}")
        self.iterations = iterations
        return checkpoints

    def get_strategy_summary(self, top_n=20):
        results = []
        for key, node in self.nodes.items():
            avg = node.avg_strategy()
            results.append({
                'info_set': key,
                'fold':  round(avg[0], 4),
                'call':  round(avg[1], 4),
                'raise': round(avg[2], 4) if node.n > 2 else 0.0,
                'visits': sum(node.strat_sum),
            })
        results.sort(key=lambda x: -x['visits'])
        return results[:top_n]
