# Poker GTO Solver — Leduc Hold'em

A from-scratch implementation of two game-theoretically optimal (GTO) poker AI algorithms:

- **Vanilla CFR** — Counterfactual Regret Minimization
- **MCCFR** — Monte Carlo CFR with External Sampling

Built on Leduc Hold'em, the standard research testbed for poker AI.

---

## Quick Start

```bash
pip install treys
python main.py --iterations 50000 --mode both
```

---

## Game: Leduc Hold'em

- 2 players, 6-card deck (J/Q/K × 2 suits)
- Both ante 1 chip
- Round 1: each player dealt 1 hole card; bet size = 2, max 2 raises
- Round 2: 1 community card dealt face-up; bet size = 4, max 2 raises
- Best hand: pair (hole matches board) > high card; K > Q > J
- Actions: `f` = fold, `c` = check/call, `r` = raise/bet

---

## Project Structure

```
poker_ai/
├── main.py                  # Training runner + comparison
├── cfr/
│   └── leduc_cfr.py         # Vanilla CFR implementation
├── mccfr/
│   └── leduc_mccfr.py       # External Sampling MCCFR
├── utils/
│   └── game_utils.py        # Hand evaluation, deck, utilities
└── results/
    └── strategies.json      # Saved strategy tables
```

---

## Key Concepts

### Information Set Keys
Format: `{player}|{hole_card}|{board_card}|{betting_history}`

- `board_card` = `?` until flop is dealt
- `betting_history` = sequence of actions in current round (e.g., `cr` = check-raise)

### Strategy Output
Each information set has a probability distribution over actions:
- `fold`, `call`, `raise` (raise unavailable at max raises)

### What the EV Numbers Mean
- Near 0 = balanced/symmetric play (expected for GTO)
- Negative EV for P0 early = P1 has positional advantage (acts second in Leduc)

---

## Algorithm Comparison

| Property | CFR | MCCFR (External) |
|---|---|---|
| Tree traversal | Full | Sampled opponent + chance |
| Variance | Low | Higher |
| Speed | ~3x slower | Faster per iteration |
| Convergence | Steady | Noisier but same endpoint |
| Memory | Same | Same |

---

## Roadmap: Extending to Full NLHE

### Stage 3: Card Abstraction
Group hands with similar equity distributions into buckets using k-means + Earth Mover's Distance. This compresses the ~10^160 NLHE states into a solvable space.

### Stage 4: Bet Abstraction
Discretize bet sizes to a small set, e.g., `{fold, check/call, 33%, 67%, pot, all-in}`.

### Stage 5: Deep CFR
Replace the strategy table with neural networks that generalize across similar game states — the approach used in Pluribus (CMU/Facebook, 2019).

### Stage 6: Real-Time Search
Blueprint strategy trained offline + limited lookahead search at runtime. This is how Pluribus beat 6 professionals at No-Limit Hold'em.

---

## References

- Zinkevich et al. (2007) — "Regret Minimization in Games with Incomplete Information" (original CFR)
- Lanctot et al. (2009) — "Monte Carlo Sampling for Regret Minimization in Extensive Games" (MCCFR)
- Brown & Sandholm (2019) — "Solving Imperfect-Information Games via Discounted Regret Minimization" (CFR+)
- Brown & Sandholm (2019) — "Superhuman AI for multiplayer poker" (Pluribus)
- OpenSpiel: https://github.com/deepmind/open_spiel
