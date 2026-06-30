# Architecture

A from-scratch implementation of a complete game-theoretically optimal (GTO) poker AI, built across five progressive stages — from foundational CFR to real-time search — following the architecture of Pluribus (Brown & Sandholm, *Science*, 2019).

Each stage exists to solve a specific limitation of the one before it. None of the five are optional shortcuts — each is a real bottleneck in scaling CFR from toy games to full No-Limit Hold'em.

```
Stage 1+2  Vanilla CFR + MCCFR      Leduc Hold'em (6-card toy game)
    ↓
Stage 3    Card Abstraction          k-means with Earth Mover's Distance
    ↓                                8 preflop / 12 flop / 12 turn / 8 river buckets
Stage 4    Deep CFR                  Neural network approximation of regret
    ↓                                Reservoir buffers + linear CFR weighting
Stage 5    Real-Time Search          Depth-limited subgame solving
                                     Blueprint oracle at leaf nodes
```

---

## Stage 1+2: Counterfactual Regret Minimization

Vanilla CFR and Monte Carlo CFR (external sampling), implemented on Leduc Hold'em — a tractable 2-player, 6-card game that is the standard research testbed for poker AI algorithms.

**Core idea:** at each information set, compute counterfactual regret for each action not taken. Strategy at the next iteration is proportional to cumulative positive regret. The *average* strategy across iterations — not the final one — converges to Nash equilibrium.

MCCFR replaces full game-tree traversal with sampling, which is what makes the approach computationally tractable at all beyond toy games. This is the bottleneck Stage 1+2 solves: **compute**.

---

## Stage 3: Card Abstraction

Full No-Limit Hold'em has roughly 10¹⁶⁰ game states — far beyond anything that can be enumerated directly. Card abstraction groups strategically similar hands into buckets so the resulting game tree becomes solvable.

- **Preflop:** Monte Carlo equity estimation → 8 percentile buckets
- **Flop / Turn:** equity histogram over future runouts → k-means clustering with Earth Mover's Distance (EMD) → 12 clusters each
- **River:** exact hand strength percentile → 8 equal-width bins

EMD clustering groups hands by the *distribution* of equity across possible runouts, not just the mean — which is what distinguishes a made hand from a draw with an identical average equity. This is the bottleneck Stage 3 solves: **state-space scale**.

---

## Stage 4: Deep CFR

Scales CFR to full NLHE by replacing the tabular strategy/regret store with neural networks that generalize across similar, even previously-unseen, game states.

- **AdvantageNetwork** — maps a 373-dimensional information-set feature vector to per-action advantage estimates. Regret matching on the output gives the current strategy.
- **StrategyNetwork** — maps the same features to a probability distribution over actions, tracking the *average* strategy, which is what converges to Nash equilibrium.
- **Reservoir buffers** — uniform random replacement so all training iterations remain equally represented, preventing catastrophic forgetting of earlier training data.
- **Linear CFR weighting** — samples from iteration *t* weighted by *t*, converging roughly 2× faster than vanilla CFR.
- **Feature encoding** — 104-dim hole cards + 260-dim board representation + 4-dim street indicator + 5 normalized scalar features.

This is the bottleneck Stage 4 solves: **generalization** — moving from a strategy table that only knows states it has explicitly seen, to a function that can estimate strategy for any state.

---

## Stage 5: Real-Time Search

The technique that distinguishes Pluribus from earlier poker AI systems. At each decision point, the current subgame is re-solved in real time, using the Stage 4 blueprint network as a value estimator at leaf nodes rather than relying purely on the pre-trained strategy.

- **Depth-limited search** — searches one street ahead, with the blueprint acting as an oracle beyond that depth
- **Blueprint bootstrapping** — blends local search regrets with the blueprint strategy, weighted proportionally to search iterations elapsed, which stabilizes early search before local regret estimates are reliable
- **75ms average decision time** on CPU in this implementation

This is the bottleneck Stage 5 solves: **blind spots** — a pre-trained blueprint strategy that hasn't seen a specific live situation in enough depth gets sharpened against what's actually happening in that subgame, rather than falling back on a coarser pre-computed approximation.

---

## How this compares to Pluribus

| Property | This repo | Pluribus |
|---|---|---|
| Algorithm | Deep CFR + depth-limited search | Blueprint CFR + subgame solving |
| Players | 2 (heads-up) | 6 |
| Bet abstraction | 5 sizes | 14 sizes |
| Traversals | ~50k (demo) | 12,400 × 1,000 iterations |
| Hardware | Single CPU | 64-core CPU |
| Training time | ~30 min | ~8 days |

The architecture is faithful to the published papers; scale is the main difference. This is a heads-up (2-player) implementation — multi-way extension is a real, unsolved engineering challenge in this codebase, not a small extension of the current code.

---

## Project structure

```
poker_ai/
├── main.py                     # Stage 1+2 entry point
├── train_abstracted.py         # Stage 3 entry point
│
├── cfr/
│   └── leduc_cfr.py            # Vanilla CFR for Leduc Hold'em
├── mccfr/
│   └── leduc_mccfr.py          # External sampling MCCFR
├── abstraction/
│   ├── equity.py               # MC equity, histograms, EMD
│   ├── card_abstraction.py     # Multi-street k-means clustering
│   └── abstracted_cfr.py       # MCCFR on abstracted game
├── deep_cfr/
│   ├── game_engine.py          # Full NLHE: stacks, all-ins, features
│   ├── networks.py             # AdvantageNet, StrategyNet, buffers
│   ├── traversal.py            # Deep CFR external sampling traversal
│   └── run_convergence.py      # Tight convergence run
└── stage5/
    ├── search.py                # Subgame solver, blueprint oracle
    └── evaluate.py               # Tournament evaluation framework
```

See [Algorithms](algorithms.md) for implementation-level detail on the math behind each stage, and [Results](results.md) for full benchmark methodology.
