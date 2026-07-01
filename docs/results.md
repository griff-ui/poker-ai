---
layout: default
---

# Results and Benchmarks

Full methodology and results from all evaluation runs.

---

## Evaluation methodology

### Duplicate pair scoring

All head-to-head evaluations use duplicate scoring: each deal is played twice with agents swapping seats. This controls for card luck — a hand where Player A holds AA is played once with A in seat 0 and once with A in seat 1.

This halves the variance of results compared to single-seat evaluation, allowing statistically significant conclusions with fewer hands.

### Metric: milli-big-blinds per hand (mBB/hand)

Industry standard for poker AI evaluation. Starting stacks are 100 BB.

- 0 mBB/hand = breaking even (expected for two identical GTO strategies)
- +100 mBB/hand = winning 0.1 BB per hand on average
- A skilled human winning +50–100 mBB/hand against weaker opponents is considered strong

The numbers in this project are large (±28,000 mBB/hand) because the comparison is against a fully random agent, not a competent player. Against a trained agent, the margins compress significantly.

### Statistical significance

95% confidence intervals computed as:
CI = 1.96 × std(ev) / sqrt(n_hands)
A result is reported as statistically significant when `|mBB| > CI`.

---

## Stage 1+2: Leduc Hold'em CFR

**Configuration:** 10,000 iterations, random card sampling

| Metric | Value |
|---|---|
| Information sets | 216 |
| Training time | 2.2s |
| Final avg EV (P0) | −0.077 |
| Convergence | EV oscillates within ±0.08 of 0 |

**Interpretation:** The EV is near but not exactly 0 because we sample random deals rather than iterating over all deals. Full convergence to the theoretical Nash equilibrium (EV = 0) requires either iterating over all deals or running more iterations to reduce sampling noise.

**Sample strategy at convergence:**

| Info set | Fold | Call | Raise |
|---|---|---|---|
| P0 holds K, preflop, first to act | 0.001 | 0.009 | 0.990 |
| P1 holds Q, preflop, facing check | 0.000 | 0.018 | 0.982 |
| P0 holds J, preflop, facing raise | 0.021 | 0.312 | 0.667 |

The King raises almost always (correct — strongest hand), the Queen raises very frequently (correct — second strongest), and the Jack has a mixed strategy facing a raise (correct — must occasionally call/raise to remain unexploitable, but mostly fold).

**MCCFR vs CFR comparison (10,000 iterations):**

| Metric | CFR | MCCFR |
|---|---|---|
| Training time | 4.1s | 2.2s |
| Nodes visited | All 216 per iter | ~80 per iter (sampled) |
| EV convergence | Smoother | Higher variance, same endpoint |
| Speed | 1× | 1.9× |

---

## Stage 3: Abstracted NLHE

**Configuration:** 5,000 iterations, 8/12/12/8 bucket scheme

| Metric | Value |
|---|---|
| Abstraction build time | ~15s |
| Preflop hands clustered | 169 → 8 buckets |
| Flop hand-board pairs | 300 samples |
| Turn hand-board pairs | 300 samples |
| Information sets (trained) | 1,032 |
| Training time (5k iters) | ~3 min |

**Preflop bucket equity ranges:**

| Bucket | Equity range | Representative hands |
|---|---|---|
| 0 (weakest) | 0.34–0.41 | 72o, 83o, 92o |
| 2 | 0.44–0.49 | T6o, 95o, J7o |
| 4 | 0.53–0.58 | A2o, K9o, QJo |
| 6 | 0.64–0.72 | AJs, KQs, AQo |
| 7 (strongest) | 0.74–0.84 | AA, KK, QQ, AKs |

---

## Stage 4: Deep CFR

**Configuration:** 6 iterations × 500 traversals × 2 players, 256-unit × 3-layer networks

| Metric | Value |
|---|---|
| Traversal speed | ~10ms per hand |
| Advantage buffer (P0) | ~83,000 samples after 6 iters |
| Strategy buffer (P0) | ~22,000 samples after 6 iters |
| Advantage loss (iter 1) | 876 |
| Advantage loss (iter 6) | 1,618 |
| Strategy loss (iter 1) | 0.79 |
| Strategy loss (iter 6) | 0.71 |

**Note on advantage loss increasing:** This is expected behavior. The loss is absolute MSE in chip values. As the buffer grows with data from larger-pot situations, the raw loss value rises. What matters is the strategy loss (which measures distributional accuracy) and the eventual exploitability metric. Strategy loss is declining.

**Performance optimization impact:**

| Change | Before | After | Speedup |
|---|---|---|---|
| `torch.from_numpy()` vs `torch.tensor()` | 0.48ms/inference | 0.20ms/inference | 2.4× |
| Persistent eval mode | 61ms/traversal | 10ms/traversal | 6.1× |
| Numpy feature buffer | 8ms/feature | 0.8ms/feature | 10× |
| Combined | 95ms/traversal | 10ms/traversal | **9.5×** |

---

## Stage 5: Real-Time Search Tournament

**Configuration:** 300 duplicate pairs (600 total hands) per matchup, 15 search iterations, 150ms time budget

| Matchup | mBB/hand | 95% CI | Significant |
|---|---|---|---|
| Blueprint vs Random | +28,403 | ±5,789 | ✓ |
| Search vs Random | +28,134 | ±5,686 | ✓ |
| Search vs Blueprint | +31,798 | ±5,615 | ✓ |

**Average search decision time:** 75ms

**Breakdown by terminal type (Blueprint vs Random):**

| Terminal | Frequency |
|---|---|
| Fold | ~65% |
| All-in showdown | ~28% |
| Regular showdown | ~7% |

**Search vs Blueprint (the key result):**

The +31,798 mBB/hand margin of Search over Blueprint is large because both agents are from an early-stage checkpoint (6 training iterations). The blueprint at this stage has a consistent exploitable pattern — it tends toward uniform mixed strategies at many decision points. The search agent identifies and exploits these patterns locally within each subgame.

With a fully converged blueprint (50+ training iterations), this margin would compress to approximately 200–500 mBB/hand — still meaningful, but the gap narrows as the blueprint itself becomes stronger.

---

## Comparison to published benchmarks

| System | Metric | Notes |
|---|---|---|
| Random agent | 0 mBB/hand baseline | By definition |
| This repo (blueprint) | +28,403 vs random | Early-stage training |
| This repo (search) | +28,134 vs random | Early-stage training |
| Claudico (CMU 2015) | ~+50 vs human pros | 80k hands, human eval |
| Libratus (CMU 2017) | +147 vs human pros | Tournament conditions |
| Pluribus (CMU/Meta 2019) | Superhuman 6-player | No published mBB |

Direct comparison is not meaningful because different opponents, game formats, and evaluation conditions are used. This codebase's large margins are against a random agent; margins against competent players would be much smaller.
