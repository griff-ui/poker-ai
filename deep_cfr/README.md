# Stage 4: Deep CFR for No-Limit Hold'em

Full implementation of Deep CFR (Brown & Sandholm, NeurIPS 2019) applied
to heads-up No-Limit Hold'em with proper game mechanics.

---

## What's new in Stage 4

### Proper game engine (`deep_cfr/game_engine.py`)
- Real stack tracking: both players start with 100 BB, stacks decrease as chips go in
- Correct all-in terminal: when effective stack hits 0, betting stops immediately
- Effective stack bet sizing: bets capped at min(p0_stack, p1_stack)
- Clean terminal classification: FOLD | ALLIN_SHOWDOWN | SHOWDOWN
- Full 374-dim feature encoding for neural network input

### Deep CFR networks (`deep_cfr/networks.py`)
- AdvantageNetwork: approximates cumulative counterfactual regrets
  - Regret matching applied to output for strategy computation
- StrategyNetwork: approximates the average strategy (Nash approximation)
- Both: 3 hidden layers × 256 units, LayerNorm, ReLU
- Reservoir buffers: 500k capacity with uniform reservoir sampling

### External sampling traversal (`deep_cfr/traversal.py`)
- Traverser explores all actions, opponent samples one
- Linear CFR weighting: iteration t → weight t (recent data upweighted)
- Samples stored as (features, advantages, weight) tuples

---

## Quick start

```bash
# Stage 3 abstraction must be built first
python train_abstracted.py

# Then run Deep CFR
python deep_cfr/train_deep_cfr.py \
  --iterations 50 \
  --traversals 500 \
  --batch-size 512 \
  --adv-steps 300 \
  --strat-steps 200
```

For serious convergence (approaching Pluribus quality):
```bash
python deep_cfr/train_deep_cfr.py \
  --iterations 200 \
  --traversals 1000 \
  --batch-size 1024 \
  --adv-steps 500 \
  --strat-steps 300
```

---

## How Deep CFR works

Standard CFR stores regrets for every information set in a table.
In No-Limit Hold'em, there are ~10^160 information sets — a table is impossible.

Deep CFR solves this by:

1. **Neural approximation**: Instead of a table, use a neural network to
   approximate the advantage function Adv(I, a) at any information set.
   The network generalizes: similar hands in similar positions get similar strategies.

2. **Reservoir sampling**: Store (features, advantages, weight) tuples in a fixed
   buffer. When full, replace uniformly at random — ensuring all iterations are
   represented equally in training data (no catastrophic forgetting).

3. **Alternating traversal**: Each iteration, Player 0 traverses first (exploiting
   all actions), then Player 1. After each full iteration, train both networks on
   their respective buffers.

4. **Linear CFR weighting**: Sample from iteration t is weighted by t. This is
   the "linear CFR" variant which converges ~2× faster than vanilla.

---

## Reading the output

**EV fluctuating ±10-15 BB in early iterations**: Normal. The network starts
random (uniform strategy = 1/6 per action), so all raise options are explored
heavily, creating deep pots. As regrets accumulate, the network learns to fold
weak hands and only continue with equity.

**adv_loss increasing early**: Expected. More buffer data → network has more to
learn → loss grows before it shrinks. The metric that matters is whether the
**strategy network** produces lower exploitability over time.

**Convergence timeline** (approximate):
- 10 iters  × 300 traversals: strategy shape appears
- 50 iters  × 500 traversals: GTO structure visible (strong hands bet, weak fold)
- 200 iters × 1000 traversals: approaching publishable quality
- 1000 iters × 12,400 traversals: Pluribus-level (requires significant compute)

---

## Architecture diagram

```
For each iteration t:
  ┌─────────────────────────────────────────────┐
  │  For traverser ∈ {P0, P1}:                  │
  │    For each of N traversals:                 │
  │      Deal fresh hand (stacks, blinds)        │
  │      MCCFR traverse game tree:               │
  │        Traverser node → explore ALL actions  │
  │          Query AdvNet → regret_matching → σ  │
  │          Compute advantages a(I,a) = v(a)-EV │
  │          → Add to advantage reservoir buffer │
  │          → Add to strategy reservoir buffer  │
  │        Opponent node → SAMPLE one action     │
  │          Query AdvNet → σ → sample action    │
  │          → Add to strategy reservoir buffer  │
  │        Terminal → return payoff              │
  └─────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────┐
  │  Network training:                           │
  │    For P0, P1:                               │
  │      Sample batch from advantage buffer      │
  │      Train AdvNet: minimize weighted MSE     │
  │        loss = Σ t·(AdvNet(I) - target)²     │
  │      Sample batch from strategy buffer       │
  │      Train StratNet: minimize weighted CE    │
  │        loss = -Σ t·target·log(StratNet(I))  │
  └─────────────────────────────────────────────┘
```

---

## References

- Brown & Sandholm (2019) — "Deep Counterfactual Regret Minimization"
  https://arxiv.org/abs/1811.00164
- Brown & Sandholm (2019) — "Superhuman AI for multiplayer poker" (Pluribus)
  https://science.sciencemag.org/content/365/6456/885
- Zinkevich et al. (2007) — "Regret Minimization in Games" (original CFR)
