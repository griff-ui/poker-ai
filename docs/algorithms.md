---
layout: default
---

# Algorithm Reference

Detailed explanations of every algorithm implemented in this codebase, with the math and the intuition side by side.

---

## Counterfactual Regret Minimization (CFR)

### The problem

Poker is an imperfect information game — you can't see your opponent's cards. This makes it fundamentally different from chess or Go, where the full game state is visible to both players. Standard minimax search doesn't apply.

The solution is to reason over *information sets* — the set of all game states a player can't distinguish given what they've observed. A player's strategy maps each information set to a probability distribution over actions.

### Regret

After a game is played, the *regret* for an action is how much better you would have done if you'd always taken that action at that information set, given everything else stayed the same.

Formally, for player *i* at information set *I* after *T* iterations:
R_i^T(I, a) = Σ_{t=1}^{T} [v_i^t(a, I) - v_i^t(σ^t, I)]
Where:
- `v_i^t(a, I)` = counterfactual value of always taking action *a* at *I* in iteration *t*
- `v_i^t(σ^t, I)` = counterfactual value of the current strategy at *I* in iteration *t*
- "Counterfactual" means weighted by the probability of reaching *I* through opponents' and chance's actions, but as if player *i* always played to reach *I*

### Regret matching

The strategy for the next iteration is proportional to positive cumulative regret:
σ^{T+1}(I, a) = max(R^T(I, a), 0) / Σ_a max(R^T(I, a), 0)
If all cumulative regrets are zero or negative, play uniformly.

### Why it works

**Theorem (Zinkevich et al. 2007):** In a two-player zero-sum game, if both players independently minimize regret, the time-average of their strategies converges to a Nash equilibrium.

The key property is that regret minimization is *self-play compatible* — each player can independently minimize their own regret, and the resulting pair of average strategies is an equilibrium. No coordination is required.

### CFR+ variant

Standard CFR allows cumulative regrets to go negative. CFR+ floors them at zero each iteration:
R_+^{T+1}(I, a) = max(R_+^T(I, a) + [new regret], 0)
This converges roughly 2× faster in practice and is what most production systems use.

---

## Monte Carlo CFR (MCCFR)

### The problem with vanilla CFR

Vanilla CFR traverses the entire game tree on every iteration. For Leduc Hold'em this is fast (216 nodes). For full NLHE it's impossible (~10^160 states).

MCCFR samples a subset of the tree each iteration, trading variance for speed.

### External sampling

The variant implemented here. On each iteration, for a designated traverser *i*:

- **Traverser nodes:** Explore all actions (like vanilla CFR)
- **Opponent nodes:** Sample one action from their current strategy
- **Chance nodes:** Sample one outcome

This gives an unbiased estimate of the counterfactual values while dramatically reducing the number of nodes visited per iteration.

The convergence rate is:
ε = O(√(|A|·|I| / T))
Where |A| is the number of actions, |I| is the number of information sets, and T is the number of iterations. Slower than vanilla CFR per iteration but much faster wall-clock because each iteration is cheaper.

### Linear CFR weighting

Standard CFR weights all iterations equally in the average strategy. Linear CFR weights iteration *t* by *t* itself:
σ_avg(I, a) = Σ_t t · σ^t(I, a) / Σ_t t
This upweights more recent (better) strategies and converges approximately 2× faster. It's what this codebase uses in Stage 4.

---

## Card Abstraction

### Why abstraction is necessary

Full NLHE has 1,755 distinct preflop hand combinations, and the number of distinct (hand, board) combinations on the flop is roughly 1.3 billion. Storing a strategy entry for each one is intractable.

Card abstraction groups similar hands into buckets. The solver works on the abstracted game, and the resulting strategy is mapped back to real hands at play time.

### Earth Mover's Distance

The key insight is that two hands with the *same average equity* can be strategically very different — a made flush and a flush draw have similar equity against a random range, but the flush draw has high variance while the made flush has low variance.

Earth Mover's Distance (EMD) captures this by comparing the full *distribution* of equity over future runouts, not just the mean.

For two equity histograms A and B over bins [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]:
EMD(A, B) = Σ_i |CDF_A(i) - CDF_B(i)|
Where CDF is the cumulative distribution function. This is the Wasserstein-1 distance for 1D distributions and is efficient to compute.

### k-means with EMD

Standard k-means uses Euclidean distance. We replace the distance metric with EMD:

1. Initialize k centroids (via k-means++ for better starting points)
2. Assign each hand to the nearest centroid by EMD
3. Update centroids to the mean histogram of assigned hands
4. Repeat until convergence
5. Sort clusters by mean equity (bucket 0 = weakest)

### Bucket counts

| Street | Hands | Buckets | Basis |
|---|---|---|---|
| Preflop | 169 canonical | 8 | Scalar equity percentile |
| Flop | ~1.3B | 12 | EMD histogram clustering |
| Turn | ~55M | 12 | EMD histogram clustering |
| River | ~2.4M | 8 | Exact strength percentile |

---

## Deep CFR

### Why neural networks

Even with card abstraction, the abstracted game tree has too many information sets to store in a table for a full training run. Deep CFR (Brown et al. 2019) replaces the table with neural networks that generalize across similar states.

### Two networks per player

**AdvantageNetwork:** Approximates the cumulative counterfactual advantage (regret) for each action at each information set.
AdvNet(features(I)) ≈ [Adv(I, a1), Adv(I, a2), ..., Adv(I, ak)]
Current strategy is derived via regret matching on the output:
σ(I, a) = max(AdvNet(I)[a], 0) / Σ_a max(AdvNet(I)[a], 0)
**StrategyNetwork:** Approximates the average strategy (the one that converges to Nash):
StratNet(features(I)) ≈ σ_avg(I)
Trained with cross-entropy loss against the accumulated average strategy.

### Reservoir buffers

To train the networks, we need data from all past iterations. Reservoir sampling maintains a fixed-size buffer where every sample seen has equal probability of being retained:

```python
if len(buffer) < capacity:
    buffer.append(sample)
else:
    idx = random.randint(0, n_seen - 1)
    if idx < capacity:
        buffer[idx] = sample
```

### Feature encoding (373 dimensions)
[0:104]    Hole cards:   2 cards × 52 one-hot

[104:364]  Board cards:  5 slots × 52 one-hot (zero-padded pre-river)

[364:368]  Street:       4-dim one-hot (preflop/flop/turn/river)

[368]      Pot / starting stack

[369]      My stack / starting stack

[370]      Opponent stack / starting stack

[371]      To call / starting stack

[372]      Raises this street / max raises
---

## Real-Time Search (Stage 5)

### The blueprint problem

The blueprint strategy trained with Deep CFR uses coarse abstractions and can't adapt to opponent tendencies observed during play. It's a good prior but not the final word.

### Depth-limited subgame solving

At each decision point, instead of querying the blueprint directly, we:

1. Build a local game tree rooted at the current state
2. Run MCCFR within that tree for a fixed time budget
3. At leaf nodes (beyond depth limit), query the blueprint for value estimates
4. Return the average strategy from the root node

The depth limit (1 street by default) keeps decision time tractable — typically 75–100ms on CPU.

### Blueprint bootstrapping

Early in the search (few iterations), the local regret estimates are noisy. We blend local and blueprint strategies:
σ_blended(I, a) = (1 - α) · σ_blueprint(I, a) + α · σ_local(I, a)
Where α increases from 0 to 1 as iterations accumulate. This gives the search a warm start and stabilizes early decisions.

### Why search beats blueprint alone

The blueprint captures global strategy structure — which hands to continue with, which ranges to build. Search captures local precision — exactly how much to bet given this specific board, pot size, and stack depth. The combination is much stronger than either alone.

Pluribus showed this is the critical ingredient: even a moderately trained blueprint combined with strong real-time search outperforms a well-trained blueprint without search.
