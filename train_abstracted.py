"""
Stage 3 training script: Card Abstraction + MCCFR for NLHE.

Steps:
  1. Build (or load) card abstraction
  2. Run Abstracted MCCFR
  3. Save strategy + convergence data
  4. Print readable strategy summary

Usage:
    python train_abstracted.py [--iterations N] [--rebuild]
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from abstraction.card_abstraction import CardAbstraction
from abstraction.abstracted_cfr import AbstractedMCCFR


def print_strategy_table(strategies: list[dict], title: str):
    print(f"\n{'─'*80}")
    print(f"  {title}")
    print(f"{'─'*80}")
    print(f"  {'Info Set':<40} {'fold':>6} {'call':>6} {'r1':>6} {'r2':>6} {'r3':>6}")
    print(f"  {'─'*40} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
    for s in strategies[:20]:
        label = s['info_set']
        if len(label) > 39:
            label = label[:36] + '...'
        print(f"  {label:<40} {s.get('f',0):>6.3f} {s.get('c',0):>6.3f} "
              f"{s.get('r1',0):>6.3f} {s.get('r2',0):>6.3f} {s.get('r3',0):>6.3f}")
    print(f"{'─'*80}")


def main():
    parser = argparse.ArgumentParser(description='Stage 3: Abstracted NLHE MCCFR')
    parser.add_argument('--iterations', type=int, default=20_000)
    parser.add_argument('--rebuild', action='store_true',
                        help='Rebuild card abstraction even if cache exists')
    parser.add_argument('--pre-samples', type=int, default=800,
                        help='Monte Carlo samples per preflop hand')
    parser.add_argument('--flop-hands', type=int, default=400,
                        help='Number of hand-board pairs for flop clustering')
    parser.add_argument('--turn-hands', type=int, default=400,
                        help='Number of hand-board pairs for turn clustering')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  STAGE 3: ABSTRACTED NLHE MCCFR")
    print(f"  Card Abstraction + External Sampling MCCFR")
    print(f"  Iterations: {args.iterations:,}")
    print(f"{'='*70}")
    print("""
  Architecture:
    Preflop  → Monte Carlo equity → 8 buckets
    Flop     → Equity histogram (EMD) → 12 clusters
    Turn     → Equity histogram (EMD) → 12 clusters
    River    → Exact strength percentile → 8 buckets
    Betting  → {fold, call, 33%pot, 67%pot, pot}
""")

    # ── Step 1: Card Abstraction ────────────────────────────────
    print(f"{'─'*70}")
    print("  STEP 1: Card Abstraction")
    print(f"{'─'*70}")

    abstraction = CardAbstraction()

    if not args.rebuild and abstraction.load():
        print("  Using cached abstraction.")
    else:
        print("  Building from scratch...")
        t0 = time.time()
        abstraction.build(
            n_pre_samples=args.pre_samples,
            n_flop_samples=150,
            n_turn_samples=200,
            n_flop_hands=args.flop_hands,
            n_turn_hands=args.turn_hands,
            verbose=True,
        )
        print(f"  Abstraction built in {time.time()-t0:.1f}s")

    # ── Step 2: MCCFR Training ──────────────────────────────────
    print(f"\n{'─'*70}")
    print("  STEP 2: Abstracted MCCFR Training")
    print(f"{'─'*70}\n")

    solver = AbstractedMCCFR(abstraction)
    checkpoints = solver.train(
        iterations=args.iterations,
        checkpoint_every=max(1, args.iterations // 20),
    )

    # ── Step 3: Results ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  RESULTS")
    print(f"{'='*70}")
    print(f"  Total info sets discovered: {len(solver.nodes):,}")
    print(f"  Final avg EV (P0): {checkpoints[-1]['ev']:+.4f}")

    strategies = solver.get_strategy_summary(top_n=40)
    print_strategy_table(strategies, "Top information sets by visits")

    # Save
    os.makedirs('results', exist_ok=True)
    solver.save_strategy('results/abstracted_strategy.json')

    # Convergence plot data
    print(f"\n  Convergence (EV over training):")
    for cp in checkpoints:
        bar = '█' * int(abs(cp['ev']) * 20)
        sign = '+' if cp['ev'] >= 0 else '-'
        print(f"    {cp['iter']:6d}: {cp['ev']:+.4f} {bar}")

    print(f"\n  Done. Strategy written to results/abstracted_strategy.json")
    print(f"  Next: increase iterations and bucket counts for tighter convergence.\n")


if __name__ == '__main__':
    main()
