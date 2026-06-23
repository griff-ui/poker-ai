"""
Main training script: runs both CFR and MCCFR and compares results.

Usage:
    python main.py [--iterations N] [--mode cfr|mccfr|both]
"""

import sys
import os
import time
import json
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from cfr.leduc_cfr import LeducCFR
from mccfr.leduc_mccfr import LeducMCCFR


def print_strategy_table(strategies: list[dict], title: str):
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")
    print(f"  {'Information Set':<35} {'Fold':>8} {'Call':>8} {'Raise':>8}")
    print(f"  {'─'*35} {'─'*8} {'─'*8} {'─'*8}")
    for s in strategies:
        label = s['info_set']
        if len(label) > 34:
            label = label[:31] + '...'
        print(f"  {label:<35} {s['fold']:>8.3f} {s['call']:>8.3f} {s['raise']:>8.3f}")
    print(f"{'─'*70}")


def run_cfr(iterations: int):
    print(f"\n{'='*70}")
    print(f"  VANILLA CFR — {iterations:,} iterations")
    print(f"{'='*70}")

    solver = LeducCFR()
    t0 = time.time()
    checkpoints = solver.train(iterations)
    elapsed = time.time() - t0

    print(f"\n  ✓ Trained in {elapsed:.2f}s | {len(solver.nodes)} information sets")
    print(f"  Final avg game value (P0): {checkpoints[-1]:+.4f}")
    print("  (In a GTO solution, this should be near 0 in a symmetric game)")

    strategies = solver.get_strategy_summary(top_n=20)
    print_strategy_table(strategies, "CFR Average Strategy (top 20 information sets by visits)")

    return solver, checkpoints, elapsed


def run_mccfr(iterations: int):
    print(f"\n{'='*70}")
    print(f"  MONTE CARLO CFR (External Sampling) — {iterations:,} iterations")
    print(f"{'='*70}")

    solver = LeducMCCFR()
    t0 = time.time()
    checkpoints = solver.train(iterations)
    elapsed = time.time() - t0

    print(f"\n  ✓ Trained in {elapsed:.2f}s | {len(solver.nodes)} information sets")
    print(f"  Final avg game value (P0): {checkpoints[-1]:+.4f}")

    strategies = solver.get_strategy_summary(top_n=20)
    print_strategy_table(strategies, "MCCFR Average Strategy (top 20 information sets by visits)")

    return solver, checkpoints, elapsed


def compare_strategies(cfr_solver: LeducCFR, mccfr_solver: LeducMCCFR):
    """
    Compare strategies at key information sets between CFR and MCCFR.
    Focus on the most meaningful spots: first action with different hole cards.
    """
    print(f"\n{'='*70}")
    print("  STRATEGY COMPARISON: CFR vs MCCFR at key spots")
    print(f"{'='*70}")

    # Key preflop spots to compare (player, hole card, no board, no history)
    key_spots = [
        "P0|Js|?|",   # P0, Jack of spades, preflop, first to act
        "P0|Qs|?|",   # P0, Queen of spades
        "P0|Ks|?|",   # P0, King of spades
        "P1|Js|?|c",  # P1 facing a check
        "P1|Qs|?|c",
        "P1|Ks|?|c",
        "P0|Js|?|cr", # P0 facing a raise
        "P0|Qs|?|cr",
        "P0|Ks|?|cr",
    ]

    actions = ['f', 'c', 'r']

    print(f"\n  {'Spot':<25} {'CFR f/c/r':>20} {'MCCFR f/c/r':>20}")
    print(f"  {'─'*25} {'─'*20} {'─'*20}")

    for spot in key_spots:
        cfr_node = cfr_solver.nodes.get(spot)
        mccfr_node = mccfr_solver.nodes.get(spot)

        if cfr_node:
            cfr_strat = cfr_node.get_average_strategy()
            cfr_str = f"{cfr_strat[0]:.2f}/{cfr_strat[1]:.2f}/{cfr_strat[2]:.2f}"
        else:
            cfr_str = "not visited"

        if mccfr_node:
            valid = ['f', 'c', 'r'] if spot.count('r') < 2 else ['f', 'c']
            mccfr_strat = mccfr_node.get_average_strategy(valid)
            mccfr_str = f"{mccfr_strat.get('f',0):.2f}/{mccfr_strat.get('c',0):.2f}/{mccfr_strat.get('r',0):.2f}"
        else:
            mccfr_str = "not visited"

        print(f"  {spot:<25} {cfr_str:>20} {mccfr_str:>20}")

    print()


def save_results(cfr_solver, mccfr_solver, cfr_time, mccfr_time, iterations, path: str):
    """Save strategy tables to JSON for further analysis."""
    results = {
        'iterations': iterations,
        'cfr': {
            'time_seconds': round(cfr_time, 2),
            'num_info_sets': len(cfr_solver.nodes),
            'strategy': cfr_solver.get_strategy_summary(top_n=50)
        },
        'mccfr': {
            'time_seconds': round(mccfr_time, 2),
            'num_info_sets': len(mccfr_solver.nodes),
            'strategy': mccfr_solver.get_strategy_summary(top_n=50)
        }
    }
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description='Poker GTO Solver: CFR & MCCFR')
    parser.add_argument('--iterations', type=int, default=10_000,
                        help='Number of training iterations (default: 10,000)')
    parser.add_argument('--mode', choices=['cfr', 'mccfr', 'both'], default='both',
                        help='Which algorithm to run')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  POKER GTO SOLVER — Leduc Hold'em")
    print(f"  Algorithms: Vanilla CFR + Monte Carlo CFR (External Sampling)")
    print(f"  Iterations: {args.iterations:,}")
    print(f"{'='*70}")
    print("""
  Game rules:
    • 2 players, 6-card deck (J/Q/K × 2 suits)
    • Both ante 1 chip
    • Round 1 (preflop): bet size = 2, max 2 raises
    • Round 2 (flop): community card dealt, bet size = 4, max 2 raises
    • Best hand: pair > high card; K > Q > J
    • Actions: f=fold, c=check/call, r=raise/bet
""")

    cfr_solver = mccfr_solver = None
    cfr_time = mccfr_time = 0

    if args.mode in ('cfr', 'both'):
        cfr_solver, cfr_checkpoints, cfr_time = run_cfr(args.iterations)

    if args.mode in ('mccfr', 'both'):
        mccfr_solver, mccfr_checkpoints, mccfr_time = run_mccfr(args.iterations)

    if args.mode == 'both' and cfr_solver and mccfr_solver:
        compare_strategies(cfr_solver, mccfr_solver)

        print(f"\n{'='*70}")
        print("  PERFORMANCE SUMMARY")
        print(f"{'='*70}")
        print(f"  CFR:   {cfr_time:.2f}s | {len(cfr_solver.nodes)} info sets")
        print(f"  MCCFR: {mccfr_time:.2f}s | {len(mccfr_solver.nodes)} info sets")
        print(f"  MCCFR speedup: {cfr_time/mccfr_time:.1f}x")
        print()

        os.makedirs('results', exist_ok=True)
        save_results(cfr_solver, mccfr_solver, cfr_time, mccfr_time,
                     args.iterations, 'results/strategies.json')

    print(f"\n  Done. Next steps:")
    print(f"  1. Increase --iterations to 100,000+ for tighter convergence")
    print(f"  2. Review results/strategies.json for full strategy tables")
    print(f"  3. See README.md for how to extend to full NLHE with abstraction\n")


if __name__ == '__main__':
    main()
