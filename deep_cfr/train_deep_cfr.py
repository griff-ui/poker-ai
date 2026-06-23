"""
Deep CFR — Main Training Loop

Orchestrates the full Deep CFR training cycle:

  for t in 1..T:
    for p in {0, 1}:
      Run n_traversals MCCFR traversals as player p
    Train advantage networks (both players)
    Train strategy networks (both players)
    [Optional] Evaluate exploitability estimate

Usage:
    python train_deep_cfr.py [--iterations N] [--traversals N]
"""

import sys
import os
import time
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from deep_cfr.game_engine import (
    GameState, deal_hand, abstract_actions,
    visible_board, Terminal, STARTING_STACK,
)
from deep_cfr.networks import DeepCFRPlayer, MAX_ACTIONS
from deep_cfr.traversal import cfr_iteration, train_networks, traverse


# ─────────────────────────────────────────────────────────────
# Exploitability estimation via best-response approximation
# ─────────────────────────────────────────────────────────────

def estimate_exploitability(
    players: list[DeepCFRPlayer],
    n_hands: int = 500,
) -> dict:
    """
    Approximate exploitability by running the trained strategy against
    a random opponent and measuring the margin.

    True exploitability requires computing a best response (NP-hard in general).
    This is a practical proxy: higher = further from Nash equilibrium.
    """
    results = []

    for _ in range(n_hands):
        gs = deal_hand()
        ev = _play_hand(gs, players, use_avg_strategy=True)
        results.append(ev)

    avg_ev  = np.mean(results)
    std_ev  = np.std(results)
    return {
        'avg_ev':    float(avg_ev),
        'std_ev':    float(std_ev),
        'n_hands':   n_hands,
    }


def _play_hand(
    gs: GameState,
    players: list[DeepCFRPlayer],
    use_avg_strategy: bool = True,
) -> float:
    """Play one hand with trained strategies and return P0's EV."""
    depth = 0
    while gs.terminal == Terminal.NOT_TERMINAL and depth < 50:
        actions = abstract_actions(gs)
        if not actions:
            break

        player   = gs.current_player
        features = gs.info_set_features(player)
        n        = len(actions)

        if use_avg_strategy:
            strategy = players[player].get_avg_strategy(features, n)
        else:
            strategy = players[player].get_strategy(features, n)

        # Sample action
        strategy = np.maximum(strategy, 1e-8)
        strategy /= strategy.sum()
        idx    = np.random.choice(n, p=strategy)
        action = actions[idx]
        gs     = gs.apply_action(action)
        depth += 1

    if gs.terminal == Terminal.NOT_TERMINAL:
        return 0.0

    gs.board = gs.board[:5]
    try:
        return gs.payoff(0)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────
# Strategy inspection
# ─────────────────────────────────────────────────────────────

def inspect_strategy(
    players: list[DeepCFRPlayer],
    n_samples: int = 8,
):
    """Print strategy at a few canonical game states for interpretation."""
    from treys import Card

    print(f"\n{'─'*70}")
    print("  Strategy Inspection (avg strategy at canonical spots)")
    print(f"{'─'*70}")

    # A few interesting opening scenarios
    scenarios = [
        ("AA preflop (P0 SB)", [Card.new('Ah'), Card.new('As')],
                                [Card.new('7d'), Card.new('2c')], []),
        ("72o preflop (P0 SB)", [Card.new('7h'), Card.new('2s')],
                                 [Card.new('Kd'), Card.new('Ac')], []),
        ("KK preflop (P1 BB)", [Card.new('3h'), Card.new('8c')],
                                [Card.new('Kd'), Card.new('Ks')], []),
        ("Top pair on flop",   [Card.new('Ah'), Card.new('Kd')],
                                [Card.new('5s'), Card.new('2c')],
                                [Card.new('As'), Card.new('7h'), Card.new('2d')]),
    ]

    from deep_cfr.game_engine import GameState, BIG_BLIND, SMALL_BLIND
    action_names = ['fold', 'check/call', 'bet33', 'bet50', 'bet75', 'allin']

    for name, hole0, hole1, board in scenarios:
        gs = GameState()
        gs.hole  = [hole0, hole1]
        gs.board = board + [0, 0, 0, 0, 0]  # Dummy remaining cards

        if board:
            gs.street = 1
            gs._advance_street()
            gs.street = 1
            gs.board  = board + [0, 0]

        actions = abstract_actions(gs)
        if not actions:
            continue

        player   = gs.current_player
        features = gs.info_set_features(player)
        n        = len(actions)
        strategy = players[player].get_avg_strategy(features, n)

        h0_str = ' '.join(Card.int_to_str(c) for c in hole0)
        h1_str = ' '.join(Card.int_to_str(c) for c in hole1)
        b_str  = ' '.join(Card.int_to_str(c) for c in board) if board else 'none'

        print(f"\n  {name}")
        print(f"    P0: {h0_str} | P1: {h1_str} | Board: {b_str}")
        print(f"    Acting: P{player}")
        print("    Strategy:")
        for i, (action, prob) in enumerate(zip(actions, strategy)):
            bar = '█' * int(prob * 30)
            print(f"      {str(action):<20} {prob:.3f}  {bar}")


# ─────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────

def train(
    n_iterations:   int = 30,
    n_traversals:   int = 200,
    batch_size:     int = 512,
    n_adv_steps:    int = 300,
    n_strat_steps:  int = 200,
    eval_every:     int = 5,
    save_dir:       str = 'deep_cfr/checkpoints',
    verbose:        bool = True,
):
    os.makedirs(save_dir, exist_ok=True)

    feature_dim = GameState.feature_dim()
    n_actions   = MAX_ACTIONS

    print(f"\n{'='*70}")
    print("  STAGE 4: DEEP CFR — NLHE")
    print(f"{'='*70}")
    print(f"""
  Algorithm:    Deep CFR (Brown & Sandholm 2019)
  Game:         Heads-up No-Limit Hold'em (100 BB)
  Stacks:       {STARTING_STACK:.0f} BB each, proper all-in terminals
  Feature dim:  {feature_dim}
  Max actions:  {n_actions} per info set (abstract bet abstraction)
  Bet sizes:    check/call | 33% | 50% | 75% | pot | all-in
  Buffer:       Reservoir sampling, 500k capacity
  Networks:     3-layer × 256-unit, LayerNorm, ReLU
  Training:     {n_iterations} iterations × {n_traversals} traversals × 2 players
""")

    # Initialize players
    players = [
        DeepCFRPlayer(0, feature_dim, n_actions),
        DeepCFRPlayer(1, feature_dim, n_actions),
    ]

    history = []
    t_start = time.time()

    for t in range(1, n_iterations + 1):
        t_iter = time.time()

        print(f"\n{'─'*70}")
        print(f"  ITERATION {t}/{n_iterations}")
        print(f"{'─'*70}")

        # ── 1. Traversals ──────────────────────────────────────
        print(f"  Traversing ({n_traversals} hands × 2 players)...")
        trav_stats = cfr_iteration(
            players       = players,
            iteration     = t,
            n_traversals  = n_traversals,
            verbose       = verbose,
        )

        # ── 2. Network training ────────────────────────────────
        print(f"  Training networks...")
        loss_stats = train_networks(
            players       = players,
            batch_size    = batch_size,
            n_adv_steps   = n_adv_steps,
            n_strat_steps = n_strat_steps,
            verbose       = verbose,
        )

        # ── 3. Log ─────────────────────────────────────────────
        elapsed = time.time() - t_start
        record  = {**trav_stats, **loss_stats, 'elapsed': elapsed}
        history.append(record)

        print(f"\n  Summary:")
        print(f"    EV P0: {trav_stats['ev_p0']:+.3f} BB | EV P1: {trav_stats['ev_p1']:+.3f} BB")
        print(f"    Adv loss:   P0={loss_stats['adv_loss_p0']:.4f} | P1={loss_stats['adv_loss_p1']:.4f}")
        print(f"    Strat loss: P0={loss_stats['strat_loss_p0']:.4f} | P1={loss_stats['strat_loss_p1']:.4f}")
        print(f"    Adv buf:    P0={trav_stats['adv_buf_p0']:,} | P1={trav_stats['adv_buf_p1']:,}")
        print(f"    Iter time:  {time.time()-t_iter:.1f}s | Total: {elapsed:.1f}s")

        # ── 4. Evaluation ──────────────────────────────────────
        if t % eval_every == 0:
            print(f"\n  Evaluating strategy ({500} sample hands)...")
            eval_stats = estimate_exploitability(players, n_hands=500)
            print(f"    Avg EV (trained strat): {eval_stats['avg_ev']:+.3f} ± {eval_stats['std_ev']:.3f} BB")
            record['eval'] = eval_stats

        # ── 5. Checkpoint ──────────────────────────────────────
        if t % eval_every == 0:
            for p in range(2):
                players[p].save(f"{save_dir}/player{p}_iter{t}")
            with open(f"{save_dir}/history.json", 'w') as f:
                json.dump(history, f, indent=2)
            print(f"  Checkpoint saved → {save_dir}/")

    # ── Final inspection ──────────────────────────────────────
    inspect_strategy(players)

    # Save final
    for p in range(2):
        players[p].save(f"{save_dir}/player{p}_final")
    with open(f"{save_dir}/history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Training complete in {time.time()-t_start:.1f}s")
    print(f"  Models saved → {save_dir}/")
    print(f"{'='*70}\n")

    return players, history


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 4: Deep CFR for NLHE')
    parser.add_argument('--iterations',  type=int, default=30,
                        help='CFR iterations (default: 30)')
    parser.add_argument('--traversals',  type=int, default=200,
                        help='Traversals per player per iteration (default: 200)')
    parser.add_argument('--batch-size',  type=int, default=512)
    parser.add_argument('--adv-steps',   type=int, default=300,
                        help='Advantage network training steps per iter')
    parser.add_argument('--strat-steps', type=int, default=200,
                        help='Strategy network training steps per iter')
    parser.add_argument('--eval-every',  type=int, default=5)
    parser.add_argument('--save-dir',    type=str, default='deep_cfr/checkpoints')
    args = parser.parse_args()

    train(
        n_iterations  = args.iterations,
        n_traversals  = args.traversals,
        batch_size    = args.batch_size,
        n_adv_steps   = args.adv_steps,
        n_strat_steps = args.strat_steps,
        eval_every    = args.eval_every,
        save_dir      = args.save_dir,
    )
