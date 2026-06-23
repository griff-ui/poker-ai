"""
Deep CFR Traversal — Stage 4

Implements the core Deep CFR algorithm from Brown & Sandholm (2019):

  For each CFR iteration t:
    For each player p:
      Traverse the game tree with external sampling
      At each info set I for player p:
        - Query advantage network σ_p^t(I) via regret matching
        - For traverser: explore ALL actions, compute advantages, add to buffer
        - For opponent:  SAMPLE one action weighted by strategy
      After traversal: train advantage network on buffer
    After all players: train strategy networks on strategy buffer

Key differences from vanilla MCCFR:
  - Information sets identified by FEATURE VECTORS not strings
  - Strategies stored in neural networks, not hash tables
  - Reservoir buffers enable generalization across similar states
  - Linear CFR weighting: iteration t samples weighted by t (more recent = more important)
"""

import random
import numpy as np
import torch
from typing import Optional

from deep_cfr.game_engine import (
    GameState, Action, Terminal,
    deal_hand, visible_board, abstract_actions,
    STARTING_STACK,
)
from deep_cfr.networks import DeepCFRPlayer

# Maximum network output actions (pad/truncate to this)
MAX_ACTIONS = 6   # fold, check/call, 3 bet sizes, all-in


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _pad_advantages(advantages: list[float], n: int) -> np.ndarray:
    """Pad or truncate advantage list to MAX_ACTIONS."""
    result = np.zeros(MAX_ACTIONS, dtype=np.float32)
    for i in range(min(n, MAX_ACTIONS)):
        result[i] = advantages[i]
    return result


def _pad_strategy(strategy: np.ndarray, n: int) -> np.ndarray:
    """Pad strategy to MAX_ACTIONS."""
    result = np.zeros(MAX_ACTIONS, dtype=np.float32)
    result[:n] = strategy[:n]
    return result


def _terminal_payoff(gs: GameState, traverser: int) -> float:
    """
    Compute terminal payoff for traverser.
    Handles fold, all-in showdown, and regular showdown.
    For showdowns, we need the full 5-card board.
    """
    if gs.terminal == Terminal.FOLD:
        return gs.payoff(traverser)

    # For showdowns: evaluate with full 5-card board
    gs_eval = gs
    if len(gs.board) < 5:
        # Board needs to be completed (shouldn't happen in well-structured traversal)
        gs_eval = gs
    board5 = gs.board[:5]
    gs_eval.board = board5
    return gs_eval.payoff(traverser)


# ─────────────────────────────────────────────────────────────
# Core traversal
# ─────────────────────────────────────────────────────────────

def traverse(
    gs: GameState,
    traverser: int,
    players: list[DeepCFRPlayer],
    iteration: int,
    depth: int = 0,
) -> float:
    """
    External sampling MCCFR traversal with neural network strategies.

    Returns expected value for `traverser` from this game state.

    External sampling means:
      - Traverser player: enumerate ALL actions, compute counterfactual values
      - Opponent player: SAMPLE one action from current strategy
      - Chance nodes (board cards): already dealt at game start

    Regret update:
      For traverser at info set I with strategy σ and action values v(a):
        advantage(a) = v(a) − Σ_a σ(a)·v(a)
      These advantages are stored in the reservoir buffer.

    Linear CFR weighting:
      Each sample is weighted by iteration number t, so more recent
      data is upweighted in the network training. This is the "linear CFR"
      variant which converges faster than vanilla CFR.
    """

    # ── Terminal states ───────────────────────────────────────
    if gs.terminal == Terminal.FOLD:
        return _terminal_payoff(gs, traverser)

    if gs.terminal in (Terminal.ALLIN_SHOWDOWN, Terminal.SHOWDOWN):
        return _terminal_payoff(gs, traverser)

    # ── Get legal actions ─────────────────────────────────────
    actions = abstract_actions(gs)
    if not actions:
        return 0.0

    n = len(actions)
    player = gs.current_player

    # ── Get feature vector and strategy ──────────────────────
    # Board is revealed street by street
    gs_view = gs
    # Features encode only what player can see
    features = gs.info_set_features(player)

    # Get strategy from advantage network via regret matching
    strategy = players[player].get_strategy(features, n)

    if player == traverser:
        # ── Traverser: explore all actions ───────────────────

        action_values = np.zeros(n, dtype=np.float64)

        for i, action in enumerate(actions):
            next_gs = gs.apply_action(action)
            # Reveal board cards at street boundaries
            _sync_board(gs, next_gs)
            v = traverse(next_gs, traverser, players, iteration, depth + 1)
            action_values[i] = v

        # Expected value under current strategy
        ev = float(np.dot(strategy, action_values[:n]))

        # Compute advantages: a(I, a) = v(I, a) - v(I)
        advantages = action_values[:n] - ev

        # Add to advantage reservoir buffer (weighted by iteration)
        players[traverser].add_advantage_sample(
            features   = features,
            advantages = list(_pad_advantages(list(advantages), n)),
            weight     = float(iteration),
            n_actions  = n,
        )

        # Add to strategy buffer (for average strategy tracking)
        players[traverser].add_strategy_sample(
            features  = features,
            strategy  = list(_pad_strategy(strategy, n)),
            weight    = float(iteration),
            n_actions = n,
        )

        return ev

    else:
        # ── Opponent: sample one action ───────────────────────

        # Add to strategy buffer (track opponent's strategy too)
        players[player].add_strategy_sample(
            features  = features,
            strategy  = list(_pad_strategy(strategy, n)),
            weight    = float(iteration),
            n_actions = n,
        )

        # Sample action from strategy
        idx    = np.random.choice(n, p=strategy)
        action = actions[idx]

        next_gs = gs.apply_action(action)
        _sync_board(gs, next_gs)

        return traverse(next_gs, traverser, players, iteration, depth + 1)


def _sync_board(prev_gs: GameState, next_gs: GameState):
    """
    When the street advances, reveal the appropriate board cards.
    The full 5-card board is pre-dealt; we just expose street-by-street.
    """
    if next_gs.street > prev_gs.street or next_gs.terminal != Terminal.NOT_TERMINAL:
        # Board is already pre-dealt in gs.board[:5]
        # For evaluation we need the right slice
        pass  # board already set in GameState; visible_board() handles slicing


def compute_terminal_payoff(gs: GameState, traverser: int) -> float:
    """
    Compute payoff at a terminal node, handling board completion.
    For all-in showdowns the board may need to be run out.
    """
    if gs.terminal == Terminal.FOLD:
        return gs.payoff(traverser)

    # Ensure we have a 5-card board for evaluation
    if len(gs.board) >= 5:
        return gs.payoff(traverser)

    # This shouldn't happen since we pre-deal all 5 cards
    return 0.0


# ─────────────────────────────────────────────────────────────
# Single CFR iteration
# ─────────────────────────────────────────────────────────────

def cfr_iteration(
    players: list[DeepCFRPlayer],
    iteration: int,
    n_traversals: int = 100,
    verbose: bool = False,
) -> dict:
    """
    One full Deep CFR iteration:
      1. For each player, run n_traversals external-sampling MCCFR traversals
      2. Collect samples into reservoir buffers
      Returns stats dict.
    """
    ev_totals = [0.0, 0.0]
    n_terminal = [0, 0]

    # Lock networks in eval mode for fast inference during traversal
    for p in players:
        p.set_inference_mode()

    for traverser in range(2):
        for _ in range(n_traversals):
            gs = deal_hand()
            try:
                ev = traverse(gs, traverser, players, iteration)
                ev_totals[traverser] += ev
                n_terminal[traverser] += 1
            except Exception:
                pass  # Skip malformed games

    stats = {
        'iteration':   iteration,
        'ev_p0':       ev_totals[0] / max(n_terminal[0], 1),
        'ev_p1':       ev_totals[1] / max(n_terminal[1], 1),
        'adv_buf_p0':  len(players[0].advantage_buf),
        'adv_buf_p1':  len(players[1].advantage_buf),
        'strat_buf_p0': len(players[0].strategy_buf),
        'strat_buf_p1': len(players[1].strategy_buf),
    }

    if verbose:
        print(f"  Traversals complete | EV P0: {stats['ev_p0']:+.3f} | "
              f"EV P1: {stats['ev_p1']:+.3f} | "
              f"Adv buf: {stats['adv_buf_p0']}/{stats['adv_buf_p1']}")

    return stats


# ─────────────────────────────────────────────────────────────
# Network training step
# ─────────────────────────────────────────────────────────────

def train_networks(
    players: list[DeepCFRPlayer],
    batch_size: int = 512,
    n_adv_steps: int = 200,
    n_strat_steps: int = 200,
    verbose: bool = False,
) -> dict:
    """
    Train advantage and strategy networks after traversals.
    Returns training loss statistics.
    """
    losses = {}
    for p in range(2):
        players[p].set_training_mode()
        adv_losses  = players[p].train_advantage_network(batch_size, n_adv_steps)
        strat_losses = players[p].train_strategy_network(batch_size, n_strat_steps)
        players[p].set_inference_mode()
        losses[f'adv_loss_p{p}']   = np.mean(adv_losses)   if adv_losses   else 0.0
        losses[f'strat_loss_p{p}'] = np.mean(strat_losses) if strat_losses else 0.0
        if verbose:
            print(f"  P{p}: adv_loss={losses[f'adv_loss_p{p}']:.4f} | "
                  f"strat_loss={losses[f'strat_loss_p{p}']:.4f}")

    return losses
