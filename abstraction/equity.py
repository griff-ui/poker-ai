"""
Equity computation for NLHE hand abstraction.
"""

import random
import numpy as np
from itertools import combinations
from treys import Card, Evaluator

evaluator = Evaluator()

RANKS = 'AKQJT98765432'
SUITS = 'hdcs'
ALL_CARDS = [Card.new(r + s) for r in RANKS for s in SUITS]
N_HISTOGRAM_BINS = 10


def remaining_deck(excluded: list[int]) -> list[int]:
    excl = set(excluded)
    return [c for c in ALL_CARDS if c not in excl]


def preflop_equity(hole: list[int], n_samples: int = 1000) -> float:
    deck = remaining_deck(hole)
    wins = 0.0
    for _ in range(n_samples):
        sample = random.sample(deck, 7)
        opp, board = sample[:2], sample[2:]
        my       = evaluator.evaluate(board, hole)
        opp_rank = evaluator.evaluate(board, opp)
        if my < opp_rank:   wins += 1.0
        elif my == opp_rank: wins += 0.5
    return wins / n_samples


def equity_histogram(
    hole: list[int],
    board: list[int],
    n_samples: int = 200,
    n_bins: int = N_HISTOGRAM_BINS,
) -> np.ndarray:
    """Equity histogram over future runouts for a single player vs random opponent."""
    deck = remaining_deck(hole + board)
    street = len(board)
    cards_to_draw = 5 - street
    equities = []
    for _ in range(n_samples):
        sample = random.sample(deck, 2 + cards_to_draw)
        opp = sample[:2]
        runout = sample[2:]
        full_board = board + runout
        my       = evaluator.evaluate(full_board, hole)
        opp_rank = evaluator.evaluate(full_board, opp)
        if my < opp_rank:   equities.append(1.0)
        elif my == opp_rank: equities.append(0.5)
        else:                equities.append(0.0)
    hist, _ = np.histogram(equities, bins=n_bins, range=(0.0, 1.0))
    total = hist.sum()
    return hist / total if total > 0 else hist.astype(float)


def dual_equity_histogram(
    hole0: list[int],
    hole1: list[int],
    board: list[int],
    n_samples: int = 80,
    n_bins: int = N_HISTOGRAM_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute equity histograms for BOTH players simultaneously.
    ~2.5x faster than calling equity_histogram twice.
    """
    deck = remaining_deck(hole0 + hole1 + board)
    street = len(board)
    cards_to_draw = 5 - street
    eq0, eq1 = [], []
    for _ in range(n_samples):
        runout = random.sample(deck, cards_to_draw)
        full_board = board + runout
        r0 = evaluator.evaluate(full_board, hole0)
        r1 = evaluator.evaluate(full_board, hole1)
        if r0 < r1:
            eq0.append(1.0); eq1.append(0.0)
        elif r0 == r1:
            eq0.append(0.5); eq1.append(0.5)
        else:
            eq0.append(0.0); eq1.append(1.0)

    def to_hist(eq):
        hist, _ = np.histogram(eq, bins=n_bins, range=(0.0, 1.0))
        t = hist.sum()
        return hist / t if t > 0 else hist.astype(float)

    return to_hist(eq0), to_hist(eq1)


def river_strength(hole: list[int], board: list[int]) -> float:
    """Exact hand strength percentile on the river."""
    assert len(board) == 5
    deck = remaining_deck(hole + board)
    my_rank = evaluator.evaluate(board, hole)
    wins = ties = total = 0
    for opp in combinations(deck, 2):
        opp_rank = evaluator.evaluate(board, list(opp))
        if my_rank < opp_rank:  wins += 1
        elif my_rank == opp_rank: ties += 1
        total += 1
    return (wins + 0.5 * ties) / total if total > 0 else 0.0


def emd(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """Earth Mover's Distance (Wasserstein-1) between two 1D histograms."""
    cdf_a = np.cumsum(hist_a)
    cdf_b = np.cumsum(hist_b)
    return float(np.sum(np.abs(cdf_a - cdf_b)))
