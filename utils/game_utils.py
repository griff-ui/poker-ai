"""
Utility functions for poker game state representation and hand evaluation.
"""

from treys import Card, Evaluator, Deck
import random

evaluator = Evaluator()

# ─────────────────────────────────────────────
# Leduc Hold'em — a tractable toy poker game
# 6 cards: J, Q, K of two suits (6 total)
# 2 players, 2 betting rounds (preflop + flop)
# ─────────────────────────────────────────────

LEDUC_RANKS = ['J', 'Q', 'K']
LEDUC_SUITS = ['s', 'h']
LEDUC_DECK = [r + s for r in LEDUC_RANKS for s in LEDUC_SUITS]  # 6 cards

RANK_VALUE = {'J': 1, 'Q': 2, 'K': 3}

def leduc_hand_strength(hole_card: str, board_card: str | None) -> int:
    """
    Simple Leduc hand strength.
    - Pair (hole matches board) beats any non-pair.
    - Within pairs/non-pairs, higher rank wins.
    Returns a score (higher = better).
    """
    rank = RANK_VALUE[hole_card[0]]
    if board_card is not None and hole_card[0] == board_card[0]:
        return 10 + rank  # Pair: 11, 12, 13
    return rank           # No pair: 1, 2, 3

def leduc_winner(p0_card: str, p1_card: str, board_card: str) -> int:
    """Returns 0 if player 0 wins, 1 if player 1 wins, -1 if tie."""
    s0 = leduc_hand_strength(p0_card, board_card)
    s1 = leduc_hand_strength(p1_card, board_card)
    if s0 > s1:
        return 0
    elif s1 > s0:
        return 1
    return -1  # Tie

def deal_leduc():
    """Deal a fresh Leduc game: returns (p0_card, p1_card, board_card)."""
    deck = LEDUC_DECK.copy()
    random.shuffle(deck)
    return deck[0], deck[1], deck[2]


# ─────────────────────────────────────────────
# Information Set Key helpers
# ─────────────────────────────────────────────

def info_set_key(hole_card: str, board_card: str | None, history: list[str]) -> str:
    """
    Encode a player's information set as a string key.
    board_card is None until the flop is dealt.
    history is a list of actions, e.g. ['r', 'c', 'r', 'c']
    """
    board_str = board_card if board_card else '?'
    return f"{hole_card}|{board_str}|{''.join(history)}"
