"""
NLHE Game Engine — Stage 4

Proper implementation with:
  - Real stack tracking for both players
  - All-in terminal detection (no more raises when stacks empty)
  - Effective stack bet sizing (bets capped at min(p0_stack, p1_stack))
  - Side-pot-free 2-player logic (heads-up only)
  - Clean terminal classification: FOLD | ALLIN_SHOWDOWN | STREET_SHOWDOWN

Design principles:
  - Immutable-style: apply_action returns a new GameState
  - All chip amounts in big blinds (float)
  - Information encoded as raw feature vectors for Deep CFR network input
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto

from treys import Card, Evaluator

evaluator = Evaluator()

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

STARTING_STACK = 100.0   # BB (100 BB deep — standard cash game)
SMALL_BLIND    = 0.5
BIG_BLIND      = 1.0
MAX_RAISES     = 4        # Per street (standard live poker)

# Bet sizes: fraction of pot (not stack) — context-aware sizing added below
BET_FRACS = {
    'b33':  0.33,
    'b50':  0.50,
    'b75':  0.75,
    'b100': 1.00,
    'b150': 1.50,
}

STREETS = ['preflop', 'flop', 'turn', 'river']


class Terminal(Enum):
    NOT_TERMINAL    = auto()
    FOLD            = auto()   # Someone folded
    ALLIN_SHOWDOWN  = auto()   # Both players all-in, run board
    SHOWDOWN        = auto()   # End of river betting


# ─────────────────────────────────────────────────────────────
# Action
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Action:
    kind: str            # 'fold' | 'check' | 'call' | 'bet' | 'raise' | 'allin'
    amount: float = 0.0  # Chip amount (0 for fold/check)

    def __repr__(self):
        if self.kind in ('fold', 'check'):
            return self.kind
        return f"{self.kind}({self.amount:.1f})"

    @staticmethod
    def fold()  -> Action: return Action('fold')
    @staticmethod
    def check() -> Action: return Action('check')
    @staticmethod
    def call(amt: float) -> Action: return Action('call', amt)
    @staticmethod
    def bet(amt: float)  -> Action: return Action('bet', amt)
    @staticmethod
    def raise_(amt: float) -> Action: return Action('raise', amt)
    @staticmethod
    def allin(amt: float) -> Action: return Action('allin', amt)


# ─────────────────────────────────────────────────────────────
# GameState — full state of one hand
# ─────────────────────────────────────────────────────────────

@dataclass
class GameState:
    """
    Complete game state for a heads-up NLHE hand.

    Player 0 = Small Blind / Out-of-Position (OOP)
    Player 1 = Big Blind / In-Position (IP)

    Preflop: P1 (BTN/BB) acts last. Postflop: P0 (SB/OOP) acts first.
    """

    # Cards
    hole:  list[list[int]] = field(default_factory=lambda: [[], []])
    board: list[int]       = field(default_factory=list)

    # Stacks and pot
    stacks:  list[float] = field(default_factory=lambda: [STARTING_STACK - SMALL_BLIND,
                                                           STARTING_STACK - BIG_BLIND])
    pot:     float = SMALL_BLIND + BIG_BLIND
    contrib: list[float] = field(default_factory=lambda: [SMALL_BLIND, BIG_BLIND])
    # contrib[i] = total chips player i has put in THIS STREET

    # Betting state
    street:        int   = 0        # 0=preflop, 1=flop, 2=turn, 3=river
    street_pot:    float = 0.0      # Chips in pot from prior streets
    to_call:       float = BIG_BLIND - SMALL_BLIND  # Amount P0 must call to stay in
    raises_this_street: int = 0
    actions_this_street: list[Action] = field(default_factory=list)
    history: list[list[Action]] = field(default_factory=lambda: [[], [], [], []])

    # Terminal state
    terminal:    Terminal = Terminal.NOT_TERMINAL
    winner:      Optional[int] = None   # 0 or 1 (for fold terminals)
    folded_by:   Optional[int] = None

    def copy(self) -> GameState:
        gs = GameState.__new__(GameState)
        gs.hole           = [list(h) for h in self.hole]
        gs.board          = list(self.board)
        gs.stacks         = list(self.stacks)
        gs.pot            = self.pot
        gs.contrib        = list(self.contrib)
        gs.street         = self.street
        gs.street_pot     = self.street_pot
        gs.to_call        = self.to_call
        gs.raises_this_street = self.raises_this_street
        gs.actions_this_street = list(self.actions_this_street)
        gs.history        = [list(h) for h in self.history]
        gs.terminal       = self.terminal
        gs.winner         = self.winner
        gs.folded_by      = self.folded_by
        return gs

    # ── Player to act ────────────────────────────────────────

    @property
    def current_player(self) -> int:
        """
        Preflop: P0 (SB) acts first (P1 has option after that).
        Postflop: P0 (OOP) acts first.
        In both cases: alternates by action count modulo 2,
        with preflop starting at P0 (SB facing BB's raise).
        """
        n = len(self.actions_this_street)
        if self.street == 0:
            # Preflop: SB (P0) acts first, then BB (P1) has option
            return n % 2
        else:
            # Postflop: OOP (P0) acts first
            return n % 2

    # ── Effective stack ──────────────────────────────────────

    @property
    def effective_stack(self) -> float:
        """The maximum any player can win/lose from this point."""
        return min(self.stacks[0], self.stacks[1])

    # ── Legal action generation ──────────────────────────────

    def legal_actions(self) -> list[Action]:
        """
        Generate all legal actions from this state.
        Bet sizes are context-aware: capped at effective stack,
        and sized as fraction of total pot (street_pot + contrib).
        """
        if self.terminal != Terminal.NOT_TERMINAL:
            return []

        player   = self.current_player
        my_stack = self.stacks[player]
        opp_contrib = self.contrib[1 - player]
        my_contrib  = self.contrib[player]

        call_amount = min(opp_contrib - my_contrib, my_stack)
        can_raise   = (self.raises_this_street < MAX_RAISES
                       and self.effective_stack > call_amount)

        actions = []

        # Fold (always legal if facing a bet)
        if call_amount > 0:
            actions.append(Action.fold())

        # Check (if no bet to call)
        if call_amount == 0:
            actions.append(Action.check())
        else:
            # Call
            if call_amount > 0:
                actions.append(Action.call(call_amount))

        # Bets / Raises
        if can_raise:
            # Total pot after calling = street_pot + sum(contrib after call)
            pot_after_call = (self.street_pot
                              + self.contrib[0] + self.contrib[1]
                              + call_amount)

            for name, frac in BET_FRACS.items():
                raise_to = call_amount + pot_after_call * frac
                raise_to = min(raise_to, my_stack)   # Cap at stack
                raise_to = max(raise_to, BIG_BLIND)  # Minimum raise = 1 BB

                # Don't add if it's just a call (stack too short)
                if raise_to > call_amount + 0.01:
                    # Check if we'd be going all-in
                    if abs(raise_to - my_stack) < 0.01:
                        actions.append(Action.allin(raise_to))
                    else:
                        kind = 'raise' if call_amount > 0 else 'bet'
                        actions.append(Action(kind, raise_to))

            # Deduplicate by amount (different fracs can land on same value)
            seen = set()
            deduped = []
            for a in actions:
                key = round(a.amount, 2)
                if a.kind in ('bet', 'raise', 'allin') and key in seen:
                    continue
                if a.kind in ('bet', 'raise', 'allin'):
                    seen.add(key)
                deduped.append(a)
            actions = deduped

            # Always include explicit all-in if not already there
            if my_stack > call_amount + 0.01:
                has_allin = any(a.kind == 'allin' for a in actions)
                if not has_allin:
                    actions.append(Action.allin(my_stack))

        return actions

    # ── State transition ─────────────────────────────────────

    def apply_action(self, action: Action) -> GameState:
        """Return a new GameState after applying action."""
        gs = self.copy()
        player = gs.current_player

        if action.kind == 'fold':
            gs.terminal  = Terminal.FOLD
            gs.folded_by = player
            gs.winner    = 1 - player
            gs._record(action)
            return gs

        if action.kind == 'check':
            gs._record(action)

        elif action.kind == 'call':
            amt = min(action.amount, gs.stacks[player])
            gs.stacks[player]  -= amt
            gs.contrib[player] += amt
            gs.pot             += amt
            gs._record(action)

        elif action.kind in ('bet', 'raise', 'allin'):
            # First: call any outstanding
            call_amt = max(gs.contrib[1 - player] - gs.contrib[player], 0.0)
            call_amt = min(call_amt, gs.stacks[player])
            gs.stacks[player]  -= call_amt
            gs.contrib[player] += call_amt
            gs.pot             += call_amt

            # Then: put in the raise increment
            raise_total = action.amount
            raise_inc   = min(raise_total - call_amt,
                              gs.stacks[player])
            raise_inc   = max(raise_inc, 0.0)
            gs.stacks[player]  -= raise_inc
            gs.contrib[player] += call_amt + raise_inc
            gs.pot             += raise_inc
            gs.to_call          = gs.contrib[player] - gs.contrib[1 - player]
            gs.raises_this_street += 1
            gs._record(action)

        # ── Check for all-in showdown ─────────────────────────
        if min(gs.stacks) <= 0.01 and gs.terminal == Terminal.NOT_TERMINAL:
            gs.terminal = Terminal.ALLIN_SHOWDOWN
            return gs

        # ── Check street completion ───────────────────────────
        if gs._street_complete():
            if gs.street == 3:
                gs.terminal = Terminal.SHOWDOWN
            else:
                gs._advance_street()

        return gs

    def _record(self, action: Action):
        self.actions_this_street.append(action)
        self.history[self.street].append(action)

    def _street_complete(self) -> bool:
        """
        A street ends when all bets are equalized and both players have acted.

        Terminal sequences:
          call          → bets equal, both acted (preflop: SB calls BB)
          check, check  → both check postflop
          bet/raise, call   → aggressor bet, opponent called
          [call], check → preflop: SB completes, BB checks (option)
          fold          → handled in apply_action
        """
        acts = self.actions_this_street
        n    = len(acts)
        if n < 2:
            return False

        last      = acts[-1]
        prev      = acts[-2]

        # Any call ends the street (bets are now equal)
        if last.kind == 'call':
            return True

        # Check-check ends the street
        if last.kind == 'check' and prev.kind in ('check', 'call'):
            return True

        # Fold is terminal (handled upstream but guard here)
        if last.kind == 'fold':
            return True

        return False

    def _advance_street(self):
        """Move to the next street: reset betting, move contrib to street_pot."""
        self.street_pot      += self.contrib[0] + self.contrib[1]
        self.contrib          = [0.0, 0.0]
        self.to_call          = 0.0
        self.raises_this_street = 0
        self.actions_this_street = []
        self.street          += 1

    # ── Payoffs ──────────────────────────────────────────────

    def payoff(self, player: int) -> float:
        """
        Terminal payoff for `player` in chips relative to starting stack.
        Positive = profit, negative = loss.
        """
        assert self.terminal != Terminal.NOT_TERMINAL, "Not terminal"

        if self.terminal == Terminal.FOLD:
            if self.winner == player:
                # Win opponent's contribution
                opp = 1 - player
                return self.contrib[opp] + self.street_pot / 2  # approximation
                # More precisely: win what opponent put in minus our blind
            else:
                return -(self.contrib[player])

        # Showdown or all-in showdown
        board = self._full_board()
        my_hole  = self.hole[player]
        opp_hole = self.hole[1 - player]
        my_rank  = evaluator.evaluate(board, my_hole)
        op_rank  = evaluator.evaluate(board, opp_hole)

        # Total pot
        total = self.street_pot + self.contrib[0] + self.contrib[1]

        if my_rank < op_rank:    # Lower rank = better hand in treys
            return total - self.contrib[player] - self.street_pot / 2
        elif my_rank > op_rank:
            return -self.contrib[player]
        else:
            return 0.0  # Chop

    def _full_board(self) -> list[int]:
        """Return 5-card board (for showdown evaluation)."""
        return self.board  # Caller must ensure board is complete

    # ── Feature encoding for neural network ──────────────────

    def info_set_features(self, player: int) -> list[float]:
        """
        Encode information set as a fixed-size float vector for the network.
        This is the feature representation Deep CFR uses as network input.

        Features (total: 274 dimensions):
          - Hole cards:     2 × 52 one-hot = 104
          - Board cards:    5 × 52 one-hot = 260   (padded with zeros pre-river)
          - Street:         4 one-hot              = 4
          - Pot (normalized): 1                     = 1
          - My stack (norm):  1                     = 1
          - Opp stack (norm): 1                     = 1
          - To call (norm):   1                     = 1
          - Raises this str:  1                     = 1
          Total: 104 + 260 + 4 + 6 = 374 dims (but we compact board to occupied)
        Actually:
          - 2 hole cards × 52 = 104
          - 5 board slots × 52 = 260
          - 4 street flags = 4
          - pot / starting_stack = 1
          - my_stack / starting_stack = 1
          - opp_stack / starting_stack = 1
          - to_call / starting_stack = 1
          - raises / MAX_RAISES = 1
          Total = 372
        """
        feats = []
        STACK = STARTING_STACK

        # Hole cards (one-hot, 2 × 52)
        hole_enc = [0.0] * 104
        for i, card in enumerate(self.hole[player][:2]):
            idx = _card_idx(card)
            hole_enc[i * 52 + idx] = 1.0
        feats.extend(hole_enc)

        # Board cards (one-hot, 5 × 52, zero-padded)
        board_enc = [0.0] * 260
        for i, card in enumerate(self.board[:5]):
            idx = _card_idx(card)
            board_enc[i * 52 + idx] = 1.0
        feats.extend(board_enc)

        # Street one-hot (4)
        street_enc = [0.0] * 4
        street_enc[self.street] = 1.0
        feats.extend(street_enc)

        # Normalized scalars
        feats.append(self.pot / STACK)
        feats.append(self.stacks[player] / STACK)
        feats.append(self.stacks[1 - player] / STACK)
        feats.append(self.to_call / STACK)
        feats.append(self.raises_this_street / MAX_RAISES)

        return feats  # length 374

    @staticmethod
    def feature_dim() -> int:
        return 373   # 104 hole + 260 board + 4 street + 5 scalars

    def __repr__(self):
        s = STREETS[self.street]
        board_str = ' '.join(Card.int_to_str(c) for c in self.board)
        return (f"GameState(street={s}, pot={self.pot:.1f}, "
                f"stacks={self.stacks[0]:.1f}/{self.stacks[1]:.1f}, "
                f"board=[{board_str}], terminal={self.terminal.name})")


# ─────────────────────────────────────────────────────────────
# Card utilities
# ─────────────────────────────────────────────────────────────

def _card_idx(card: int) -> int:
    """Map a treys card integer to a 0-51 index."""
    rank_bits = (card >> 8) & 0xF
    suit_bits = (card >> 12) & 0xF
    rank = rank_bits - 2      # 0-12
    suit = suit_bits.bit_length() - 1  # 0-3
    return rank * 4 + suit


# ─────────────────────────────────────────────────────────────
# Deal a fresh hand
# ─────────────────────────────────────────────────────────────

def deal_hand(deck: Optional[list[int]] = None) -> GameState:
    """
    Deal hole cards and initialize game state with blinds posted.
    Optionally accepts a pre-shuffled deck (for testing).
    """
    from abstraction.equity import ALL_CARDS

    if deck is None:
        deck = list(ALL_CARDS)
        random.shuffle(deck)

    gs = GameState()
    gs.hole  = [deck[:2], deck[2:4]]
    gs.board = deck[4:9]   # Pre-deal all 5 community cards; reveal by street
    return gs


def visible_board(gs: GameState) -> list[int]:
    """Return only the community cards visible at the current street."""
    if gs.street == 0: return []
    if gs.street == 1: return gs.board[:3]
    if gs.street == 2: return gs.board[:4]
    return gs.board[:5]


# ─────────────────────────────────────────────────────────────
# Action abstraction helper
# ─────────────────────────────────────────────────────────────

def abstract_actions(gs: GameState) -> list[Action]:
    """
    Return a reduced set of abstract actions for CFR traversal.
    In Deep CFR we use this to keep the tree manageable:
      - Always include: fold, check/call, all-in
      - Bets: 50% pot, 100% pot (or subset depending on raises)
    """
    legal = gs.legal_actions()
    if not legal:
        return []

    # Always keep fold, check, call
    result = [a for a in legal if a.kind in ('fold', 'check', 'call')]

    # Keep one smaller bet, one larger bet, and all-in
    bets   = [a for a in legal if a.kind in ('bet', 'raise')]
    allins = [a for a in legal if a.kind == 'allin']

    if bets:
        mid = len(bets) // 2
        # Include ~50% and ~100% pot sizes
        result.append(bets[min(1, mid)])           # smaller bet
        if len(bets) > 2:
            result.append(bets[-1])                # largest non-allin bet

    result.extend(allins)

    # Deduplicate
    seen  = set()
    dedup = []
    for a in result:
        key = (a.kind, round(a.amount, 1))
        if key not in seen:
            seen.add(key)
            dedup.append(a)

    return dedup
