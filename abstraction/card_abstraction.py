"""
Card Abstraction: clusters hands into buckets at each street.

Strategy:
  Preflop  → equity vs random → k-means on scalar equity → N_PRE buckets
  Flop     → equity histogram over runouts → k-means with EMD → N_FLOP buckets
  Turn     → equity histogram over runouts → k-means with EMD → N_TURN buckets
  River    → exact strength percentile → N_RIVER equal-width percentile bins

Bucket IDs are integers 0..N-1 at each street.
Lower bucket ID = weaker hand (convention).

Usage:
    abstraction = CardAbstraction()
    abstraction.build(n_samples_equity=500)
    bucket = abstraction.bucket(hole_cards, board_cards)
"""

import random
import pickle
import os
import time
import numpy as np
from itertools import combinations
from collections import defaultdict
from treys import Card, Evaluator

from abstraction.equity import (
    preflop_equity,
    equity_histogram,
    river_strength,
    emd,
    ALL_CARDS,
    remaining_deck,
    RANKS,
    SUITS,
    N_HISTOGRAM_BINS,
)

# ─────────────────────────────────────────────────────────────
# Bucket counts per street (tune for compute vs precision)
# ─────────────────────────────────────────────────────────────
N_PRE   = 8    # Preflop buckets   (169 hands → 8 groups)
N_FLOP  = 12   # Flop buckets
N_TURN  = 12   # Turn buckets
N_RIVER = 8    # River buckets


# ─────────────────────────────────────────────────────────────
# Canonical preflop hand enumeration
# ─────────────────────────────────────────────────────────────

def canonical_preflop_hands() -> list[tuple[int, int, bool]]:
    """
    Returns all 169 canonical preflop hand types as
    (card1, card2, is_suited) using fixed canonical suit assignments.
    Pairs: same rank, different suits (hd).
    Suited: r1h + r2h.
    Offsuit: r1h + r2s.
    """
    hands = []
    rank_list = list(RANKS)
    for i, r1 in enumerate(rank_list):
        for r2 in rank_list[i:]:
            if r1 == r2:
                hands.append((Card.new(r1 + 'h'), Card.new(r2 + 'd'), False))
            else:
                # Suited
                hands.append((Card.new(r1 + 'h'), Card.new(r2 + 'h'), True))
                # Offsuit
                hands.append((Card.new(r1 + 'h'), Card.new(r2 + 's'), False))
    return hands


# ─────────────────────────────────────────────────────────────
# k-means with EMD (for histogram-based streets)
# ─────────────────────────────────────────────────────────────

def kmeans_emd(
    histograms: np.ndarray,
    n_clusters: int,
    n_iters: int = 30,
    n_restarts: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    k-means clustering using Earth Mover's Distance as the metric.
    Standard Euclidean k-means uses L2; we use EMD on the CDF.

    Args:
        histograms: (N, bins) array of equity histograms
        n_clusters: number of buckets
        n_iters: iterations per restart
        n_restarts: independent restarts (take best result)

    Returns:
        labels: (N,) cluster assignments
        centroids: (n_clusters, bins) centroid histograms
    """
    N = len(histograms)
    best_cost = float('inf')
    best_labels = None
    best_centroids = None

    for restart in range(n_restarts):
        # k-means++ initialization
        centroids = []
        first = random.randint(0, N - 1)
        centroids.append(histograms[first].copy())

        for _ in range(1, n_clusters):
            dists = np.array([
                min(emd(h, c) for c in centroids)
                for h in histograms
            ])
            probs = dists / dists.sum()
            idx = np.random.choice(N, p=probs)
            centroids.append(histograms[idx].copy())

        centroids = np.array(centroids)

        for iteration in range(n_iters):
            # Assignment step: assign each hand to nearest centroid by EMD
            labels = np.zeros(N, dtype=int)
            for i, h in enumerate(histograms):
                dists = [emd(h, c) for c in centroids]
                labels[i] = int(np.argmin(dists))

            # Update step: new centroid = mean of assigned histograms
            new_centroids = np.zeros_like(centroids)
            for k in range(n_clusters):
                members = histograms[labels == k]
                if len(members) > 0:
                    new_centroids[k] = members.mean(axis=0)
                else:
                    new_centroids[k] = centroids[k]

            if np.allclose(new_centroids, centroids, atol=1e-6):
                break
            centroids = new_centroids

        # Compute total cost
        cost = sum(
            emd(histograms[i], centroids[labels[i]])
            for i in range(N)
        )

        if cost < best_cost:
            best_cost = cost
            best_labels = labels.copy()
            best_centroids = centroids.copy()

    # Sort clusters by mean equity (ascending → bucket 0 = weakest)
    centroid_means = [c @ np.linspace(0, 1, len(c)) for c in best_centroids]
    order = np.argsort(centroid_means)
    remap = {old: new for new, old in enumerate(order)}
    sorted_labels = np.array([remap[l] for l in best_labels])
    sorted_centroids = best_centroids[order]

    return sorted_labels, sorted_centroids


# ─────────────────────────────────────────────────────────────
# Main abstraction class
# ─────────────────────────────────────────────────────────────

class CardAbstraction:
    """
    Multi-street card abstraction for NLHE.

    Build once (slow), then query instantly via .bucket().
    Saves/loads from disk to avoid recomputation.
    """

    CACHE_PATH = 'abstraction/cache/card_abstraction.pkl'

    def __init__(self):
        self._pre_buckets: dict   = {}   # (c1, c2) -> bucket_id
        self._flop_centroids: np.ndarray | None = None
        self._turn_centroids: np.ndarray | None = None
        self._n_flop_bins = N_HISTOGRAM_BINS
        self._n_turn_bins = N_HISTOGRAM_BINS

    # ── Build ──────────────────────────────────────────────────

    def build(
        self,
        n_pre_samples:  int = 1000,
        n_flop_samples: int = 150,
        n_turn_samples: int = 200,
        n_flop_hands:   int = 500,
        n_turn_hands:   int = 500,
        verbose: bool = True,
    ):
        """
        Compute all equity data and cluster at each street.
        This is the expensive one-time offline computation.
        """
        os.makedirs(os.path.dirname(self.CACHE_PATH), exist_ok=True)
        t0 = time.time()

        # ── Preflop ──────────────────────────────────────────
        if verbose:
            print("\n[Abstraction] Building preflop buckets...")
        self._build_preflop(n_pre_samples, verbose)

        # ── Flop ─────────────────────────────────────────────
        if verbose:
            print(f"\n[Abstraction] Building flop buckets ({n_flop_hands} hand-board pairs)...")
        self._flop_centroids = self._build_street(
            street=3,
            n_hands=n_flop_hands,
            n_samples=n_flop_samples,
            n_clusters=N_FLOP,
            verbose=verbose,
            label='Flop',
        )

        # ── Turn ──────────────────────────────────────────────
        if verbose:
            print(f"\n[Abstraction] Building turn buckets ({n_turn_hands} hand-board pairs)...")
        self._turn_centroids = self._build_street(
            street=4,
            n_hands=n_turn_hands,
            n_samples=n_turn_samples,
            n_clusters=N_TURN,
            verbose=verbose,
            label='Turn',
        )

        elapsed = time.time() - t0
        if verbose:
            print(f"\n[Abstraction] Complete in {elapsed:.1f}s")
            self._print_preflop_summary()

        self._save()

    def _build_preflop(self, n_samples: int, verbose: bool):
        hands = canonical_preflop_hands()
        equities = []

        for i, (c1, c2, suited) in enumerate(hands):
            eq = preflop_equity([c1, c2], n_samples=n_samples)
            equities.append(eq)
            if verbose and (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(hands)} hands done...")

        equities = np.array(equities)

        # Simple percentile bucketing on scalar equity
        # (k-means on a scalar is equivalent to equal-frequency binning)
        sorted_idx = np.argsort(equities)
        bucket_size = len(hands) // N_PRE
        labels = np.zeros(len(hands), dtype=int)
        for i, idx in enumerate(sorted_idx):
            labels[idx] = min(i // bucket_size, N_PRE - 1)

        for i, (c1, c2, suited) in enumerate(hands):
            key = (min(c1, c2), max(c1, c2))
            self._pre_buckets[key] = int(labels[i])

        if verbose:
            print(f"  Preflop: {len(hands)} hands → {N_PRE} buckets")
            # Show bucket equity ranges
            for b in range(N_PRE):
                eqs = [equities[i] for i, (c1, c2, s) in enumerate(hands)
                       if labels[i] == b]
                if eqs:
                    print(f"    Bucket {b}: eq {min(eqs):.3f} – {max(eqs):.3f}")

    def _build_street(
        self,
        street: int,
        n_hands: int,
        n_samples: int,
        n_clusters: int,
        verbose: bool,
        label: str,
    ) -> np.ndarray:
        """
        Sample hand-board pairs, compute equity histograms, cluster with EMD.
        Returns centroid histograms for later lookup.
        """
        histograms = []

        for i in range(n_hands):
            # Sample a random hole + board
            sample = random.sample(ALL_CARDS, 2 + street)
            hole = sample[:2]
            board = sample[2:]
            hist = equity_histogram(hole, board, n_samples=n_samples)
            histograms.append(hist)

            if verbose and (i + 1) % 100 == 0:
                print(f"  {i+1}/{n_hands} {label} histograms done...")

        histograms = np.array(histograms)

        labels, centroids = kmeans_emd(histograms, n_clusters)

        if verbose:
            print(f"  {label}: {n_hands} samples → {n_clusters} clusters")
            for k in range(n_clusters):
                count = (labels == k).sum()
                centroid_mean = centroids[k] @ np.linspace(0, 1, N_HISTOGRAM_BINS)
                print(f"    Cluster {k}: {count} hands, centroid equity ≈ {centroid_mean:.3f}")

        return centroids

    # ── Query ──────────────────────────────────────────────────

    def bucket(self, hole: list[int], board: list[int]) -> int:
        """
        Return the bucket ID for a given (hole, board) combination.
        Board length determines the street:
          0 → preflop
          3 → flop
          4 → turn
          5 → river
        """
        street = len(board)

        if street == 0:
            return self._preflop_bucket(hole)
        elif street == 3:
            return self._histogram_bucket(hole, board, self._flop_centroids, n_samples=100)
        elif street == 4:
            return self._histogram_bucket(hole, board, self._turn_centroids, n_samples=150)
        elif street == 5:
            return self._river_bucket(hole, board)
        else:
            raise ValueError(f"Invalid board length: {street}")

    def _preflop_bucket(self, hole: list[int]) -> int:
        key = (min(hole[0], hole[1]), max(hole[0], hole[1]))
        return self._pre_buckets.get(key, N_PRE - 1)

    def _histogram_bucket(
        self,
        hole: list[int],
        board: list[int],
        centroids: np.ndarray,
        n_samples: int,
    ) -> int:
        hist = equity_histogram(hole, board, n_samples=n_samples)
        dists = [emd(hist, c) for c in centroids]
        return int(np.argmin(dists))

    def _river_bucket(self, hole: list[int], board: list[int]) -> int:
        strength = river_strength(hole, board)
        # Map to [0, N_RIVER-1]
        bucket = int(strength * N_RIVER)
        return min(bucket, N_RIVER - 1)

    # ── Serialization ─────────────────────────────────────────

    def _save(self):
        with open(self.CACHE_PATH, 'wb') as f:
            pickle.dump({
                'pre_buckets': self._pre_buckets,
                'flop_centroids': self._flop_centroids,
                'turn_centroids': self._turn_centroids,
            }, f)
        print(f"[Abstraction] Saved to {self.CACHE_PATH}")

    def load(self) -> bool:
        if not os.path.exists(self.CACHE_PATH):
            return False
        with open(self.CACHE_PATH, 'rb') as f:
            data = pickle.load(f)
        self._pre_buckets = data['pre_buckets']
        self._flop_centroids = data['flop_centroids']
        self._turn_centroids = data['turn_centroids']
        print(f"[Abstraction] Loaded from {self.CACHE_PATH}")
        return True

    # ── Diagnostics ───────────────────────────────────────────

    def _print_preflop_summary(self):
        from treys import Card
        bucket_hands = defaultdict(list)
        hands = canonical_preflop_hands()
        for c1, c2, suited in hands:
            key = (min(c1, c2), max(c1, c2))
            b = self._pre_buckets.get(key, -1)
            r1 = Card.int_to_str(c1)[0]
            r2 = Card.int_to_str(c2)[0]
            suffix = 's' if suited else ('p' if r1 == r2 else 'o')
            bucket_hands[b].append(f"{r1}{r2}{suffix}")

        print("\n[Abstraction] Preflop bucket assignments:")
        for b in sorted(bucket_hands):
            hands_str = ', '.join(bucket_hands[b][:8])
            more = f'... (+{len(bucket_hands[b])-8})' if len(bucket_hands[b]) > 8 else ''
            print(f"  Bucket {b}: {hands_str}{more}")
