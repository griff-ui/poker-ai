"""
Deep CFR Neural Networks — Stage 4

Two networks per player (following Brown & Sandholm 2019):

1. AdvantageNetwork: approximates cumulative counterfactual regret
   Input:  info set features (373-dim)
   Output: advantage estimate per action (|A| outputs)
   Used:   during CFR traversal to compute regret-matching strategy

2. StrategyNetwork: approximates the average strategy
   Input:  info set features (373-dim)
   Output: probability distribution over actions
   Used:   final policy extraction at convergence

Both are simple feedforward networks. The paper used relatively shallow
architectures — deeper is not always better here because the abstraction
(card bucketing) already reduces complexity. We use:
  3 hidden layers × 256 units, ReLU, LayerNorm for training stability.

Reservoir Buffer: stores (feature, advantage, weight) tuples.
  - Fixed capacity with reservoir sampling (uniform over all seen samples)
  - Prevents catastrophic forgetting across CFR iterations
  - Separate buffers per player per network type
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Network architecture
# ─────────────────────────────────────────────────────────────

MAX_ACTIONS = 6   # fold, check/call, 3 bet sizes, all-in (matches traversal.py)


class PokerNet(nn.Module):
    """
    Feedforward network for both advantage and strategy approximation.

    Architecture:
        Linear(in_dim → 256) → LayerNorm → ReLU
        Linear(256 → 256)    → LayerNorm → ReLU
        Linear(256 → 256)    → LayerNorm → ReLU
        Linear(256 → out_dim)

    LayerNorm instead of BatchNorm: works better with small batch sizes
    and doesn't depend on batch statistics at inference time.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, n_layers: int = 3):
        super().__init__()
        layers = []
        prev = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU())
            prev = hidden
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdvantageNetwork(PokerNet):
    """
    Approximates cumulative counterfactual advantages (regrets).
    Output is raw (can be negative) — regret-matching clips to positive.
    """
    def __init__(self, feature_dim: int, n_actions: int):
        super().__init__(feature_dim, n_actions)
        self.n_actions = n_actions

    def regret_matching(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute current strategy via regret matching on network output.
        σ(a) ∝ max(advantage(a), 0)
        """
        advantages = self.forward(features)
        pos = F.relu(advantages)
        total = pos.sum(dim=-1, keepdim=True)
        # Uniform if all advantages non-positive
        n = advantages.shape[-1]
        uniform = torch.ones_like(advantages) / n
        strategy = torch.where(total > 1e-6, pos / total, uniform)
        return strategy


class StrategyNetwork(PokerNet):
    """
    Approximates the average strategy (converges to Nash equilibrium).
    Output is a probability distribution (softmax).
    """
    def __init__(self, feature_dim: int, n_actions: int):
        super().__init__(feature_dim, n_actions)
        self.n_actions = n_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = super().forward(x)
        return F.softmax(logits, dim=-1)


# ─────────────────────────────────────────────────────────────
# Reservoir Buffer
# ─────────────────────────────────────────────────────────────

@dataclass
class ReservoirSample:
    features:   np.ndarray    # info set feature vector
    advantages: np.ndarray    # per-action advantage estimates
    weight:     float         # iteration weight (for linear CFR weighting)
    n_actions:  int           # number of valid actions at this info set


class ReservoirBuffer:
    """
    Fixed-capacity reservoir buffer with uniform sampling guarantee.

    In Deep CFR, we need to ensure that past data is represented uniformly
    to avoid the network overfitting to recent iterations. Reservoir sampling
    achieves this: every sample seen has equal probability of being retained,
    regardless of when it was added.

    Weight: iteration number t (for linear CFR, more recent data is weighted
    more heavily via the iteration multiplier in regret updates).
    """

    def __init__(self, capacity: int = 2_000_000):
        self.capacity  = capacity
        self.buffer: list[ReservoirSample] = []
        self.n_seen    = 0

    def add(self, sample: ReservoirSample):
        """Add with reservoir sampling: uniform random replacement when full."""
        self.n_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            idx = random.randint(0, self.n_seen - 1)
            if idx < self.capacity:
                self.buffer[idx] = sample

    def sample_batch(self, batch_size: int) -> Optional[list[ReservoirSample]]:
        """Sample a random batch. Returns None if buffer is too small."""
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────
# Deep CFR Player: networks + buffers for one player
# ─────────────────────────────────────────────────────────────

class DeepCFRPlayer:
    """
    All ML components for one player in Deep CFR:
      - Advantage network (for regret approximation)
      - Strategy network (for final policy)
      - Advantage reservoir buffer
      - Strategy reservoir buffer
    """

    def __init__(
        self,
        player_id: int,
        feature_dim: int,
        n_actions: int,
        hidden_size: int = 256,
        buffer_capacity: int = 500_000,
        device: str = 'cpu',
    ):
        self.player_id   = player_id
        self.feature_dim = feature_dim
        self.n_actions   = n_actions
        self.device      = torch.device(device)

        # Networks
        self.advantage_net = AdvantageNetwork(feature_dim, n_actions).to(self.device)
        self.strategy_net  = StrategyNetwork(feature_dim, n_actions).to(self.device)

        # Buffers
        self.advantage_buf = ReservoirBuffer(buffer_capacity)
        self.strategy_buf  = ReservoirBuffer(buffer_capacity)

        # Training state
        self.adv_optimizer  = torch.optim.Adam(self.advantage_net.parameters(), lr=1e-3)
        self.strat_optimizer = torch.optim.Adam(self.strategy_net.parameters(), lr=1e-3)

    def add_advantage_sample(
        self,
        features: list[float],
        advantages: list[float],
        weight: float,
        n_actions: int,
    ):
        sample = ReservoirSample(
            features   = np.array(features, dtype=np.float32),
            advantages = np.array(advantages, dtype=np.float32),
            weight     = weight,
            n_actions  = n_actions,
        )
        self.advantage_buf.add(sample)

    def add_strategy_sample(
        self,
        features: list[float],
        strategy: list[float],
        weight: float,
        n_actions: int,
    ):
        sample = ReservoirSample(
            features   = np.array(features, dtype=np.float32),
            advantages = np.array(strategy, dtype=np.float32),
            weight     = weight,
            n_actions  = n_actions,
        )
        self.strategy_buf.add(sample)

    def train_advantage_network(
        self,
        batch_size: int = 512,
        n_steps:    int = 300,
    ) -> list[float]:
        """
        Train advantage network on reservoir buffer.
        Loss: weighted MSE between predicted and target advantages.
        Returns list of losses per step.
        """
        return self._train_network(
            net       = self.advantage_net,
            optimizer = self.adv_optimizer,
            buffer    = self.advantage_buf,
            batch_size = batch_size,
            n_steps   = n_steps,
            is_strategy = False,
        )

    def train_strategy_network(
        self,
        batch_size: int = 512,
        n_steps:    int = 300,
    ) -> list[float]:
        """
        Train strategy network on strategy reservoir buffer.
        Loss: weighted cross-entropy between predicted and target strategy.
        """
        return self._train_network(
            net       = self.strategy_net,
            optimizer = self.strat_optimizer,
            buffer    = self.strategy_buf,
            batch_size = batch_size,
            n_steps   = n_steps,
            is_strategy = True,
        )

    def _train_network(
        self,
        net, optimizer, buffer, batch_size, n_steps, is_strategy
    ) -> list[float]:
        if len(buffer) < batch_size:
            return []

        losses = []

        for _ in range(n_steps):
            batch = buffer.sample_batch(batch_size)
            if batch is None:
                break

            feats  = torch.tensor(
                np.stack([s.features for s in batch]),
                dtype=torch.float32, device=self.device
            )
            targets = torch.tensor(
                np.stack([s.advantages for s in batch]),
                dtype=torch.float32, device=self.device
            )
            weights = torch.tensor(
                [s.weight for s in batch],
                dtype=torch.float32, device=self.device
            )

            optimizer.zero_grad()
            preds = net(feats)

            if is_strategy:
                # Weighted cross-entropy: − Σ w·target·log(pred)
                eps = 1e-8
                loss = -(weights.unsqueeze(1) * targets * torch.log(preds + eps)).sum(dim=1).mean()
            else:
                # Weighted MSE
                diff = (preds - targets) ** 2
                loss = (weights.unsqueeze(1) * diff).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        return losses

    @torch.no_grad()
    def get_strategy(self, features: list[float], n_actions: int) -> np.ndarray:
        x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        strat = self.advantage_net.regret_matching(x).squeeze(0).cpu().numpy()
        result = strat[:n_actions]
        total = result.sum()
        if total > 1e-8:
            return result / total
        return np.ones(n_actions, dtype=np.float32) / n_actions

    @torch.no_grad()
    def get_avg_strategy(self, features: list[float], n_actions: int) -> np.ndarray:
        x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.strategy_net(x).squeeze(0).cpu().numpy()
        result = probs[:n_actions]
        total = result.sum()
        if total > 1e-8:
            return result / total
        return np.ones(n_actions, dtype=np.float32) / n_actions

    def set_inference_mode(self):
        """Call once before traversal to lock networks in eval mode."""
        self.advantage_net.eval()
        self.strategy_net.eval()

    def set_training_mode(self):
        """Call before network training."""
        self.advantage_net.train()
        self.strategy_net.train()

    def save(self, path_prefix: str):
        torch.save(self.advantage_net.state_dict(), f"{path_prefix}_adv.pt")
        torch.save(self.strategy_net.state_dict(),  f"{path_prefix}_strat.pt")

    def load(self, path_prefix: str):
        self.advantage_net.load_state_dict(torch.load(f"{path_prefix}_adv.pt", map_location=self.device))
        self.strategy_net.load_state_dict(torch.load(f"{path_prefix}_strat.pt", map_location=self.device))


# ─────────────────────────────────────────────────────────────
# Action encoding utilities
# ─────────────────────────────────────────────────────────────

def encode_actions(actions) -> np.ndarray:
    """Encode action amounts as normalized feature vector."""
    amounts = [a.amount / 100.0 for a in actions]
    return np.array(amounts, dtype=np.float32)
