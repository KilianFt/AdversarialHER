"""Network architectures for RLPD: LayerNorm Critic Ensemble and Squashed Gaussian Policy."""

from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class LayerNormMLP(nn.Module):
    """MLP layer with optional LayerNorm after each linear layer before non-linearity.

    Essential for RLPD to stabilize value learning under high Update-To-Data (UTD) ratios.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int] = [256, 256],
        use_layernorm: bool = True,
    ):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.Mish())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CriticEnsemble(nn.Module):
    """Ensemble of Q-networks with optional LayerNorm."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int] = [256, 256],
        num_critics: int = 2,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.num_critics = num_critics
        self.critics = nn.ModuleList([
            LayerNormMLP(obs_dim + act_dim, 1, hidden_dims, use_layernorm=use_layernorm)
            for _ in range(num_critics)
        ])


    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Evaluate all critics on (obs, act).

        Returns tensor of shape (num_critics, batch_size, 1).
        """
        x = torch.cat([obs, act], dim=-1)
        q_values = [critic(x) for critic in self.critics]
        return torch.stack(q_values, dim=0)

    def q1(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """First critic value."""
        x = torch.cat([obs, act], dim=-1)
        return self.critics[0](x)


class DualityCriticEnsemble(nn.Module):
    """Full-State Cost-Decomposed Critic Ensemble for Safe Multi-Agent RL.

    Architecture:
        Q_eff(s, a, mode) = Q_nav(s, a) + sigma(mode) * Q_coll(s, a)

    Inductive Bias:
        - Q_nav(s, a): Receives FULL observation (including lidar, goal, heading, opponent).
                       Learns obstacle-conditioned task progress to the goal.
        - Q_coll(s, a): Receives FULL observation. Predicts calibrated collision probability in [0, 1].
        - sigma(mode): +collision_reward (+5.0) in Mode 1 (Adversarial Attractor),
                       -loss_penalty (-2.5) in Mode 0 (Obstacle/Opponent Repeller).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int] = [256, 256],
        num_critics: int = 2,
        use_layernorm: bool = True,
        collision_reward: float = 5.0,
        loss_penalty: float = 2.5,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dims = hidden_dims
        self.num_critics = num_critics
        self.use_layernorm = use_layernorm
        self.collision_reward = collision_reward
        self.loss_penalty = loss_penalty

        in_dim = obs_dim + act_dim

        # Navigation ensemble (unbounded Q-values for distance progress & control cost)
        self.nav_critics = nn.ModuleList([
            LayerNormMLP(in_dim, 1, hidden_dims, use_layernorm=use_layernorm)
            for _ in range(num_critics)
        ])

        # Collision risk ensemble (strictly bounded in [0, 1] via Sigmoid)
        self.coll_critics = nn.ModuleList([
            nn.Sequential(
                LayerNormMLP(in_dim, 1, hidden_dims, use_layernorm=use_layernorm),
                nn.Sigmoid(),
            )
            for _ in range(num_critics)
        ])

    def forward_components(
        self, obs: torch.Tensor, act: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (q_navs, q_colls) each of shape (num_critics, batch_size, 1)."""
        x = torch.cat([obs, act], dim=-1)
        q_navs = torch.stack([c(x) for c in self.nav_critics], dim=0)
        q_colls = torch.stack([c(x) for c in self.coll_critics], dim=0)
        return q_navs, q_colls

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Evaluate all critics on (obs, act), returning Q_eff = Q_nav + sigma(mode) * Q_coll.

        Returns tensor of shape (num_critics, batch_size, 1).
        """
        x = torch.cat([obs, act], dim=-1)
        if obs.shape[-1] > 13:
            mode = obs[..., 13:14]
        else:
            mode = torch.zeros((*obs.shape[:-1], 1), device=obs.device, dtype=obs.dtype)

        sigma = torch.where(mode >= 0.5, self.collision_reward, -self.loss_penalty)
        q_navs = torch.stack([c(x) for c in self.nav_critics], dim=0)
        q_colls = torch.stack([c(x) for c in self.coll_critics], dim=0)
        return q_navs + sigma * q_colls

    def q1(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """First critic value."""
        x = torch.cat([obs, act], dim=-1)
        if obs.shape[-1] > 13:
            mode = obs[..., 13:14]
        else:
            mode = torch.zeros((*obs.shape[:-1], 1), device=obs.device, dtype=obs.dtype)

        sigma = torch.where(mode >= 0.5, self.collision_reward, -self.loss_penalty)
        return self.nav_critics[0](x) + sigma * self.coll_critics[0](x)


class SquashedGaussianPolicy(nn.Module):
    """Continuous action policy with Tanh squashing."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: List[int] = [256, 256],
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.act_dim = act_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers = []
        prev_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.Mish())
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, act_dim)
        self.log_std_head = nn.Linear(prev_dim, act_dim)

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False, with_log_prob: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        features = self.backbone(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if deterministic:
            action = torch.tanh(mean)
            return action, None

        normal = Normal(mean, std)
        # Reparameterization trick
        u = normal.rsample()
        action = torch.tanh(u)

        if with_log_prob:
            # Compute squashed Gaussian log probability with numerical stability
            log_prob = normal.log_prob(u).sum(dim=-1, keepdim=True)
            # Correction for tanh squashing: log(1 - tanh(u)^2 + eps)
            correction = torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
            log_prob = log_prob - correction
        else:
            log_prob = None

        return action, log_prob
