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
