"""RLPD (Reinforcement Learning with Prior Data) Agent.

Implements high-UTD Soft Actor-Critic with LayerNorm Critic Ensemble,
automatic entropy temperature tuning, and shared multi-agent policy.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.networks import CriticEnsemble, SquashedGaussianPolicy
from replay.replay_buffer import ReplayBuffer


class RLPDAgent:
    """RLPD agent sharing a single policy and critic ensemble across all agents."""

    def __init__(
        self,
        obs_dim: int = 17,
        act_dim: int = 2,
        hidden_dims: List[int] = [256, 256],
        num_critics: int = 10,
        actor_lr: float = 3e-4,

        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        init_temperature: float = 0.2,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        auto_entropy_tuning: bool = True,
        use_layernorm: bool = True,
        loss_type: str = "smooth_l1",
        device: torch.device = torch.device("cpu"),
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.auto_entropy_tuning = auto_entropy_tuning
        self.device = device
        self.num_critics = num_critics
        self.use_layernorm = use_layernorm
        self.loss_type = loss_type

        # 1. Actor network
        self.actor = SquashedGaussianPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=hidden_dims,
        ).to(device)

        # CPU actor for fast rollout inference on Apple Silicon / CPU
        self.cpu_actor = copy.deepcopy(self.actor).to("cpu")
        for param in self.cpu_actor.parameters():
            param.requires_grad = False

        # 2. Critic Ensemble with LayerNorm
        self.critics = CriticEnsemble(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=hidden_dims,
            num_critics=num_critics,
            use_layernorm=use_layernorm,
        ).to(device)

        # Target critic ensemble
        self.target_critics = copy.deepcopy(self.critics).to(device)
        for param in self.target_critics.parameters():
            param.requires_grad = False

        # 3. Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critics.parameters(), lr=critic_lr)

        # 4. Entropy temperature (RLPD default: -dim(action) / 2)
        if target_entropy is None:
            self.target_entropy = -float(act_dim) / 2.0
        else:
            self.target_entropy = target_entropy


        self.log_alpha = torch.tensor(np.log(init_temperature), dtype=torch.float32, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    def select_actions(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select actions for a batch of agents/observations in a single fast forward pass."""
        with torch.no_grad():
            if self.device.type != "cpu":
                actor_net = self.cpu_actor
                dev = torch.device("cpu")
            else:
                actor_net = self.actor
                dev = self.device

            if isinstance(obs, np.ndarray):
                obs_t = torch.from_numpy(obs).to(device=dev, dtype=torch.float32)
            else:
                obs_t = obs.to(device=dev, dtype=torch.float32)
            if obs_t.ndim == 1:
                obs_t = obs_t.unsqueeze(0)
            actions, _ = actor_net(obs_t, deterministic=deterministic, with_log_prob=False)
            return actions.cpu().numpy()


    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action for an agent given its egocentric observation."""
        return self.select_actions(obs, deterministic=deterministic).squeeze(0)



    def update(
        self,
        buffer: ReplayBuffer,
        batch_size: int = 256,
        prior_buffer: Optional[ReplayBuffer] = None,
    ) -> Dict[str, float]:
        """Perform one gradient update step for Critic, Actor, and Temperature."""
        batch = buffer.sample(batch_size=batch_size, prior_buffer=prior_buffer)

        obs = batch["observations"]
        acts = batch["actions"]
        rews = batch["rewards"]
        next_obs = batch["next_observations"]
        dones = batch["dones"]

        # ===================
        # 1. Update Critics
        # ===================
        with torch.no_grad():
            next_acts, next_log_prob = self.actor(next_obs, deterministic=False, with_log_prob=True)
            # Evaluate target critic ensemble on next state-action
            target_qs = self.target_critics(next_obs, next_acts)  # (num_critics, batch_size, 1)

            # Subsample 2 critics for target min
            if self.num_critics > 2:
                subset = np.random.choice(self.num_critics, 2, replace=False)
                min_target_q = torch.min(target_qs[subset], dim=0)[0]
            else:
                min_target_q = torch.min(target_qs, dim=0)[0]

            alpha_val = self.log_alpha.exp()
            target_v = min_target_q - alpha_val * next_log_prob
            q_target = rews + self.gamma * (1.0 - dones) * target_v

        # Evaluate online critics
        current_qs = self.critics(obs, acts)  # (num_critics, batch_size, 1)
        if self.loss_type == "smooth_l1":
            critic_loss = 0.5 * sum(F.smooth_l1_loss(current_qs[i], q_target, beta=1.0) for i in range(self.num_critics))
        else:
            critic_loss = 0.5 * sum(F.mse_loss(current_qs[i], q_target) for i in range(self.num_critics))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ===================
        # 2. Update Actor
        # ===================
        pred_acts, log_prob = self.actor(obs, deterministic=False, with_log_prob=True)
        # Evaluate current critics on predicted actions
        pred_qs = self.critics(obs, pred_acts)
        min_pred_q = torch.min(pred_qs, dim=0)[0]

        actor_loss = (alpha_val.detach() * log_prob - min_pred_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ===================
        # 3. Update Alpha
        # ===================
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.0)

        # ===================
        # 4. Fast Polyak Target Update
        # ===================
        with torch.no_grad():
            c_params = list(self.critics.parameters())
            tc_params = list(self.target_critics.parameters())
            torch._foreach_mul_(tc_params, 1.0 - self.tau)
            torch._foreach_add_(tc_params, c_params, alpha=self.tau)


        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha,
            "mean_q": min_pred_q.mean().item(),
        }

    def update_rlpd(
        self,
        buffer: ReplayBuffer,
        utd_ratio: int = 2,
        batch_size: int = 256,
        prior_buffer: Optional[ReplayBuffer] = None,
    ) -> Dict[str, float]:
        """Perform utd_ratio gradient updates on the shared network."""
        metrics_list = []
        for _ in range(utd_ratio):
            metrics = self.update(buffer=buffer, batch_size=batch_size, prior_buffer=prior_buffer)
            metrics_list.append(metrics)

        # Average metrics across UTD iterations
        avg_metrics = {}
        for k in metrics_list[0].keys():
            avg_metrics[k] = float(np.mean([m[k] for m in metrics_list]))

        # Sync CPU actor weights for fast rollout inference
        if self.device.type != "cpu":
            self.cpu_actor.load_state_dict(self.actor.state_dict())

        return avg_metrics

    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critics": self.critics.state_dict(),
                "target_critics": self.target_critics.state_dict(),
                "log_alpha": self.log_alpha,
            },
            filepath,
        )

    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critics.load_state_dict(checkpoint["critics"])
        self.target_critics.load_state_dict(checkpoint["target_critics"])
        self.log_alpha.data.copy_(checkpoint["log_alpha"].data)
        if self.device.type != "cpu":
            self.cpu_actor.load_state_dict(self.actor.state_dict())

