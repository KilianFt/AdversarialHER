"""Replay Buffer with support for symmetric multi-agent transitions, dual RLPD sampling, and 50/50 mode balancing."""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import torch


class _RingBuffer:
    """Fast, contiguous numpy ring buffer for a single mode."""

    def __init__(self, obs_dim: int, act_dim: int, capacity: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.nobs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add_batch(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: np.ndarray,
        nobs: np.ndarray,
        done: np.ndarray,
    ):
        n = obs.shape[0]
        if n == 0:
            return

        end_idx = self.ptr + n
        if end_idx <= self.capacity:
            self.obs[self.ptr:end_idx] = obs
            self.act[self.ptr:end_idx] = act
            self.rew[self.ptr:end_idx] = rew.reshape(-1, 1)
            self.nobs[self.ptr:end_idx] = nobs
            self.done[self.ptr:end_idx] = done.reshape(-1, 1).astype(np.float32)
        else:
            first = self.capacity - self.ptr
            second = n - first
            self.obs[self.ptr:] = obs[:first]
            self.act[self.ptr:] = act[:first]
            self.rew[self.ptr:] = rew[:first].reshape(-1, 1)
            self.nobs[self.ptr:] = nobs[:first]
            self.done[self.ptr:] = done[:first].reshape(-1, 1).astype(np.float32)

            self.obs[:second] = obs[first:]
            self.act[:second] = act[first:]
            self.rew[:second] = rew[first:].reshape(-1, 1)
            self.nobs[:second] = nobs[first:]
            self.done[:second] = done[first:].reshape(-1, 1).astype(np.float32)

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample_indices(self, n: int) -> np.ndarray:
        return np.random.randint(0, self.size, size=n)


class ReplayBuffer:
    """Numpy-backed replay buffer supporting mode-balanced sampling and dual RLPD sampling.

    Features:
    - Mode-balanced sampling (50% Goal-Seeking Mode 0, 50% Adversarial Mode 1) to prevent
      Goal HER from overwhelming Adversarial training gradients.
    - Symmetric multi-agent ingestion.
    - Dual-buffer sampling (online + prior data) for RLPD.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        capacity: int = 500000,
        device: torch.device = torch.device("cpu"),
        mode_balanced: bool = True,
        mode_sampling_ratio: float = 0.5,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.capacity = capacity
        self.device = device
        self.mode_balanced = mode_balanced
        self.mode_sampling_ratio = mode_sampling_ratio

        # Dedicated sub-buffers for Mode 0 (Goal) and Mode 1 (Adversarial)
        half_cap = capacity // 2
        self.buffer_m0 = _RingBuffer(obs_dim, act_dim, half_cap)
        self.buffer_m1 = _RingBuffer(obs_dim, act_dim, half_cap)

    @property
    def mode0_size(self) -> int:
        return self.buffer_m0.size

    @property
    def mode1_size(self) -> int:
        return self.buffer_m1.size

    @property
    def mode0_ratio(self) -> float:
        total = self.size
        return float(self.buffer_m0.size / total) if total > 0 else 0.5

    @property
    def mode1_ratio(self) -> float:
        total = self.size
        return float(self.buffer_m1.size / total) if total > 0 else 0.5

    @property
    def size(self) -> int:
        return self.buffer_m0.size + self.buffer_m1.size

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Add a single transition."""
        self.add_batch(
            obs[np.newaxis, :],
            action[np.newaxis, :],
            np.array([reward], dtype=np.float32),
            next_obs[np.newaxis, :],
            np.array([done], dtype=bool),
        )

    def add_batch(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
    ):
        """Add a batch of transitions, dispatching to respective mode sub-buffers."""
        if len(obs) == 0:
            return

        # Mode indicator is at index 13 in the egocentric observation space (0.0=Goal, 1.0=Adversarial)
        if obs.shape[1] > 13:
            m1_mask = (obs[:, 13] >= 0.5)
            m0_mask = ~m1_mask
        else:
            m0_mask = np.ones(len(obs), dtype=bool)
            m1_mask = np.zeros(len(obs), dtype=bool)

        if np.any(m0_mask):
            self.buffer_m0.add_batch(
                obs[m0_mask], action[m0_mask], reward[m0_mask], next_obs[m0_mask], done[m0_mask]
            )
        if np.any(m1_mask):
            self.buffer_m1.add_batch(
                obs[m1_mask], action[m1_mask], reward[m1_mask], next_obs[m1_mask], done[m1_mask]
            )


    def _sample_internal(self, batch_size: int) -> Dict[str, np.ndarray]:
        s0 = self.buffer_m0.size
        s1 = self.buffer_m1.size

        if self.mode_balanced and s0 > 0 and s1 > 0:
            b0 = int(round(batch_size * self.mode_sampling_ratio))
            b1 = batch_size - b0
            idx0 = self.buffer_m0.sample_indices(b0)
            idx1 = self.buffer_m1.sample_indices(b1)

            obs = np.concatenate([self.buffer_m0.obs[idx0], self.buffer_m1.obs[idx1]])
            acts = np.concatenate([self.buffer_m0.act[idx0], self.buffer_m1.act[idx1]])
            rews = np.concatenate([self.buffer_m0.rew[idx0], self.buffer_m1.rew[idx1]])
            nobs = np.concatenate([self.buffer_m0.nobs[idx0], self.buffer_m1.nobs[idx1]])
            dones = np.concatenate([self.buffer_m0.done[idx0], self.buffer_m1.done[idx1]])
        elif s0 > 0:
            idx0 = self.buffer_m0.sample_indices(batch_size)
            obs = self.buffer_m0.obs[idx0]
            acts = self.buffer_m0.act[idx0]
            rews = self.buffer_m0.rew[idx0]
            nobs = self.buffer_m0.nobs[idx0]
            dones = self.buffer_m0.done[idx0]
        elif s1 > 0:
            idx1 = self.buffer_m1.sample_indices(batch_size)
            obs = self.buffer_m1.obs[idx1]
            acts = self.buffer_m1.act[idx1]
            rews = self.buffer_m1.rew[idx1]
            nobs = self.buffer_m1.nobs[idx1]
            dones = self.buffer_m1.done[idx1]
        else:
            raise RuntimeError("Cannot sample from an empty ReplayBuffer!")

        return {"obs": obs, "acts": acts, "rews": rews, "nobs": nobs, "dones": dones}

    def sample(
        self, batch_size: int, prior_buffer: Optional[ReplayBuffer] = None
    ) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        if prior_buffer is not None and len(prior_buffer) > 0:
            online_batch = batch_size // 2
            prior_batch = batch_size - online_batch

            b_online = self._sample_internal(online_batch)
            b_prior = prior_buffer._sample_internal(prior_batch)

            obs = np.concatenate([b_online["obs"], b_prior["obs"]])
            acts = np.concatenate([b_online["acts"], b_prior["acts"]])
            rews = np.concatenate([b_online["rews"], b_prior["rews"]])
            nobs = np.concatenate([b_online["nobs"], b_prior["nobs"]])
            dones = np.concatenate([b_online["dones"], b_prior["dones"]])
        else:
            b = self._sample_internal(batch_size)
            obs, acts, rews, nobs, dones = b["obs"], b["acts"], b["rews"], b["nobs"], b["dones"]

        return {
            "observations": torch.as_tensor(obs, device=self.device),
            "actions": torch.as_tensor(acts, device=self.device),
            "rewards": torch.as_tensor(rews, device=self.device),
            "next_observations": torch.as_tensor(nobs, device=self.device),
            "dones": torch.as_tensor(dones, device=self.device),
        }

