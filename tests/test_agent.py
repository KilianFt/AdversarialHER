"""Unit tests for LayerNorm Critic Ensemble, Actor, and RLPDAgent."""

import numpy as np
import pytest
import torch
from models import CriticEnsemble, LayerNormMLP, RLPDAgent, SquashedGaussianPolicy
from replay import ReplayBuffer


def test_layernorm_critic_ensemble():
    obs_dim = 17
    act_dim = 2
    num_critics = 3
    critics = CriticEnsemble(obs_dim=obs_dim, act_dim=act_dim, num_critics=num_critics)

    batch_size = 8
    obs = torch.randn(batch_size, obs_dim)
    act = torch.randn(batch_size, act_dim)

    q_vals = critics(obs, act)
    assert q_vals.shape == (num_critics, batch_size, 1)

    q1_val = critics.q1(obs, act)
    assert q1_val.shape == (batch_size, 1)


def test_squashed_gaussian_policy():
    obs_dim = 17
    act_dim = 2
    policy = SquashedGaussianPolicy(obs_dim=obs_dim, act_dim=act_dim)

    batch_size = 8
    obs = torch.randn(batch_size, obs_dim)

    # Stochastic forward
    action, log_prob = policy(obs, deterministic=False, with_log_prob=True)
    assert action.shape == (batch_size, act_dim)
    assert log_prob.shape == (batch_size, 1)
    # Tanh bounds
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)

    # Deterministic forward
    det_action, _ = policy(obs, deterministic=True, with_log_prob=False)
    assert det_action.shape == (batch_size, act_dim)
    assert torch.all(det_action >= -1.0) and torch.all(det_action <= 1.0)


def test_rlpd_agent_update():
    obs_dim = 17
    act_dim = 2
    agent = RLPDAgent(obs_dim=obs_dim, act_dim=act_dim, num_critics=2)
    buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, capacity=500)

    for _ in range(50):
        obs = np.random.randn(obs_dim).astype(np.float32)
        act = np.random.uniform(-1, 1, size=act_dim).astype(np.float32)
        rew = float(np.random.randn())
        next_obs = np.random.randn(obs_dim).astype(np.float32)
        buffer.add(obs, act, rew, next_obs, False)

    # Update with UTD ratio = 3
    metrics = agent.update_rlpd(buffer, utd_ratio=3, batch_size=16)
    assert "critic_loss" in metrics
    assert "actor_loss" in metrics
    assert "alpha" in metrics
    assert metrics["critic_loss"] >= 0.0


def test_mode_balanced_replay_buffer():
    """Verify that mode-balanced buffer guarantees 50/50 batch sampling despite skewed ingestion."""
    obs_dim = 17
    act_dim = 2
    buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, capacity=1000, mode_balanced=True, mode_sampling_ratio=0.5)

    # Ingest heavily imbalanced data: 95 Mode 0 transitions, only 5 Mode 1 transitions
    obs_m0 = np.zeros((95, obs_dim), dtype=np.float32)
    obs_m0[:, 13] = 0.0
    act_m0 = np.zeros((95, act_dim), dtype=np.float32)
    rew_m0 = np.zeros(95, dtype=np.float32)

    obs_m1 = np.zeros((5, obs_dim), dtype=np.float32)
    obs_m1[:, 13] = 1.0
    act_m1 = np.zeros((5, act_dim), dtype=np.float32)
    rew_m1 = np.zeros(5, dtype=np.float32)

    buffer.add_batch(obs_m0, act_m0, rew_m0, obs_m0, np.zeros(95, dtype=bool))
    buffer.add_batch(obs_m1, act_m1, rew_m1, obs_m1, np.zeros(5, dtype=bool))

    assert buffer.mode0_size == 95
    assert buffer.mode1_size == 5
    assert abs(buffer.mode0_ratio - 0.95) < 1e-4

    # Sample batch of 30 transitions
    batch = buffer.sample(batch_size=30)
    sampled_modes = batch["observations"][:, 13].cpu().numpy()
    m0_count = np.sum(sampled_modes < 0.5)
    m1_count = np.sum(sampled_modes >= 0.5)

    # Must be exactly 15 Mode 0 and 15 Mode 1!
    assert m0_count == 15
    assert m1_count == 15


def test_duality_critic_ensemble():
    """Verify DualityCriticEnsemble factorized heads and sign reversal."""
    obs_dim = 33
    act_dim = 2
    num_critics = 3
    from models import DualityCriticEnsemble
    critics = DualityCriticEnsemble(obs_dim=obs_dim, act_dim=act_dim, num_critics=num_critics)

    batch_size = 6
    obs = torch.randn(batch_size, obs_dim)
    # 3 in Mode 0, 3 in Mode 1
    obs[:3, 13] = 0.0
    obs[3:, 13] = 1.0
    act = torch.randn(batch_size, act_dim)

    # Test forward
    q_eff = critics(obs, act)
    assert q_eff.shape == (num_critics, batch_size, 1)

    # Test components
    q_nav, q_coll = critics.forward_components(obs, act)
    assert q_nav.shape == (num_critics, batch_size, 1)
    assert q_coll.shape == (num_critics, batch_size, 1)

    # Collision potential must be strictly bounded in [0, 1] (Sigmoid)
    assert torch.all(q_coll >= 0.0) and torch.all(q_coll <= 1.0)

    # Verify algebraic sign composition:
    # In Mode 0: Q_eff = Q_nav - 2.5 * Q_coll
    # In Mode 1: Q_eff = Q_nav + 5.0 * Q_coll
    diff_m0 = q_eff[:, :3] - (q_nav[:, :3] - 2.5 * q_coll[:, :3])
    diff_m1 = q_eff[:, 3:] - (q_nav[:, 3:] + 5.0 * q_coll[:, 3:])
    assert torch.allclose(diff_m0, torch.zeros_like(diff_m0), atol=1e-5)
    assert torch.allclose(diff_m1, torch.zeros_like(diff_m1), atol=1e-5)

    # Test q1
    q1 = critics.q1(obs, act)
    assert q1.shape == (batch_size, 1)


def test_rlpd_agent_duality_update():
    """Verify RLPDAgent gradient updates with decoupled duality critic."""
    obs_dim = 33
    act_dim = 2
    agent = RLPDAgent(obs_dim=obs_dim, act_dim=act_dim, num_critics=2, critic_type="duality")
    buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, capacity=500)

    for i in range(50):
        obs = np.random.randn(obs_dim).astype(np.float32)
        obs[13] = 1.0 if (i % 2 == 1) else 0.0
        act = np.random.uniform(-1, 1, size=act_dim).astype(np.float32)
        rew = float(np.random.randn())
        next_obs = np.random.randn(obs_dim).astype(np.float32)
        next_obs[13] = obs[13]
        coll = 1.0 if (i % 5 == 0) else 0.0
        buffer.add(obs, act, rew, next_obs, False, coll=coll)

    metrics = agent.update_rlpd(buffer, utd_ratio=2, batch_size=16)
    assert "critic_loss" in metrics
    assert "critic_nav_loss" in metrics
    assert "critic_coll_loss" in metrics
    assert "mean_q_coll" in metrics
    assert "mean_q_nav" in metrics
    assert "actor_loss" in metrics
    assert metrics["critic_loss"] >= 0.0
    assert 0.0 <= metrics["mean_q_coll"] <= 1.0


def test_rlpd_agent_save_load_duality(tmp_path):
    """Verify save and load roundtrip with duality critic."""
    obs_dim = 33
    act_dim = 2
    agent = RLPDAgent(obs_dim=obs_dim, act_dim=act_dim, num_critics=2, critic_type="duality")
    save_path = str(tmp_path / "test_duality.pt")
    agent.save(save_path)

    new_agent = RLPDAgent(obs_dim=obs_dim, act_dim=act_dim, num_critics=2, critic_type="standard")
    new_agent.load(save_path)
    assert new_agent.critic_type == "duality"


