"""Unit tests for MultiGoal0 environment and egocentric wrapper."""

import numpy as np
import pytest
from envs import MultiGoal0Env, DualModeEnvWrapper, rotate_to_ego, build_ego_observation


def test_env_reset_and_step():
    base_env = MultiGoal0Env()
    env = DualModeEnvWrapper(base_env)

    obs, info = env.reset(seed=123)
    assert len(obs) == 2
    assert obs[0].shape == (33,)
    assert obs[1].shape == (33,)
    assert "agent_positions" in info
    assert "goal_positions" in info
    assert "collision" in info
    assert "modes" in info

    # Take step
    actions = (np.array([0.5, 0.2], dtype=np.float32), np.array([-0.3, -0.1], dtype=np.float32))
    next_obs, rews, term, trunc, next_info = env.step(actions)

    assert len(next_obs) == 2
    assert next_obs[0].shape == (33,)
    assert next_obs[1].shape == (33,)
    assert len(rews) == 2
    assert not term
    assert not trunc
    assert "dist_agents" in next_info


def test_rotate_to_ego():
    # Facing east (theta = 0)
    vec = np.array([1.0, 0.0], dtype=np.float32)
    rotated = rotate_to_ego(vec, rot=0.0)
    np.testing.assert_allclose(rotated, [1.0, 0.0], atol=1e-5)

    # Facing north (theta = pi/2)
    # A vector pointing north [0, 1] in global frame should be straight ahead [1, 0] in body frame
    vec_north = np.array([0.0, 1.0], dtype=np.float32)
    rotated_north = rotate_to_ego(vec_north, rot=np.pi / 2)
    np.testing.assert_allclose(rotated_north, [1.0, 0.0], atol=1e-5)


def test_ego_observation_mode_switch():
    pos_self = np.array([0.0, 0.0], dtype=np.float32)
    rot_self = 0.0
    vel_self = np.zeros(2, dtype=np.float32)
    rot_vel_self = 0.0
    pos_other = np.array([2.0, 0.0], dtype=np.float32)
    vel_other = np.zeros(2, dtype=np.float32)
    goal_pos = np.array([0.0, 3.0], dtype=np.float32)

    # Mode 0: Goal Seeking -> target should be goal_pos (0, 3)
    obs_m0 = build_ego_observation(
        pos_self, rot_self, vel_self, rot_vel_self,
        pos_other, vel_other, goal_pos, mode=0.0
    )
    # target_ego is at indices [14, 15], dist_target at index 16
    np.testing.assert_allclose(obs_m0[14:16], [0.0, 3.0], atol=1e-5)
    np.testing.assert_allclose(obs_m0[16], 3.0, atol=1e-5)

    # Mode 1: Adversarial -> target should be other agent (2, 0)
    obs_m1 = build_ego_observation(
        pos_self, rot_self, vel_self, rot_vel_self,
        pos_other, vel_other, goal_pos, mode=1.0
    )
    np.testing.assert_allclose(obs_m1[14:16], [2.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(obs_m1[16], 2.0, atol=1e-5)


def test_collision_detection():
    base_env = MultiGoal0Env(collision_threshold=0.3)
    base_env.reset()

    # Move agents right on top of each other
    base_env.data.qpos[base_env.agent0_x_id] = 0.0
    base_env.data.qpos[base_env.agent0_y_id] = 0.0
    base_env.data.qpos[base_env.agent1_x_id] = 0.1  # dist = 0.1 <= 0.3
    base_env.data.qpos[base_env.agent1_y_id] = 0.0

    info = base_env._get_info()
    assert info["collision"] is True
    assert info["dist_agents"] <= 0.3


def test_dwell_time_goal_success():
    base_env = MultiGoal0Env()
    env = DualModeEnvWrapper(base_env, dwell_steps=20, goal_reward=5.0)
    env.reset()
    env.modes = np.array([0.0, 0.0], dtype=np.float32)

    # Place agent 0 inside its goal
    g0 = env.goals[0]
    base_env.data.qpos[base_env.agent0_x_id] = g0[0]
    base_env.data.qpos[base_env.agent0_y_id] = g0[1]

    # Place agent 1 far away
    base_env.data.qpos[base_env.agent1_x_id] = -2.0
    base_env.data.qpos[base_env.agent1_y_id] = -2.0

    no_act = np.zeros(2, dtype=np.float32)

    # Step for 10 steps (dwell counter should be 10, not yet achieved)
    for _ in range(10):
        _, rews, term, _, info = env.step((no_act, no_act))
        base_env.data.qpos[base_env.agent0_x_id] = g0[0]
        base_env.data.qpos[base_env.agent0_y_id] = g0[1]

    assert info["dwell_counters"][0] == 10
    assert not info["goal_achieved"][0]
    assert not term

    # Step 10 more steps to complete 20 steps dwell
    for step in range(10):
        _, rews, term, _, info = env.step((no_act, no_act))
        base_env.data.qpos[base_env.agent0_x_id] = g0[0]
        base_env.data.qpos[base_env.agent0_y_id] = g0[1]

    assert info["dwell_counters"][0] >= 20
    assert bool(info["goal_achieved"][0]) is True
    # In Case [0, 0], episode does NOT terminate until agent 1 also succeeds!
    assert term is False


def test_case1_both_goal_seeking_termination():
    base_env = MultiGoal0Env()
    env = DualModeEnvWrapper(base_env, dwell_steps=5, goal_reward=5.0)
    env.reset()
    env.modes = np.array([0.0, 0.0], dtype=np.float32)

    g0 = env.goals[0]
    g1 = env.goals[1]
    no_act = np.zeros(2, dtype=np.float32)

    # Put both agents in their respective goals
    for _ in range(5):
        base_env.data.qpos[base_env.agent0_x_id] = g0[0]
        base_env.data.qpos[base_env.agent0_y_id] = g0[1]
        base_env.data.qpos[base_env.agent1_x_id] = g1[0]
        base_env.data.qpos[base_env.agent1_y_id] = g1[1]
        _, rews, term, _, info = env.step((no_act, no_act))

    assert bool(info["goal_achieved"][0]) is True
    assert bool(info["goal_achieved"][1]) is True
    assert term is True
    assert info["winner"] == "both"



def test_case2_asymmetric_adversarial_win():
    base_env = MultiGoal0Env()
    env = DualModeEnvWrapper(base_env, collision_reward=5.0, loss_penalty=2.5)
    env.reset()
    # Agent 0 is Adversarial (1.0), Agent 1 is Goal-seeking (0.0)
    env.modes = np.array([1.0, 0.0], dtype=np.float32)

    # Cause collision
    base_env.data.qpos[base_env.agent0_x_id] = 0.0
    base_env.data.qpos[base_env.agent0_y_id] = 0.0
    base_env.data.qpos[base_env.agent1_x_id] = 0.1
    base_env.data.qpos[base_env.agent1_y_id] = 0.0

    no_act = np.zeros(2, dtype=np.float32)
    _, rews, term, _, info = env.step((no_act, no_act))

    assert term is True
    assert info["collision"] is True
    assert info["winner"] == "agent_0_adversarial"
    # Adversarial agent wins (+5.0 bonus), Goal-seeking agent loses (-2.5 penalty)
    assert rews[0] > 4.0
    assert rews[1] < -2.0


def test_case2_asymmetric_goal_win():
    base_env = MultiGoal0Env()
    env = DualModeEnvWrapper(base_env, dwell_steps=5, goal_reward=5.0, loss_penalty=2.5)
    env.reset()
    # Agent 0 is Adversarial (1.0), Agent 1 is Goal-seeking (0.0)
    env.modes = np.array([1.0, 0.0], dtype=np.float32)

    g1 = env.goals[1]
    no_act = np.zeros(2, dtype=np.float32)

    # Keep Agent 0 far away, place Agent 1 in its goal for 5 steps
    for _ in range(5):
        base_env.data.qpos[base_env.agent0_x_id] = -2.0
        base_env.data.qpos[base_env.agent0_y_id] = -2.0
        base_env.data.qpos[base_env.agent1_x_id] = g1[0]
        base_env.data.qpos[base_env.agent1_y_id] = g1[1]
        _, rews, term, _, info = env.step((no_act, no_act))

    assert term is True
    assert info["winner"] == "agent_1_goal"
    # Goal-seeking agent 1 wins (+5.0 bonus), Adversarial agent 0 loses (-2.5 penalty)
    assert rews[1] > 4.0
    assert rews[0] < -2.0


def test_hazards_lidar_perception():
    from envs.wrappers import compute_hazards_lidar

    pos_self = np.array([0.0, 0.0], dtype=np.float32)
    rot_self = 0.0  # Facing East (positive x)
    # Hazard directly in front at (1.0, 0.0)
    hazards = np.array([[1.0, 0.0]], dtype=np.float32)

    lidar = compute_hazards_lidar(pos_self, rot_self, hazards, hazard_radius=0.2, num_bins=16)
    assert lidar.shape == (16,)
    # Bin 0 corresponds to forward angle 0
    assert lidar[0] > 0.4
    # Bins behind should be 0.0
    assert lidar[8] == 0.0

