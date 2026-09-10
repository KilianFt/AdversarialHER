"""Unit tests for Dual-Mode Adversarial Hindsight Experience Replay (HER)."""

import numpy as np
import pytest
from replay import DualModeHERRelabeler, StepData


def test_adversarial_crash_relabeling():
    """Verify that an accidental crash during goal-seeking is relabeled as an adversarial hit."""
    relabeler = DualModeHERRelabeler(
        collision_reward=5.0,
        crash_window=5,
        her_ratio=0.0,  # Turn off goal HER to isolate crash relabeling
    )

    trajectory = []
    # 6 steps, with crash occurring at step 5
    for t in range(6):
        pos_self = np.array([t * 0.2, 0.0], dtype=np.float32)
        next_pos_self = np.array([(t + 1) * 0.2, 0.0], dtype=np.float32)
        pos_other = np.array([1.2, 0.0], dtype=np.float32)
        next_pos_other = np.array([1.2, 0.0], dtype=np.float32)
        collision = (t == 5)

        step = StepData(
            pos_self=pos_self, rot_self=0.0, vel_self=np.zeros(2, dtype=np.float32), rot_vel_self=0.0,
            next_pos_self=next_pos_self, next_rot_self=0.0, next_vel_self=np.zeros(2, dtype=np.float32), next_rot_vel_self=0.0,
            pos_other=pos_other, vel_other=np.zeros(2, dtype=np.float32), next_pos_other=next_pos_other, next_vel_other=np.zeros(2, dtype=np.float32),
            goal_pos=np.array([5.0, 5.0], dtype=np.float32), nominal_mode=0.0, action=np.zeros(2, dtype=np.float32),
            collision=collision, goal_reached=False,
            obs=np.zeros(33, dtype=np.float32), next_obs=np.zeros(33, dtype=np.float32), reward=0.0, done=False,
        )
        trajectory.append(step)

    obs, act, rew, next_obs, done = relabeler.relabel_trajectory(trajectory)

    # Original 6 + 6 crash-relabeled = 12 transitions
    assert len(obs) == 12

    # Check that the terminal crash transition has done=True and collision_reward >= 5.0
    # In relabeled portion (last 6 items):
    relabeled_rews = rew[6:]
    relabeled_dones = done[6:]
    relabeled_obs = obs[6:]

    assert relabeled_dones[-1] == True
    assert relabeled_rews[-1] >= 5.0
    # Check that mode indicator at index 13 is 1.0 (Adversarial mode)
    assert relabeled_obs[-1][13] == 1.0


def test_crash_free_goal_relabeling():
    """Verify that crash-free segments are relabeled as achieved goals, but crashing segments are skipped."""
    relabeler = DualModeHERRelabeler(
        goal_radius=0.3,
        goal_reward=5.0,
        her_ratio=1.0,  # Always relabel
        num_future_goals=2,
    )

    trajectory = []
    for t in range(5):
        pos_self = np.array([t * 0.5, 0.0], dtype=np.float32)
        next_pos_self = np.array([(t + 1) * 0.5, 0.0], dtype=np.float32)
        step = StepData(
            pos_self=pos_self, rot_self=0.0, vel_self=np.zeros(2, dtype=np.float32), rot_vel_self=0.0,
            next_pos_self=next_pos_self, next_rot_self=0.0, next_vel_self=np.zeros(2, dtype=np.float32), next_rot_vel_self=0.0,
            pos_other=np.array([10.0, 10.0], dtype=np.float32), vel_other=np.zeros(2, dtype=np.float32),
            next_pos_other=np.array([10.0, 10.0], dtype=np.float32), next_vel_other=np.zeros(2, dtype=np.float32),
            goal_pos=np.array([0.0, 5.0], dtype=np.float32), nominal_mode=0.0, action=np.zeros(2, dtype=np.float32),
            collision=False, goal_reached=False,
            obs=np.zeros(33, dtype=np.float32), next_obs=np.zeros(33, dtype=np.float32), reward=0.0, done=False,
        )
        trajectory.append(step)

    obs, act, rew, next_obs, done = relabeler.relabel_trajectory(trajectory)
    # Original 5 + HER transitions
    assert len(obs) > 5
    # Mode indicator in goal HER should be 0.0
    for o in obs[5:]:
        assert o[13] == 0.0

    # Verify synthetic dwell completion transitions exist with done=True and reward >= 5.0
    assert np.any(done[5:] == True)
    assert np.any(rew[5:] >= 5.0)



def test_obstacle_crash_relabeling():
    """Verify that colliding with a stationary obstacle (Level 1) is relabeled into Mode 1 targeting that obstacle."""
    relabeler = DualModeHERRelabeler(
        collision_reward=5.0,
        crash_window=3,
        her_ratio=0.0,  # Isolate crash relabeling
        relabel_obstacle_crashes=True,
    )

    obstacle_pos = np.array([1.5, 1.5], dtype=np.float32)
    trajectory = []

    for t in range(4):
        pos_self = np.array([t * 0.4, t * 0.4], dtype=np.float32)
        next_pos_self = np.array([(t + 1) * 0.4, (t + 1) * 0.4], dtype=np.float32)
        is_obs_crash = (t == 3)

        step = StepData(
            pos_self=pos_self, rot_self=0.0, vel_self=np.zeros(2, dtype=np.float32), rot_vel_self=0.0,
            next_pos_self=next_pos_self, next_rot_self=0.0, next_vel_self=np.zeros(2, dtype=np.float32), next_rot_vel_self=0.0,
            pos_other=np.array([5.0, 5.0], dtype=np.float32), vel_other=np.zeros(2, dtype=np.float32),
            next_pos_other=np.array([5.0, 5.0], dtype=np.float32), next_vel_other=np.zeros(2, dtype=np.float32),
            goal_pos=np.array([0.0, 5.0], dtype=np.float32), nominal_mode=0.0, action=np.zeros(2, dtype=np.float32),
            collision=False,
            obstacle_collision=is_obs_crash,
            obstacle_pos=obstacle_pos if is_obs_crash else None,
            goal_reached=False,
            obs=np.zeros(33, dtype=np.float32), next_obs=np.zeros(33, dtype=np.float32), reward=0.0, done=False,
        )
        trajectory.append(step)

    obs, act, rew, next_obs, done = relabeler.relabel_trajectory(trajectory)

    # 4 original + 4 obstacle-crash relabeled = 8 transitions
    assert len(obs) == 8

    relabeled_rews = rew[4:]
    relabeled_dones = done[4:]
    relabeled_obs = obs[4:]

    # Last step should be done=True with collision reward >= 5.0
    assert relabeled_dones[-1] == True
    assert relabeled_rews[-1] >= 5.0
    # Mode should be 1.0 (Adversarial intercept targeting obstacle)
    assert relabeled_obs[-1][13] == 1.0


def test_her_collision_flag_recording():
    """Verify that return_collisions=True outputs matching 6-tuple with exact collision indicators."""
    relabeler = DualModeHERRelabeler(enable_goal_her=False, enable_crash_her=True)
    trajectory = []
    for t in range(5):
        is_crash = (t == 4)
        step = StepData(
            pos_self=np.array([float(t), 0.0]), rot_self=0.0, vel_self=np.zeros(2), rot_vel_self=0.0,
            next_pos_self=np.array([float(t) + 0.1, 0.0]), next_rot_self=0.0, next_vel_self=np.zeros(2), next_rot_vel_self=0.0,
            pos_other=np.array([5.0, 0.0]), vel_other=np.zeros(2), next_pos_other=np.array([5.0, 0.0]), next_vel_other=np.zeros(2),
            goal_pos=np.array([10.0, 0.0]), nominal_mode=0.0, action=np.zeros(2),
            collision=is_crash, goal_reached=False,
            obs=np.zeros(33, dtype=np.float32), next_obs=np.zeros(33, dtype=np.float32), reward=0.0, done=is_crash,
        )
        trajectory.append(step)

    obs, act, rew, next_obs, done, coll = relabeler.relabel_trajectory(trajectory, return_collisions=True)
    assert len(obs) == len(coll)
    assert coll.shape == (len(obs),)
    # 5 original + 5 crash-relabeled = 10 transitions
    assert len(coll) == 10
    # Collision flag should be 1.0 at step 4 (original crash) and step 9 (relabeled crash)
    assert coll[4] == 1.0
    assert coll[9] == 1.0
    # Other steps should be 0.0
    assert np.all(coll[:4] == 0.0)
    assert np.all(coll[5:9] == 0.0)



