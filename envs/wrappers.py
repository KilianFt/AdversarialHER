"""Egocentric observation and dual-mode conditioning wrappers for multi-agent RL."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np


def rotate_to_ego(vec: np.ndarray, rot: float) -> np.ndarray:
    """Rotate a 2D world vector into the agent's egocentric body frame.

    vec: [x, y] in global coordinates
    rot: heading angle in radians (0 = positive x-axis)
    """
    cos_theta = np.cos(rot)
    sin_theta = np.sin(rot)
    rot_mat = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]], dtype=np.float32)
    return rot_mat @ vec


def compute_hazards_lidar(
    pos_self: np.ndarray,
    rot_self: float,
    hazard_positions: Optional[np.ndarray],
    hazard_radius: float = 0.2,
    num_bins: int = 16,
    max_dist: float = 3.0,
    lidar_exp: float = 1.0,
) -> np.ndarray:
    """Compute official Safety-Gym robot-centric pseudo-lidar for hazards.

    Bins are arranged in a circle in the robot body frame covering 360 degrees.
    Values are exp(-dist_to_surface * lidar_exp), matching Safety-Gym default.
    """
    lidar = np.zeros(num_bins, dtype=np.float32)
    if hazard_positions is None or len(hazard_positions) == 0:
        return lidar

    bin_width = 2.0 * np.pi / num_bins
    for h_pos in hazard_positions:
        rel_world = h_pos - pos_self
        rel_ego = rotate_to_ego(rel_world, rot_self)
        dist = np.linalg.norm(rel_ego)
        dist_surf = max(0.0, dist - hazard_radius)
        if dist_surf >= max_dist:
            continue

        # In body frame: x is forward (angle=0), y is left (angle=pi/2)
        angle = np.arctan2(rel_ego[1], rel_ego[0]) % (2.0 * np.pi)
        bin_float = angle / bin_width
        bin_idx0 = int(bin_float) % num_bins
        bin_idx1 = (bin_idx0 + 1) % num_bins
        frac1 = bin_float - int(bin_float)
        frac0 = 1.0 - frac1

        val = np.exp(-dist_surf * lidar_exp)
        lidar[bin_idx0] = max(lidar[bin_idx0], val * frac0)
        lidar[bin_idx1] = max(lidar[bin_idx1], val * frac1)

    return lidar


def build_ego_observation(
    pos_self: np.ndarray,
    rot_self: float,
    vel_self: np.ndarray,
    rot_vel_self: float,
    pos_other: np.ndarray,
    vel_other: np.ndarray,
    goal_pos: np.ndarray,
    mode: float,
    hazards: Optional[np.ndarray] = None,
    hazard_radius: float = 0.2,
    num_lidar_bins: int = 16,
) -> np.ndarray:
    """Construct an egocentric, mode-conditioned observation vector with Safety-Gym hazards_lidar.

    Mode:
      0.0: Goal-Seeking (target is goal_pos)
      1.0: Adversarial / Intercept (target is pos_other)

    Returns a 33-dimensional vector (17 proprioceptive/navigation dims + 16 hazards_lidar bins).
    """
    # 1. Self proprioception in body frame
    vel_local = rotate_to_ego(vel_self, rot_self)
    compass = np.array([np.cos(rot_self), np.sin(rot_self)], dtype=np.float32)

    # 2. Goal in body frame
    rel_goal_world = goal_pos - pos_self
    rel_goal_ego = rotate_to_ego(rel_goal_world, rot_self)
    dist_goal = np.linalg.norm(rel_goal_world)

    # 3. Other agent in body frame
    rel_other_world = pos_other - pos_self
    rel_other_ego = rotate_to_ego(rel_other_world, rot_self)
    vel_other_ego = rotate_to_ego(vel_other - vel_self, rot_self)
    dist_other = np.linalg.norm(rel_other_world)

    # 4. Conditioned target vector
    if mode < 0.5:
        # Goal seeking mode
        target_ego = rel_goal_ego
        dist_target = dist_goal
    else:
        # Adversarial mode: target is the other agent
        target_ego = rel_other_ego
        dist_target = dist_other

    # 5. Official Safety-Gym hazards_lidar
    hazards_lidar = compute_hazards_lidar(
        pos_self=pos_self,
        rot_self=rot_self,
        hazard_positions=hazards,
        hazard_radius=hazard_radius,
        num_bins=num_lidar_bins,
    )

    obs = np.concatenate([
        vel_local,                    # (2,)
        np.array([rot_vel_self], dtype=np.float32),  # (1,)
        compass,                      # (2,)
        rel_goal_ego,                 # (2,)
        np.array([dist_goal], dtype=np.float32),     # (1,)
        rel_other_ego,                # (2,)
        vel_other_ego,                # (2,)
        np.array([dist_other], dtype=np.float32),    # (1,)
        np.array([mode], dtype=np.float32),          # (1,)
        target_ego,                   # (2,)
        np.array([dist_target], dtype=np.float32),   # (1,)
        hazards_lidar,                # (num_lidar_bins,) e.g. 16
    ]).astype(np.float32)

    return obs



class DualModeEnvWrapper(gym.Wrapper):
    """Wrapper that handles mode assignment, reward shaping, and egocentric observations.

    Assigns modes to each agent during rollouts and computes rewards:
    - Mode 0: Reward for reaching stationary goal + distance progress shaping.
    - Mode 1: Reward for colliding/intercepting the other robot + distance progress shaping.
    """

    def __init__(
        self,
        env: gym.Env,
        mode_distribution: str = "mixed",
        mixed_weights: Tuple[float, float, float] = (0.5, 0.25, 0.25),
        dense_reward_scale: float = 1.0,
        goal_reward: float = 5.0,
        collision_reward: float = 5.0,
        loss_penalty: float = 2.5,
        control_cost_weight: float = 0.001,
        dwell_steps: int = 20,
        dwell_reward_rate: float = 0.05,
    ):
        super().__init__(env)
        self.mode_distribution = mode_distribution
        self.mixed_weights = mixed_weights
        self.dense_reward_scale = dense_reward_scale
        self.goal_reward = goal_reward
        self.collision_reward = collision_reward
        self.loss_penalty = loss_penalty
        self.control_cost_weight = control_cost_weight
        self.dwell_steps = dwell_steps
        self.dwell_reward_rate = dwell_reward_rate

        self.single_action_space = self.env.single_action_space
        self.action_space = self.env.action_space
        # Egocentric obs dim is 33 (17 proprioception/navigation + 16 hazards_lidar bins)
        self.ego_obs_dim = 33

        self.single_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.ego_obs_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Tuple(
            [self.single_observation_space for _ in range(self.env.num_agents)]
        )

        self.modes = np.zeros(2, dtype=np.float32)
        self.prev_distances = np.zeros(2, dtype=np.float32)
        self.dwell_counters = np.zeros(2, dtype=int)
        self.goal_achieved = np.zeros(2, dtype=bool)

    @property
    def goals(self) -> np.ndarray:
        return self.env.goals



    def sample_modes(self) -> np.ndarray:
        """Sample operating modes for Agent 0 and Agent 1."""
        if self.mode_distribution == "goal_only":
            return np.array([0.0, 0.0], dtype=np.float32)
        elif self.mode_distribution == "adversarial_only":
            return np.array([1.0, 1.0], dtype=np.float32)
        elif self.mode_distribution == "random":
            m0 = float(np.random.choice([0.0, 1.0]))
            m1 = float(np.random.choice([0.0, 1.0]))
            return np.array([m0, m1], dtype=np.float32)
        else:
            # Mixed distribution:
            # 0: Both goal-seeking [0, 0]
            # 1: One adv, one goal [1, 0] or [0, 1]
            # 2: Both adversarial [1, 1]
            idx = np.random.choice(3, p=self.mixed_weights)
            if idx == 0:
                return np.array([0.0, 0.0], dtype=np.float32)
            elif idx == 1:
                if np.random.random() < 0.5:
                    return np.array([1.0, 0.0], dtype=np.float32)
                else:
                    return np.array([0.0, 1.0], dtype=np.float32)
            else:
                return np.array([1.0, 1.0], dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
        raw_obs, info = self.env.reset(seed=seed, options=options)
        self.modes = self.sample_modes()
        self.dwell_counters = np.zeros(2, dtype=int)
        self.goal_achieved = np.zeros(2, dtype=bool)

        # Compute initial target distances for progress shaping
        p0, p1 = info["agent_positions"]
        g0, g1 = info["goal_positions"]

        d0 = np.linalg.norm(p0 - g0) if self.modes[0] == 0.0 else np.linalg.norm(p0 - p1)
        d1 = np.linalg.norm(p1 - g1) if self.modes[1] == 0.0 else np.linalg.norm(p1 - p0)
        self.prev_distances = np.array([d0, d1], dtype=np.float32)

        ego_obs = self._build_observations(info)
        info["modes"] = self.modes.copy()
        info["dwell_counters"] = self.dwell_counters.copy()
        info["goal_achieved"] = self.goal_achieved.copy()
        return ego_obs, info

    def _build_observations(self, info: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        p0, p1 = info["agent_positions"]
        r0, r1 = info["agent_rotations"]
        g0, g1 = info["goal_positions"]

        s0 = self.env._get_agent_state(0)
        s1 = self.env._get_agent_state(1)

        hazards = info.get("hazard_positions", None)
        hazard_radius = getattr(self.env.unwrapped, "hazard_radius", getattr(self.env, "hazard_radius", 0.2))

        obs0 = build_ego_observation(
            pos_self=s0["pos"],
            rot_self=s0["rot"],
            vel_self=s0["vel"],
            rot_vel_self=s0["rot_vel"],
            pos_other=s1["pos"],
            vel_other=s1["vel"],
            goal_pos=g0,
            mode=self.modes[0],
            hazards=hazards,
            hazard_radius=hazard_radius,
        )

        obs1 = build_ego_observation(
            pos_self=s1["pos"],
            rot_self=s1["rot"],
            vel_self=s1["vel"],
            rot_vel_self=s1["rot_vel"],
            pos_other=s0["pos"],
            vel_other=s0["vel"],
            goal_pos=g1,
            mode=self.modes[1],
            hazards=hazards,
            hazard_radius=hazard_radius,
        )

        return obs0, obs1

    def step(
        self, actions: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, bool, bool, Dict[str, Any]]:
        raw_obs, _, terminated, truncated, info = self.env.step(actions)

        p0, p1 = info["agent_positions"]
        g0, g1 = info["goal_positions"]
        collision = bool(info["collision"])
        goal_in_range = info["goal_reached"]  # dist <= goal_radius

        # Current target distances
        curr_d0 = np.linalg.norm(p0 - g0) if self.modes[0] == 0.0 else np.linalg.norm(p0 - p1)
        curr_d1 = np.linalg.norm(p1 - g1) if self.modes[1] == 0.0 else np.linalg.norm(p1 - p0)

        # 1. Update Dwell Counters for Goal-Seeking agents
        dwell_completed_now = np.zeros(2, dtype=bool)
        for i in range(2):
            if self.modes[i] == 0.0:
                if goal_in_range[i]:
                    self.dwell_counters[i] += 1
                    if self.dwell_counters[i] >= self.dwell_steps and not self.goal_achieved[i]:
                        self.goal_achieved[i] = True
                        dwell_completed_now[i] = True
                else:
                    # Slipped or pushed out of goal circle before completing dwell!
                    if not self.goal_achieved[i]:
                        self.dwell_counters[i] = 0

        # 2. Rewards & Shaping
        rewards = np.zeros(2, dtype=np.float32)
        for i, (mode, curr_d, prev_d, a) in enumerate(
            zip(self.modes, [curr_d0, curr_d1], self.prev_distances, actions)
        ):
            if not self.goal_achieved[i]:
                progress = (prev_d - curr_d) * self.dense_reward_scale
                rewards[i] += progress
            ctrl_cost = self.control_cost_weight * np.sum(np.square(a))
            rewards[i] -= ctrl_cost

            # Level 1: Obstacle / hazard collision penalty for goal-seeking agents
            hazard_collision = info.get("hazard_collision", [False, False])
            if mode == 0.0 and hazard_collision[i]:
                rewards[i] -= self.loss_penalty

        winner = None


        # Case 1: Both robots are Goal-Seeking ([0.0, 0.0])
        if self.modes[0] == 0.0 and self.modes[1] == 0.0:
            for i in range(2):
                if dwell_completed_now[i]:
                    rewards[i] += self.goal_reward
                elif self.goal_achieved[i]:
                    rewards[i] += self.dwell_reward_rate
            # Terminate only when BOTH had the chance and succeeded!
            if self.goal_achieved[0] and self.goal_achieved[1]:
                terminated = True
                winner = "both"

        # Case 2: Asymmetric matchup - One Adversarial, One Goal-Seeking
        elif (self.modes[0] == 1.0 and self.modes[1] == 0.0) or (self.modes[0] == 0.0 and self.modes[1] == 1.0):
            adv_idx = 0 if self.modes[0] == 1.0 else 1
            goal_idx = 1 if self.modes[0] == 1.0 else 0

            if collision:
                # Adversarial wins!
                rewards[adv_idx] += self.collision_reward
                rewards[goal_idx] -= self.loss_penalty
                terminated = True
                winner = f"agent_{adv_idx}_adversarial"
            elif dwell_completed_now[goal_idx]:
                # Goal-seeker wins!
                rewards[goal_idx] += self.goal_reward
                rewards[adv_idx] -= self.loss_penalty
                terminated = True
                winner = f"agent_{goal_idx}_goal"

        # Case 3: Both robots are Adversarial ([1.0, 1.0])
        else:
            if collision:
                rewards[0] += self.collision_reward
                rewards[1] += self.collision_reward
                terminated = True
                winner = "both"

        self.prev_distances = np.array([curr_d0, curr_d1], dtype=np.float32)

        ego_obs = self._build_observations(info)
        info["modes"] = self.modes.copy()
        info["dwell_counters"] = self.dwell_counters.copy()
        info["goal_achieved"] = self.goal_achieved.copy()
        info["winner"] = winner
        info["rewards_breakdown"] = rewards.copy()

        return ego_obs, rewards, terminated, truncated, info


