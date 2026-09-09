"""MultiGoal0 environment implementation for multi-agent safe navigation.

Provides a unified interface with 2 Point robots and 2 goals, supporting
both built-in MuJoCo physics simulation and native Safety-Gymnasium.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from envs.point_robot import generate_multigoal_xml


class MultiGoal0Env(gym.Env):
    """MultiGoal0 environment with 2 Point robots and 2 Goals.

    Agent 0: Red agent
    Agent 1: Blue agent
    Goal 0: Target for Agent 0
    Goal 1: Target for Agent 1
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        arena_size: float = 2.5,
        agent_radius: float = 0.15,
        goal_radius: float = 0.3,
        collision_threshold: float = 0.3,
        max_episode_steps: int = 200,
        substeps: int = 5,
        level: int = 0,
        num_hazards: int = 4,
        hazard_radius: float = 0.2,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.arena_size = arena_size
        self.agent_radius = agent_radius
        self.goal_radius = goal_radius
        self.collision_threshold = collision_threshold
        self.max_episode_steps = max_episode_steps
        self.substeps = substeps
        self.level = level
        self.num_hazards = num_hazards if level >= 1 else 0
        self.hazard_radius = hazard_radius
        self.render_mode = render_mode

        self.num_agents = 2
        # Action space per agent: [forward_drive, steer_turn]
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Tuple(
            [self.single_action_space for _ in range(self.num_agents)]
        )

        # Raw observation dictionary per agent
        self.single_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        self.observation_space = spaces.Tuple(
            [self.single_observation_space for _ in range(self.num_agents)]
        )

        # Compile MuJoCo model
        xml_str = generate_multigoal_xml(
            arena_size=self.arena_size,
            agent_radius=self.agent_radius,
            goal_radius=self.goal_radius,
            level=self.level,
            num_hazards=self.num_hazards,
            hazard_radius=self.hazard_radius,
        )
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)

        # MuJoCo IDs
        self.agent0_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "agent0_x")
        self.agent0_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "agent0_y")
        self.agent0_rot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "agent0_rot")

        self.agent1_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "agent1_x")
        self.agent1_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "agent1_y")
        self.agent1_rot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "agent1_rot")

        self.goal0_mocap_id = self.model.body("goal0").mocapid[0]
        self.goal1_mocap_id = self.model.body("goal1").mocapid[0]

        if self.level >= 1 and self.num_hazards > 0:
            self.hazard_mocap_ids = [
                self.model.body(f"hazard{k}").mocapid[0] for k in range(self.num_hazards)
            ]
            self.hazards = np.zeros((self.num_hazards, 2), dtype=np.float32)
        else:
            self.hazard_mocap_ids = []
            self.hazards = np.empty((0, 2), dtype=np.float32)

        # Actuator IDs
        self.act_agent0_drive_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "agent0_drive_x")
        self.act_agent0_drive_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "agent0_drive_y")
        self.act_agent0_steer = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "agent0_steer")

        self.act_agent1_drive_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "agent1_drive_x")
        self.act_agent1_drive_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "agent1_drive_y")
        self.act_agent1_steer = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "agent1_steer")

        self.renderer = None
        self.step_count = 0
        self.goals = np.zeros((2, 2), dtype=np.float32)

    def _sample_valid_positions(self, min_dist: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample agent, goal, and hazard positions inside arena ensuring separation."""
        limit = self.arena_size * 0.75
        total_items = 4 + (self.num_hazards if self.level >= 1 else 0)
        positions = []
        for _ in range(total_items):
            for _ in range(100):
                pos = np.random.uniform(-limit, limit, size=2)
                if all(np.linalg.norm(pos - p) >= min_dist for p in positions):
                    positions.append(pos)
                    break
            else:
                positions.append(np.random.uniform(-limit, limit, size=2))

        agent_pos = np.array(positions[:2], dtype=np.float32)
        goal_pos = np.array(positions[2:4], dtype=np.float32)
        hazard_pos = np.array(positions[4:], dtype=np.float32) if self.level >= 1 else np.empty((0, 2), dtype=np.float32)
        return agent_pos, goal_pos, hazard_pos


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        mujoco.mj_resetData(self.model, self.data)

        # Sample initial poses and goals
        agent_positions, goal_positions, hazard_positions = self._sample_valid_positions()
        self.goals = goal_positions.copy()

        # Set Agent 0 state
        self.data.qpos[self.agent0_x_id] = agent_positions[0, 0]
        self.data.qpos[self.agent0_y_id] = agent_positions[0, 1]
        self.data.qpos[self.agent0_rot_id] = np.random.uniform(-np.pi, np.pi)

        # Set Agent 1 state
        self.data.qpos[self.agent1_x_id] = agent_positions[1, 0]
        self.data.qpos[self.agent1_y_id] = agent_positions[1, 1]
        self.data.qpos[self.agent1_rot_id] = np.random.uniform(-np.pi, np.pi)

        # Set Goals (mocap bodies)
        self.data.mocap_pos[self.goal0_mocap_id][:2] = self.goals[0]
        self.data.mocap_pos[self.goal1_mocap_id][:2] = self.goals[1]

        # Set Hazards (Level 1)
        if self.level >= 1 and self.num_hazards > 0:
            self.hazards = hazard_positions.copy()
            for k in range(self.num_hazards):
                self.data.mocap_pos[self.hazard_mocap_ids[k]][:2] = self.hazards[k]

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info


    def _get_agent_state(self, agent_idx: int) -> Dict[str, np.ndarray]:
        """Extract physical state for a specific agent."""
        if agent_idx == 0:
            x = self.data.qpos[self.agent0_x_id]
            y = self.data.qpos[self.agent0_y_id]
            rot = self.data.qpos[self.agent0_rot_id]
            vx = self.data.qvel[self.agent0_x_id]
            vy = self.data.qvel[self.agent0_y_id]
            vrot = self.data.qvel[self.agent0_rot_id]
        else:
            x = self.data.qpos[self.agent1_x_id]
            y = self.data.qpos[self.agent1_y_id]
            rot = self.data.qpos[self.agent1_rot_id]
            vx = self.data.qvel[self.agent1_x_id]
            vy = self.data.qvel[self.agent1_y_id]
            vrot = self.data.qvel[self.agent1_rot_id]

        pos = np.array([x, y], dtype=np.float32)
        vel = np.array([vx, vy], dtype=np.float32)
        compass = np.array([np.cos(rot), np.sin(rot)], dtype=np.float32)
        return {
            "pos": pos,
            "vel": vel,
            "rot": float(rot),
            "rot_vel": float(vrot),
            "compass": compass,
        }

    def _get_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build raw observations for Agent 0 and Agent 1."""
        s0 = self._get_agent_state(0)
        s1 = self._get_agent_state(1)

        # Agent 0 vector
        obs0 = np.concatenate([
            s0["pos"],
            s0["vel"],
            [s0["rot"], s0["rot_vel"]],
            s0["compass"],
            self.goals[0],
            s1["pos"],
            s1["vel"],
        ]).astype(np.float32)

        # Agent 1 vector
        obs1 = np.concatenate([
            s1["pos"],
            s1["vel"],
            [s1["rot"], s1["rot_vel"]],
            s1["compass"],
            self.goals[1],
            s0["pos"],
            s0["vel"],
        ]).astype(np.float32)

        return obs0, obs1

    def _get_info(self) -> Dict[str, Any]:
        """Compute status information including distances and collision checks."""
        s0 = self._get_agent_state(0)
        s1 = self._get_agent_state(1)

        dist_agents = float(np.linalg.norm(s0["pos"] - s1["pos"]))
        dist_goal0 = float(np.linalg.norm(s0["pos"] - self.goals[0]))
        dist_goal1 = float(np.linalg.norm(s1["pos"] - self.goals[1]))

        collision = dist_agents <= self.collision_threshold
        goal0_reached = dist_goal0 <= self.goal_radius
        goal1_reached = dist_goal1 <= self.goal_radius

        # Hazard / obstacle collision checking (Level 1)
        if self.level >= 1 and len(self.hazards) > 0:
            dists_h0 = np.linalg.norm(self.hazards - s0["pos"], axis=1)
            dists_h1 = np.linalg.norm(self.hazards - s1["pos"], axis=1)
            coll_h0 = bool(np.any(dists_h0 <= (self.agent_radius + self.hazard_radius)))
            coll_h1 = bool(np.any(dists_h1 <= (self.agent_radius + self.hazard_radius)))
            closest_h0 = self.hazards[np.argmin(dists_h0)]
            closest_h1 = self.hazards[np.argmin(dists_h1)]
        else:
            coll_h0, coll_h1 = False, False
            closest_h0, closest_h1 = None, None

        agent_collision = collision
        obstacle_collision = np.array([coll_h0, coll_h1], dtype=bool)

        return {
            "agent_positions": np.array([s0["pos"], s1["pos"]], dtype=np.float32),
            "agent_rotations": np.array([s0["rot"], s1["rot"]], dtype=np.float32),
            "goal_positions": self.goals.copy(),
            "hazard_positions": self.hazards.copy() if hasattr(self, "hazards") else np.empty((0, 2), dtype=np.float32),
            "agent_collision": agent_collision,
            "obstacle_collision": obstacle_collision,
            "hazard_collision": obstacle_collision,
            "closest_hazard": [closest_h0, closest_h1],
            "dist_agents": dist_agents,
            "dist_goals": np.array([dist_goal0, dist_goal1], dtype=np.float32),
            "collision": agent_collision,
            "goal_reached": np.array([goal0_reached, goal1_reached], dtype=bool),
            "step_count": self.step_count,
        }



    def step(
        self, actions: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray, bool, bool, Dict[str, Any]]:
        """Advance simulation by applying body-relative drive & steer actions."""
        self.step_count += 1

        a0 = np.clip(actions[0], -1.0, 1.0)
        a1 = np.clip(actions[1], -1.0, 1.0)

        s0 = self._get_agent_state(0)
        s1 = self._get_agent_state(1)

        # Transform body-relative drive to global forces
        # a[0]: forward drive force, a[1]: steer torque
        drive0_fx = a0[0] * np.cos(s0["rot"])
        drive0_fy = a0[0] * np.sin(s0["rot"])
        steer0_torque = a0[1]

        drive1_fx = a1[0] * np.cos(s1["rot"])
        drive1_fy = a1[0] * np.sin(s1["rot"])
        steer1_torque = a1[1]

        self.data.ctrl[self.act_agent0_drive_x] = drive0_fx
        self.data.ctrl[self.act_agent0_drive_y] = drive0_fy
        self.data.ctrl[self.act_agent0_steer] = steer0_torque

        self.data.ctrl[self.act_agent1_drive_x] = drive1_fx
        self.data.ctrl[self.act_agent1_drive_y] = drive1_fy
        self.data.ctrl[self.act_agent1_steer] = steer1_torque

        # Step physics
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()

        # Check termination and truncation
        truncated = self.step_count >= self.max_episode_steps
        terminated = False  # By default episodes continue or end on truncation

        # Base nominal rewards (progress to goals)
        # Note: Handled in detail by the DualModeHER wrapper
        rewards = np.zeros(2, dtype=np.float32)

        return obs, rewards, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render overhead view of the arena."""
        if self.render_mode != "rgb_array":
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        self.renderer.update_scene(self.data, camera="overhead")
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            del self.renderer
            self.renderer = None
