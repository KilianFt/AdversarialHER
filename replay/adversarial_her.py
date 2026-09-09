"""Dual-Mode Adversarial Hindsight Experience Replay (HER) Relabeler.

Relabels:
1. Accidental collisions during goal-seeking as intentional adversarial intercepts (Mode 1).
2. Crash-free navigation segments as successful goal reaches to achieved positions (Mode 0).
3. Symmetrically ingests data from all robots into a universal agent replay buffer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from envs.wrappers import build_ego_observation


@dataclass
class StepData:
    """Stores full step information needed for retrospective relabeling."""

    # Self agent state
    pos_self: np.ndarray        # (2,)
    rot_self: float
    vel_self: np.ndarray        # (2,)
    rot_vel_self: float

    # Next self agent state
    next_pos_self: np.ndarray   # (2,)
    next_rot_self: float
    next_vel_self: np.ndarray   # (2,)
    next_rot_vel_self: float

    # Other agent state
    pos_other: np.ndarray       # (2,)
    vel_other: np.ndarray       # (2,)
    next_pos_other: np.ndarray  # (2,)
    next_vel_other: np.ndarray  # (2,)

    # Targets & control
    goal_pos: np.ndarray        # (2,)
    nominal_mode: float         # 0.0 or 1.0
    action: np.ndarray          # (2,)

    # Level 1: Obstacles / Hazards
    hazards: Optional[np.ndarray] = None
    hazard_radius: float = 0.2

    # Outcomes
    collision: bool = False
    obstacle_collision: bool = False
    obstacle_pos: Optional[np.ndarray] = None
    goal_reached: bool = False
    obs: np.ndarray = field(default_factory=lambda: np.zeros(33, dtype=np.float32))
    next_obs: np.ndarray = field(default_factory=lambda: np.zeros(33, dtype=np.float32))
    reward: float = 0.0
    done: bool = False


class DualModeHERRelabeler:
    """Hindsight Experience Replay engine supporting Goal and Adversarial modes."""

    def __init__(
        self,
        goal_radius: float = 0.3,
        collision_threshold: float = 0.3,
        goal_reward: float = 5.0,
        collision_reward: float = 5.0,
        loss_penalty: float = 2.5,
        dense_reward_scale: float = 1.0,
        control_cost_weight: float = 0.001,
        her_ratio: float = 0.8,
        num_future_goals: int = 4,
        crash_window: int = 25,
        dwell_steps: int = 20,
        dwell_reward_rate: float = 0.05,
        enable_goal_her: bool = True,
        enable_crash_her: bool = True,
        relabel_obstacle_crashes: bool = True,
    ):
        self.goal_radius = goal_radius
        self.collision_threshold = collision_threshold
        self.goal_reward = goal_reward
        self.collision_reward = collision_reward
        self.loss_penalty = loss_penalty
        self.dense_reward_scale = dense_reward_scale
        self.control_cost_weight = control_cost_weight
        self.her_ratio = her_ratio
        self.num_future_goals = num_future_goals
        self.crash_window = crash_window
        self.dwell_steps = dwell_steps
        self.dwell_reward_rate = dwell_reward_rate

        self.enable_goal_her = enable_goal_her
        self.enable_crash_her = enable_crash_her
        self.relabel_obstacle_crashes = relabel_obstacle_crashes



    def relabel_trajectory(
        self, trajectory: List[StepData]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Relabel a single agent's trajectory using Dual-Mode HER.

        Returns (obs_batch, action_batch, reward_batch, next_obs_batch, done_batch).
        """
        T = len(trajectory)
        if T == 0:
            empty_obs = np.empty((0, 33), dtype=np.float32)
            empty_act = np.empty((0, 2), dtype=np.float32)
            empty_rew = np.empty((0,), dtype=np.float32)
            empty_done = np.empty((0,), dtype=bool)
            return empty_obs, empty_act, empty_rew, empty_obs, empty_done

        obs_list = []
        action_list = []
        reward_list = []
        next_obs_list = []
        done_list = []

        # 1. Add all original transitions
        for step in trajectory:
            obs_list.append(step.obs)
            action_list.append(step.action)
            reward_list.append(step.reward)
            next_obs_list.append(step.next_obs)
            done_list.append(step.done)

        # 2. Crash Relabeling (Mode 1: Adversarial HER)
        # Identify all steps where an inter-agent collision occurred
        crash_indices = [t for t, step in enumerate(trajectory) if step.collision]

        if self.enable_crash_her:
            for t_crash in crash_indices:
                start_t = max(0, t_crash - self.crash_window)
                # Relabel all transitions leading up to the crash as Adversarial Intercept (mode 1.0)
                for t in range(start_t, t_crash + 1):
                    step = trajectory[t]
                    # Recompute ego obs with mode=1.0 (target is other agent)
                    relabeled_obs = build_ego_observation(
                        pos_self=step.pos_self,
                        rot_self=step.rot_self,
                        vel_self=step.vel_self,
                        rot_vel_self=step.rot_vel_self,
                        pos_other=step.pos_other,
                        vel_other=step.vel_other,
                        goal_pos=step.goal_pos,
                        mode=1.0,
                        hazards=step.hazards,
                        hazard_radius=step.hazard_radius,
                    )
                    relabeled_next_obs = build_ego_observation(
                        pos_self=step.next_pos_self,
                        rot_self=step.next_rot_self,
                        vel_self=step.next_vel_self,
                        rot_vel_self=step.next_rot_vel_self,
                        pos_other=step.next_pos_other,
                        vel_other=step.next_vel_other,
                        goal_pos=step.goal_pos,
                        mode=1.0,
                        hazards=step.hazards,
                        hazard_radius=step.hazard_radius,
                    )

                    d_prev = np.linalg.norm(step.pos_self - step.pos_other)
                    d_curr = np.linalg.norm(step.next_pos_self - step.next_pos_other)
                    progress = (d_prev - d_curr) * self.dense_reward_scale
                    ctrl_cost = self.control_cost_weight * np.sum(np.square(step.action))

                    if t == t_crash:
                        # Terminal collision reward
                        r_relabeled = progress - ctrl_cost + self.collision_reward
                        d_relabeled = True
                    else:
                        r_relabeled = progress - ctrl_cost
                        d_relabeled = False

                    obs_list.append(relabeled_obs)
                    action_list.append(step.action)
                    reward_list.append(r_relabeled)
                    next_obs_list.append(relabeled_next_obs)
                    done_list.append(d_relabeled)

        # 3. Obstacle Collision Relabeling (Mode 1: Adversarial HER targeting obstacles)
        # Scalable foundation for Safety-Gym Level 1 obstacle avoidance
        if self.enable_crash_her and self.relabel_obstacle_crashes:
            obs_crash_indices = [
                t for t, step in enumerate(trajectory)
                if getattr(step, "obstacle_collision", False) and getattr(step, "obstacle_pos", None) is not None
            ]
            for t_crash in obs_crash_indices:
                target_obs_pos = trajectory[t_crash].obstacle_pos
                start_t = max(0, t_crash - self.crash_window)
                for t in range(start_t, t_crash + 1):
                    step = trajectory[t]
                    # Relabel with mode=1.0 and target = obstacle position (stationary obstacle: vel=0)
                    relabeled_obs = build_ego_observation(
                        pos_self=step.pos_self,
                        rot_self=step.rot_self,
                        vel_self=step.vel_self,
                        rot_vel_self=step.rot_vel_self,
                        pos_other=target_obs_pos,
                        vel_other=np.zeros(2, dtype=np.float32),
                        goal_pos=step.goal_pos,
                        mode=1.0,
                        hazards=step.hazards,
                        hazard_radius=step.hazard_radius,
                    )
                    relabeled_next_obs = build_ego_observation(
                        pos_self=step.next_pos_self,
                        rot_self=step.next_rot_self,
                        vel_self=step.next_vel_self,
                        rot_vel_self=step.next_rot_vel_self,
                        pos_other=target_obs_pos,
                        vel_other=np.zeros(2, dtype=np.float32),
                        goal_pos=step.goal_pos,
                        mode=1.0,
                        hazards=step.hazards,
                        hazard_radius=step.hazard_radius,
                    )
                    d_prev = np.linalg.norm(step.pos_self - target_obs_pos)
                    d_curr = np.linalg.norm(step.next_pos_self - target_obs_pos)
                    progress = (d_prev - d_curr) * self.dense_reward_scale
                    ctrl_cost = self.control_cost_weight * np.sum(np.square(step.action))

                    if t == t_crash:
                        r_relabeled = progress - ctrl_cost + self.collision_reward
                        d_relabeled = True
                    else:
                        r_relabeled = progress - ctrl_cost
                        d_relabeled = False

                    obs_list.append(relabeled_obs)
                    action_list.append(step.action)
                    reward_list.append(r_relabeled)
                    next_obs_list.append(relabeled_next_obs)
                    done_list.append(d_relabeled)

        # 4. Goal Relabeling (Mode 0: Crash-Free Goal HER with Dwell Synchronization)
        if self.enable_goal_her:
            crash_set = set(crash_indices)
            synthesized_dwell_targets = set()
            for t in range(T - 1):

                # Only perform HER relabeling with probability her_ratio
                if np.random.random() > self.her_ratio:
                    continue

                for _ in range(self.num_future_goals):
                    future_t = np.random.randint(t + 1, T)

                    # Check: did any collision occur between t and future_t?
                    # User constraint: "obviously in cases where it didn't crash"
                    has_intervening_crash = any((k in crash_set) for k in range(t, future_t + 1))
                    if has_intervening_crash:
                        continue  # Skip crashing segments for goal relabeling!

                    # Achieved goal position at future_t
                    achieved_goal = trajectory[future_t].next_pos_self.copy()
                    step = trajectory[t]

                    # Recompute ego obs with mode=0.0 and achieved goal
                    relabeled_obs = build_ego_observation(
                        pos_self=step.pos_self,
                        rot_self=step.rot_self,
                        vel_self=step.vel_self,
                        rot_vel_self=step.rot_vel_self,
                        pos_other=step.pos_other,
                        vel_other=step.vel_other,
                        goal_pos=achieved_goal,
                        mode=0.0,
                        hazards=step.hazards,
                        hazard_radius=step.hazard_radius,
                    )
                    relabeled_next_obs = build_ego_observation(
                        pos_self=step.next_pos_self,
                        rot_self=step.next_rot_self,
                        vel_self=step.next_vel_self,
                        rot_vel_self=step.next_rot_vel_self,
                        pos_other=step.next_pos_other,
                        vel_other=step.next_vel_other,
                        goal_pos=achieved_goal,
                        mode=0.0,
                        hazards=step.hazards,
                        hazard_radius=step.hazard_radius,
                    )

                    d_prev = np.linalg.norm(step.pos_self - achieved_goal)
                    d_curr = np.linalg.norm(step.next_pos_self - achieved_goal)
                    progress = (d_prev - d_curr) * self.dense_reward_scale
                    ctrl_cost = self.control_cost_weight * np.sum(np.square(step.action))

                    # Dwell holding reward when inside goal circle; passing through does NOT terminate
                    dwell_bonus = self.dwell_reward_rate if d_curr <= self.goal_radius else 0.0
                    r_relabeled = progress - ctrl_cost + dwell_bonus
                    d_relabeled = False

                    obs_list.append(relabeled_obs)
                    action_list.append(step.action)
                    reward_list.append(r_relabeled)
                    next_obs_list.append(relabeled_next_obs)
                    done_list.append(d_relabeled)

                    # Synthesize stationary dwell completion sequence for this achieved goal
                    goal_key = (round(float(achieved_goal[0]), 3), round(float(achieved_goal[1]), 3))
                    if goal_key not in synthesized_dwell_targets:
                        synthesized_dwell_targets.add(goal_key)
                        stop_step = trajectory[future_t]
                        dwell_obs = build_ego_observation(
                            pos_self=achieved_goal,
                            rot_self=stop_step.next_rot_self,
                            vel_self=np.zeros(2, dtype=np.float32),
                            rot_vel_self=0.0,
                            pos_other=stop_step.next_pos_other,
                            vel_other=stop_step.next_vel_other,
                            goal_pos=achieved_goal,
                            mode=0.0,
                            hazards=stop_step.hazards,
                            hazard_radius=stop_step.hazard_radius,
                        )
                        dwell_action = np.zeros(2, dtype=np.float32)

                        for k in range(self.dwell_steps):
                            is_final_dwell = (k == self.dwell_steps - 1)
                            r_dwell = self.goal_reward if is_final_dwell else self.dwell_reward_rate
                            d_dwell = is_final_dwell

                            obs_list.append(dwell_obs)
                            action_list.append(dwell_action)
                            reward_list.append(r_dwell)
                            next_obs_list.append(dwell_obs)
                            done_list.append(d_dwell)




        return (
            np.array(obs_list, dtype=np.float32),
            np.array(action_list, dtype=np.float32),
            np.array(reward_list, dtype=np.float32),
            np.array(next_obs_list, dtype=np.float32),
            np.array(done_list, dtype=bool),
        )
