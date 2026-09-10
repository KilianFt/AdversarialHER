"""Main training script for Dual-Mode Adversarial HER with RLPD."""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import yaml

from envs import MultiGoal0Env, DualModeEnvWrapper
from models import RLPDAgent
from replay import ReplayBuffer, DualModeHERRelabeler, StepData


def parse_args():
    parser = argparse.ArgumentParser(description="Train Adversarial HER with RLPD")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--exp_name", type=str, default="adv_her_rlpd", help="Experiment name")
    parser.add_argument("--total_steps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Override warmup steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--utd", type=int, default=None, help="Override update-to-data ratio")
    parser.add_argument("--eval_interval", type=int, default=None, help="Override eval interval")
    parser.add_argument("--eval_episodes", type=int, default=None, help="Override number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint directory")
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")

    # Baseline & ablation switches
    parser.add_argument(
        "--her_mode",
        type=str,
        default="full",
        choices=["full", "goal_only", "none"],
        help="HER mode: 'full' (Goal+Crash HER), 'goal_only' (standard HER), 'none' (vanilla RL)",
    )
    parser.add_argument(
        "--no_layernorm",
        action="store_true",
        help="Disable LayerNorm in critics (vanilla SAC baseline)",
    )

    parser.add_argument(
        "--level",
        type=int,
        default=0,
        choices=[0, 1],
        help="Environment level (0: MultiGoal0 without obstacles, 1: MultiGoal1 with hazards/obstacles)",
    )
    parser.add_argument(
        "--no_mode_balanced",
        action="store_true",
        help="Disable 50/50 mode-balanced sampling in replay buffer",
    )
    parser.add_argument(
        "--mode_sampling_ratio",
        type=float,
        default=None,
        help="Fraction of Mode 0 (Goal-Seeking) in training batches (e.g. 0.5, 0.7, 0.85)",
    )
    parser.add_argument(
        "--mode_distribution",
        type=str,
        default=None,
        choices=["mixed", "goal_only", "adversarial_only", "random"],
        help="Rollout mode distribution ('mixed', 'goal_only', 'adversarial_only')",
    )
    parser.add_argument(
        "--no_obstacle_relabel",
        action="store_true",
        help="Disable relabeling obstacle collisions into Mode 1 in Dual-Mode HER",
    )
    parser.add_argument(
        "--num_critics",
        type=int,
        default=None,
        help="Number of critics in RLPD ensemble (default from config: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda', 'cuda:0', 'mps', 'cpu', or 'auto')",
    )

    parser.add_argument(
        "--critic_type",
        type=str,
        default="standard",
        choices=["standard", "duality"],
        help="Critic architecture: 'standard' (monolithic Q-ensemble) or 'duality' (factorized Q-ensemble with inductive sign reversal)",
    )

    # Weights & Biases logging
    parser.add_argument("--wandb", action="store_true", help="Enable logging to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="adversarial-her", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity/username")
    return parser.parse_args()




def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate_policy(
    env: DualModeEnvWrapper,
    agent: RLPDAgent,
    num_episodes: int = 10,
    eval_mode: float = 0.0,
) -> Dict[str, float]:
    """Evaluate agent performance in a specific mode (0.0=Goal-seeking, 1.0=Adversarial)."""
    successes = []
    episode_rewards = []
    final_distances = []
    episode_lengths = []
    obstacle_hits = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        # Enforce evaluation mode for Agent 0
        if eval_mode == 0.0:
            env.modes = np.array([0.0, 0.0], dtype=np.float32)
        else:
            # Agent 0 is Adversary (1.0), Agent 1 is Target (0.0)
            env.modes = np.array([1.0, 0.0], dtype=np.float32)

        # Re-build initial observations with enforced modes
        obs = env._build_observations(info)

        done = False
        ep_ret = 0.0
        step_count = 0
        ep_obst_hit = False

        while not done:
            # Batched action selection for both agents in single forward pass
            acts = agent.select_actions(np.stack([obs[0], obs[1]]), deterministic=True)
            a0, a1 = acts[0], acts[1]

            obs, rews, terminated, truncated, info = env.step((a0, a1))

            hazard_coll = info.get("hazard_collision", [False, False])
            if hazard_coll[0]:
                ep_obst_hit = True

            ep_ret += float(rews[0])
            step_count += 1
            done = terminated or truncated

        episode_rewards.append(ep_ret)
        episode_lengths.append(step_count)
        obstacle_hits.append(float(ep_obst_hit))

        if eval_mode == 0.0:
            success = bool(info["goal_achieved"][0])
            dist = float(info["dist_goals"][0])
        else:
            success = bool(info["collision"])
            dist = float(info["dist_agents"])

        successes.append(float(success))
        final_distances.append(dist)

    return {
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(episode_rewards)),
        "mean_distance": float(np.mean(final_distances)),
        "mean_ep_length": float(np.mean(episode_lengths)),
        "obstacle_hit_rate": float(np.mean(obstacle_hits)),
    }


def main():
    cli_args = parse_args()
    cfg = load_config(cli_args.config)

    # Overrides from CLI
    if cli_args.total_steps is not None:
        cfg["train"]["total_timesteps"] = cli_args.total_steps
    if cli_args.warmup_steps is not None:
        cfg["train"]["warmup_steps"] = cli_args.warmup_steps
    if cli_args.batch_size is not None:

        cfg["rlpd"]["batch_size"] = cli_args.batch_size
    if cli_args.utd is not None:
        cfg["rlpd"]["utd_ratio"] = cli_args.utd
    if cli_args.eval_interval is not None:
        cfg["train"]["eval_interval"] = cli_args.eval_interval
    if cli_args.seed is not None:
        cfg["seed"] = cli_args.seed
    if cli_args.checkpoint_dir is not None:
        cfg["train"]["checkpoint_dir"] = cli_args.checkpoint_dir
    if cli_args.log_dir is not None:
        cfg["train"]["log_dir"] = cli_args.log_dir
    if cli_args.num_critics is not None:
        cfg["rlpd"]["num_critics"] = cli_args.num_critics
    if cli_args.mode_distribution is not None:

        cfg["rollout"]["mode_distribution"] = cli_args.mode_distribution
    elif cli_args.her_mode == "goal_only":
        cfg["rollout"]["mode_distribution"] = "goal_only"


    seed = cfg.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device selection
    dev_str = cli_args.device or cfg.get("device", "auto")
    if dev_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(dev_str)
    print(f"Using device: {device}")

    # Output directories
    os.makedirs(cfg["train"]["log_dir"], exist_ok=True)
    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
    metrics_file = os.path.join(cfg["train"]["log_dir"], f"{cli_args.exp_name}_metrics.jsonl")

    # Weights & Biases Initialization
    use_wandb = cli_args.wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=cli_args.wandb_project,
                entity=cli_args.wandb_entity,
                name=cli_args.exp_name,
                config={
                    **cfg,
                    "level": cli_args.level,
                    "her_mode": cli_args.her_mode,
                    "use_layernorm": not cli_args.no_layernorm,
                    "critic_type": cli_args.critic_type,
                },
            )
            print(f"Weights & Biases initialized: project='{cli_args.wandb_project}', run='{cli_args.exp_name}'")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb ({e}). Continuing with local logging.")
            use_wandb = False

    # 1. Initialize Environments
    print(f"Initializing Safety-Gym MultiGoal environment: Level {cli_args.level} (Obstacles: {cli_args.level >= 1})")
    base_train_env = MultiGoal0Env(
        arena_size=cfg["env"]["arena_size"],
        agent_radius=cfg["env"]["agent_radius"],
        goal_radius=cfg["env"]["goal_radius"],
        collision_threshold=cfg["env"]["collision_threshold"],
        max_episode_steps=cfg["env"]["max_episode_steps"],
        level=cli_args.level,
    )
    dwell_steps = cfg["env"].get("dwell_steps", 20)
    loss_penalty = cfg["env"].get("loss_penalty", 2.5)
    dwell_reward_rate = cfg["env"].get("dwell_reward_rate", 0.05)

    train_env = DualModeEnvWrapper(
        base_train_env,
        mode_distribution=cfg["rollout"]["mode_distribution"],
        mixed_weights=tuple(cfg["rollout"]["mixed_weights"]),
        dense_reward_scale=cfg["env"]["dense_reward_scale"],
        goal_reward=cfg["env"]["goal_reward"],
        collision_reward=cfg["env"]["collision_reward"],
        loss_penalty=loss_penalty,
        control_cost_weight=cfg["env"]["control_cost_weight"],
        dwell_steps=dwell_steps,
        dwell_reward_rate=dwell_reward_rate,
    )

    base_eval_env = MultiGoal0Env(
        arena_size=cfg["env"]["arena_size"],
        agent_radius=cfg["env"]["agent_radius"],
        goal_radius=cfg["env"]["goal_radius"],
        collision_threshold=cfg["env"]["collision_threshold"],
        max_episode_steps=cfg["env"]["max_episode_steps"],
        level=cli_args.level,
    )
    eval_env = DualModeEnvWrapper(
        base_eval_env,
        dwell_steps=dwell_steps,
        loss_penalty=loss_penalty,
    )


    # 2. Replay Buffer & HER Relabeler
    obs_dim = train_env.ego_obs_dim
    act_dim = train_env.single_action_space.shape[0]

    mode_balanced = not cli_args.no_mode_balanced if cli_args.no_mode_balanced else cfg["rlpd"].get("mode_balanced", True)
    mode_sampling_ratio = cli_args.mode_sampling_ratio if cli_args.mode_sampling_ratio is not None else cfg["rlpd"].get("mode_sampling_ratio", 0.5)

    buffer = ReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        capacity=cfg["rlpd"]["buffer_capacity"],
        device=device,
        mode_balanced=mode_balanced,
        mode_sampling_ratio=mode_sampling_ratio,
    )
    print(f"Replay Buffer: Mode-Balanced Sampling = {mode_balanced} (Target Mode 0 Ratio = {mode_sampling_ratio:.1%})")

    # Configure HER relabeling mode
    enable_goal_her = (cli_args.her_mode in ["full", "goal_only"])
    enable_crash_her = (cli_args.her_mode == "full")
    relabel_obstacle_crashes = (enable_crash_her and not cli_args.no_obstacle_relabel)
    print(
        f"HER Configuration: mode='{cli_args.her_mode}' "
        f"(Goal HER: {enable_goal_her}, Crash HER: {enable_crash_her}, Obstacle Relabel: {relabel_obstacle_crashes})"
    )

    her_relabeler = DualModeHERRelabeler(
        goal_radius=cfg["env"]["goal_radius"],
        collision_threshold=cfg["env"]["collision_threshold"],
        goal_reward=cfg["env"]["goal_reward"],
        collision_reward=cfg["env"]["collision_reward"],
        loss_penalty=loss_penalty,
        dense_reward_scale=cfg["env"]["dense_reward_scale"],
        control_cost_weight=cfg["env"]["control_cost_weight"],
        her_ratio=cfg["her"]["her_ratio"] if enable_goal_her else 0.0,
        num_future_goals=cfg["her"]["num_future_goals"],
        crash_window=cfg["her"]["crash_window"],
        dwell_steps=dwell_steps,
        dwell_reward_rate=dwell_reward_rate,
        enable_goal_her=enable_goal_her,
        enable_crash_her=enable_crash_her,
        relabel_obstacle_crashes=relabel_obstacle_crashes,
    )




    # 3. Agent (Shared Actor-Critic RLPD)
    use_layernorm = not cli_args.no_layernorm
    target_entropy = cfg["rlpd"].get("target_entropy", None)
    agent = RLPDAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dims=cfg["rlpd"]["hidden_dims"],
        num_critics=cfg["rlpd"]["num_critics"],
        actor_lr=cfg["rlpd"]["actor_lr"],
        critic_lr=cfg["rlpd"]["critic_lr"],
        alpha_lr=cfg["rlpd"]["alpha_lr"],
        init_temperature=cfg["rlpd"]["init_temperature"],
        gamma=cfg["rlpd"]["gamma"],
        tau=cfg["rlpd"]["tau"],
        target_entropy=target_entropy,
        auto_entropy_tuning=cfg["rlpd"]["auto_entropy_tuning"],
        use_layernorm=use_layernorm,
        critic_type=cli_args.critic_type,
        collision_reward=cfg["env"]["collision_reward"],
        loss_penalty=loss_penalty,
        device=device,
    )
    print(
        f"Agent Architecture: Critic Type = {cli_args.critic_type}, "
        f"LayerNorm Critics = {use_layernorm}, "
        f"Num Critics = {cfg['rlpd']['num_critics']}, "
        f"UTD = {cfg['rlpd']['utd_ratio']}, "
        f"Target Entropy = {agent.target_entropy:.2f}"
    )


    # 4. Training Loop

    total_timesteps = cfg["train"]["total_timesteps"]
    warmup_steps = cfg["train"]["warmup_steps"]
    batch_size = cfg["rlpd"]["batch_size"]
    utd_ratio = cfg["rlpd"]["utd_ratio"]
    eval_interval = cfg["train"]["eval_interval"]
    eval_episodes = cli_args.eval_episodes if cli_args.eval_episodes is not None else cfg["train"]["eval_episodes"]
    save_interval = cfg["train"]["save_interval"]


    obs, info = train_env.reset(seed=seed)
    total_env_steps = 0
    episodes_completed = 0
    best_eval_score = -float("inf")

    # Rolling diagnostic statistics across training rollouts
    recent_rew0 = deque(maxlen=50)
    recent_rew1 = deque(maxlen=50)
    recent_agent_colls = deque(maxlen=50)
    recent_obstacle_colls = deque(maxlen=50)
    recent_goal_reaches = deque(maxlen=50)

    ep_rew0 = 0.0
    ep_rew1 = 0.0
    ep_agent_collision = False
    ep_obstacle_collision = False
    ep_goal_reached = False

    print(f"Starting training for {total_timesteps} environment steps...")

    traj0: List[StepData] = []
    traj1: List[StepData] = []

    while total_env_steps < total_timesteps:
        # Step policy or sample random action during warmup
        if total_env_steps < warmup_steps:
            a0 = train_env.single_action_space.sample()
            a1 = train_env.single_action_space.sample()
        else:
            acts = agent.select_actions(np.stack([obs[0], obs[1]]), deterministic=False)
            a0, a1 = acts[0], acts[1]

        # Snapshot pre-step states
        s0_pre = train_env.env._get_agent_state(0)
        s1_pre = train_env.env._get_agent_state(1)
        g0_pre, g1_pre = info["goal_positions"]
        m0_pre, m1_pre = train_env.modes

        next_obs, rews, terminated, truncated, next_info = train_env.step((a0, a1))
        total_env_steps += 1

        # Track episode diagnostics
        ep_rew0 += float(rews[0])
        ep_rew1 += float(rews[1])
        agent_coll = bool(next_info.get("agent_collision", next_info.get("collision", False)))
        obst_coll = bool(np.any(next_info.get("obstacle_collision", [False, False])))
        goal_done = bool(np.any(next_info.get("goal_achieved", [False, False])))

        ep_agent_collision = ep_agent_collision or agent_coll
        ep_obstacle_collision = ep_obstacle_collision or obst_coll
        ep_goal_reached = ep_goal_reached or goal_done

        # Snapshot post-step states
        s0_post = train_env.env._get_agent_state(0)
        s1_post = train_env.env._get_agent_state(1)
        collision = bool(next_info["collision"])
        goal_reached = next_info["goal_reached"]
        hazard_coll = next_info.get("hazard_collision", [False, False])
        closest_h = next_info.get("closest_hazard", [None, None])
        hazards_pos = next_info.get("hazard_positions", None)
        hazard_r = float(getattr(train_env.env.unwrapped, "hazard_radius", getattr(train_env.env, "hazard_radius", 0.2)))

        step0 = StepData(
            pos_self=s0_pre["pos"], rot_self=s0_pre["rot"], vel_self=s0_pre["vel"], rot_vel_self=s0_pre["rot_vel"],
            next_pos_self=s0_post["pos"], next_rot_self=s0_post["rot"], next_vel_self=s0_post["vel"], next_rot_vel_self=s0_post["rot_vel"],
            pos_other=s1_pre["pos"], vel_other=s1_pre["vel"], next_pos_other=s1_post["pos"], next_vel_other=s1_post["vel"],
            goal_pos=g0_pre, nominal_mode=m0_pre, action=a0,
            hazards=hazards_pos, hazard_radius=hazard_r,
            collision=agent_coll,
            obstacle_collision=bool(hazard_coll[0]),
            obstacle_pos=closest_h[0],
            goal_reached=bool(goal_reached[0]),
            obs=obs[0], next_obs=next_obs[0], reward=float(rews[0]), done=(terminated or truncated),
        )
        step1 = StepData(
            pos_self=s1_pre["pos"], rot_self=s1_pre["rot"], vel_self=s1_pre["vel"], rot_vel_self=s1_pre["rot_vel"],
            next_pos_self=s1_post["pos"], next_rot_self=s1_post["rot"], next_vel_self=s1_post["vel"], next_rot_vel_self=s1_post["rot_vel"],
            pos_other=s0_pre["pos"], vel_other=s0_pre["vel"], next_pos_other=s0_post["pos"], next_vel_other=s0_post["vel"],
            goal_pos=g1_pre, nominal_mode=m1_pre, action=a1,
            hazards=hazards_pos, hazard_radius=hazard_r,
            collision=agent_coll,
            obstacle_collision=bool(hazard_coll[1]),
            obstacle_pos=closest_h[1],
            goal_reached=bool(goal_reached[1]),
            obs=obs[1], next_obs=next_obs[1], reward=float(rews[1]), done=(terminated or truncated),
        )
        traj0.append(step0)
        traj1.append(step1)


        obs = next_obs
        info = next_info
        done = terminated or truncated

        # If episode ends, relabel with Dual-Mode HER and add to buffer
        if done:
            episodes_completed += 1
            recent_rew0.append(ep_rew0)
            recent_rew1.append(ep_rew1)
            recent_agent_colls.append(float(ep_agent_collision))
            recent_obstacle_colls.append(float(ep_obstacle_collision))
            recent_goal_reaches.append(float(ep_goal_reached))

            ep_rew0 = 0.0
            ep_rew1 = 0.0
            ep_agent_collision = False
            ep_obstacle_collision = False
            ep_goal_reached = False

            # Relabel Agent 0 and Agent 1 trajectories
            r_obs0, r_act0, r_rew0, r_nobs0, r_done0, r_coll0 = her_relabeler.relabel_trajectory(traj0, return_collisions=True)
            r_obs1, r_act1, r_rew1, r_nobs1, r_done1, r_coll1 = her_relabeler.relabel_trajectory(traj1, return_collisions=True)

            # Symmetrically push into the single universal buffer
            buffer.add_batch(r_obs0, r_act0, r_rew0, r_nobs0, r_done0, r_coll0)
            buffer.add_batch(r_obs1, r_act1, r_rew1, r_nobs1, r_done1, r_coll1)

            traj0.clear()
            traj1.clear()
            obs, info = train_env.reset()

        # Update network via RLPD (high UTD)
        if total_env_steps >= warmup_steps and len(buffer) >= batch_size:
            update_metrics = agent.update_rlpd(buffer, utd_ratio=utd_ratio, batch_size=batch_size)
        else:
            update_metrics = {}

        # Periodic Evaluation
        if total_env_steps % eval_interval == 0 and total_env_steps >= warmup_steps:
            # 1. Eval Mode 0 (Goal-Seeking)
            eval_mode0 = evaluate_policy(eval_env, agent, num_episodes=eval_episodes, eval_mode=0.0)

            is_goal_only = (cfg["rollout"]["mode_distribution"] == "goal_only")
            if is_goal_only:
                eval_mode1 = {
                    "success_rate": 0.0,
                    "mean_return": 0.0,
                    "mean_distance": 0.0,
                    "mean_ep_length": 0.0,
                }
                combined_score = eval_mode0["success_rate"]
            else:
                # 2. Eval Mode 1 (Adversarial / Intercept)
                eval_mode1 = evaluate_policy(eval_env, agent, num_episodes=eval_episodes, eval_mode=1.0)
                combined_score = 0.5 * (eval_mode0["success_rate"] + eval_mode1["success_rate"])

            rollout_mean_rew = float(np.mean(list(recent_rew0) + list(recent_rew1))) if (recent_rew0 or recent_rew1) else 0.0
            rollout_agent_coll_rate = float(np.mean(recent_agent_colls)) if recent_agent_colls else 0.0
            rollout_obst_coll_rate = float(np.mean(recent_obstacle_colls)) if recent_obstacle_colls else 0.0
            rollout_goal_rate = float(np.mean(recent_goal_reaches)) if recent_goal_reaches else 0.0

            log_entry = {
                "step": total_env_steps,
                "episodes": episodes_completed,

                # Evaluation Metrics (Rewards, Success, Distances, Lengths, Collisions)
                "eval/combined_score": combined_score,
                "eval/mode0_success": eval_mode0["success_rate"],
                "eval/mode0_reward": eval_mode0["mean_return"],
                "eval/mode0_dist": eval_mode0["mean_distance"],
                "eval/mode0_length": eval_mode0["mean_ep_length"],
                "eval/mode0_obstacle_coll_rate": eval_mode0.get("obstacle_hit_rate", 0.0),
                "eval/mode1_success": eval_mode1["success_rate"],
                "eval/mode1_reward": eval_mode1["mean_return"],
                "eval/mode1_dist": eval_mode1["mean_distance"],
                "eval/mode1_length": eval_mode1["mean_ep_length"],
                "eval/mode1_obstacle_coll_rate": eval_mode1.get("obstacle_hit_rate", 0.0),

                # Rollout Metrics (Rewards and Collisions Separated)
                "rollout/mean_reward": rollout_mean_rew,
                "rollout/agent0_reward": float(np.mean(recent_rew0)) if recent_rew0 else 0.0,
                "rollout/agent1_reward": float(np.mean(recent_rew1)) if recent_rew1 else 0.0,
                "rollout/agent_collision_rate": rollout_agent_coll_rate,
                "rollout/obstacle_collision_rate": rollout_obst_coll_rate,
                "rollout/goal_reach_rate": rollout_goal_rate,

                # Buffer Diagnostics (Mode Balance)
                "buffer/total_size": len(buffer),
                "buffer/mode0_size": buffer.mode0_size,
                "buffer/mode1_size": buffer.mode1_size,
                "buffer/mode0_ratio": buffer.mode0_ratio,
                "buffer/mode1_ratio": buffer.mode1_ratio,

                # Backward compatibility for existing wandb dashboards
                "eval_mode0_success": eval_mode0["success_rate"],
                "eval_mode0_dist": eval_mode0["mean_distance"],
                "eval_mode0_reward": eval_mode0["mean_return"],
                "eval_mode1_success": eval_mode1["success_rate"],
                "eval_mode1_dist": eval_mode1["mean_distance"],
                "eval_mode1_reward": eval_mode1["mean_return"],
                "combined_score": combined_score,
                "buffer_size": len(buffer),

                **update_metrics,
            }

            with open(metrics_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            if use_wandb:
                wandb.log(log_entry, step=total_env_steps)

            if is_goal_only:
                print(
                    f"[Step {total_env_steps:6d}] "
                    f"Mode0 (Goal HER Baseline): {eval_mode0['success_rate']*100:5.1f}% (rew: {eval_mode0['mean_return']:+.2f}, dist: {eval_mode0['mean_distance']:.2f}m, obst_coll: {eval_mode0.get('obstacle_hit_rate', 0.0)*100:4.1f}%) | "
                    f"Alpha: {agent.alpha:.4f} | Buffer: {len(buffer)}"
                )
            else:
                print(
                    f"[Step {total_env_steps:6d}] "
                    f"Mode0: {eval_mode0['success_rate']*100:5.1f}% (rew: {eval_mode0['mean_return']:+.2f}, dist: {eval_mode0['mean_distance']:.2f}m, obst_coll: {eval_mode0.get('obstacle_hit_rate', 0.0)*100:4.1f}%) | "
                    f"Mode1: {eval_mode1['success_rate']*100:5.1f}% (rew: {eval_mode1['mean_return']:+.2f}, dist: {eval_mode1['mean_distance']:.2f}m) | "
                    f"Rollout Coll: (Agent: {rollout_agent_coll_rate*100:4.1f}%, Obst: {rollout_obst_coll_rate*100:4.1f}%) | "
                    f"Buf M0/M1: {buffer.mode0_ratio*100:2.0f}%/{buffer.mode1_ratio*100:2.0f}%"
                )

            # Save best checkpoint
            if combined_score > best_eval_score:
                best_eval_score = combined_score
                agent.save(os.path.join(cfg["train"]["checkpoint_dir"], f"{cli_args.exp_name}_best.pt"))


        # Periodic Save
        if total_env_steps % save_interval == 0 and total_env_steps >= warmup_steps:
            agent.save(os.path.join(cfg["train"]["checkpoint_dir"], f"{cli_args.exp_name}_step_{total_env_steps}.pt"))

    # Final Save
    agent.save(os.path.join(cfg["train"]["checkpoint_dir"], f"{cli_args.exp_name}_final.pt"))
    if use_wandb:
        wandb.finish()
    print(f"Training completed successfully. Checkpoints saved in {cfg['train']['checkpoint_dir']}.")



if __name__ == "__main__":
    main()
