"""Evaluation and trajectory visualization script for Dual-Mode Adversarial Mobile Robot."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from envs import MultiGoal0Env, DualModeEnvWrapper
from models import RLPDAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Dual-Mode Adversarial Policy")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes per mode")
    parser.add_argument("--plot", action="store_true", help="Plot and save trajectory visualizations")
    parser.add_argument("--show", action="store_true", help="Display plot window interactively")
    parser.add_argument("--num_plots", type=int, default=3, help="Number of trajectory plots to save per mode")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory to save evaluation artifacts")
    parser.add_argument("--level", type=int, default=0, choices=[0, 1], help="Environment level (0: MultiGoal0, 1: MultiGoal1 with hazards)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'mps', 'cpu', or 'auto')")
    return parser.parse_args()




def run_episode(
    env: DualModeEnvWrapper,
    agent: RLPDAgent,
    mode: float = 0.0,
) -> Dict[str, Any]:
    """Run a single evaluation episode and record positions."""
    obs, info = env.reset()
    if mode == 0.0:
        env.modes = np.array([0.0, 0.0], dtype=np.float32)
    else:
        env.modes = np.array([1.0, 0.0], dtype=np.float32)

    obs = env._build_observations(info)

    pos_agent0 = [info["agent_positions"][0].copy()]
    pos_agent1 = [info["agent_positions"][1].copy()]
    goals = info["goal_positions"].copy()

    done = False
    step_count = 0
    total_rew = 0.0

    while not done:
        acts = agent.select_actions(np.stack([obs[0], obs[1]]), deterministic=True)
        a0, a1 = acts[0], acts[1]

        obs, rews, terminated, truncated, info = env.step((a0, a1))

        pos_agent0.append(info["agent_positions"][0].copy())
        pos_agent1.append(info["agent_positions"][1].copy())
        total_rew += float(rews[0])
        step_count += 1
        done = terminated or truncated

    success = bool(info["goal_achieved"][0]) if mode == 0.0 else bool(info["collision"])


    return {
        "success": success,
        "reward": total_rew,
        "steps": step_count,
        "traj_agent0": np.array(pos_agent0),
        "traj_agent1": np.array(pos_agent1),
        "goals": goals,
        "hazards": info.get("hazard_positions", np.empty((0, 2))),
        "collision": bool(info["collision"]),
        "dist_final": float(info["dist_goals"][0] if mode == 0.0 else info["dist_agents"]),
    }


def plot_trajectories(
    ep_data: Dict[str, Any],
    mode: float,
    save_path: str,
    show: bool = False,
):
    """Plot 2D bird's-eye view of robots, goals, and obstacles."""
    traj0 = ep_data["traj_agent0"]
    traj1 = ep_data["traj_agent1"]
    goals = ep_data["goals"]
    hazards = ep_data.get("hazards", np.empty((0, 2)))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim([-2.6, 2.6])
    ax.set_ylim([-2.6, 2.6])
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Arena boundaries
    arena_box = plt.Rectangle((-2.5, -2.5), 5.0, 5.0, fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(arena_box)

    # Hazards / Obstacles (Level 1)
    for k, h in enumerate(hazards):
        h_circle = plt.Circle(h, 0.2, color="forestgreen", alpha=0.4, label="Obstacle" if k == 0 else "")
        ax.add_patch(h_circle)

    # Goals
    g0_circle = plt.Circle(goals[0], 0.3, color="red", alpha=0.3, label="Goal 0 (Red)")
    g1_circle = plt.Circle(goals[1], 0.3, color="blue", alpha=0.3, label="Goal 1 (Blue)")
    ax.add_patch(g0_circle)
    ax.add_patch(g1_circle)


    # Robot 0 (Red) trajectory
    ax.plot(traj0[:, 0], traj0[:, 1], "r-", linewidth=2, label="Agent 0 (Red)")
    ax.plot(traj0[0, 0], traj0[0, 1], "ro", markersize=8, label="Agent 0 Start")
    ax.plot(traj0[-1, 0], traj0[-1, 1], "r*", markersize=14, label="Agent 0 End")

    # Robot 1 (Blue) trajectory
    ax.plot(traj1[:, 0], traj1[:, 1], "b--", linewidth=2, label="Agent 1 (Blue)")
    ax.plot(traj1[0, 0], traj1[0, 1], "bo", markersize=8, label="Agent 1 Start")
    ax.plot(traj1[-1, 0], traj1[-1, 1], "b*", markersize=14, label="Agent 1 End")

    mode_name = "Mode 0: Goal-Seeking" if mode == 0.0 else "Mode 1: Adversarial Intercept"
    outcome = "Success" if ep_data["success"] else "Failed"
    ax.set_title(f"{mode_name} | {outcome} ({ep_data['steps']} steps, final dist: {ep_data['dist_final']:.2f}m)")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  -> Saved trajectory plot: {save_path}")
    if show:
        plt.show()
    plt.close()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    dev_str = args.device or cfg.get("device", "auto")
    if dev_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(dev_str)
    print(f"Evaluating on device: {device}")

    # Initialize environment and agent
    base_env = MultiGoal0Env(
        arena_size=cfg["env"]["arena_size"],
        agent_radius=cfg["env"]["agent_radius"],
        goal_radius=cfg["env"]["goal_radius"],
        collision_threshold=cfg["env"]["collision_threshold"],
        max_episode_steps=cfg["env"]["max_episode_steps"],
        level=args.level,
    )
    env = DualModeEnvWrapper(base_env, dwell_steps=cfg["env"].get("dwell_steps", 20))


    agent = RLPDAgent(
        obs_dim=env.ego_obs_dim,
        act_dim=env.single_action_space.shape[0],
        hidden_dims=cfg["rlpd"]["hidden_dims"],
        num_critics=cfg["rlpd"]["num_critics"],
        device=device,
    )

    if os.path.exists(args.checkpoint):
        agent.load(args.checkpoint)
        print(f"Loaded checkpoint from: {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Running with initialized policy.")

    print(f"\n--- Evaluating {args.episodes} episodes per mode ---")
    num_plots = min(args.num_plots, args.episodes) if (args.plot or args.show) else 0

    # Evaluate Mode 0 (Goal Seeking)
    m0_results = []
    for i in range(args.episodes):
        res = run_episode(env, agent, mode=0.0)
        m0_results.append(res)
        if (args.plot or args.show) and i < num_plots:
            plot_trajectories(
                res,
                mode=0.0,
                save_path=os.path.join(args.output_dir, f"mode0_ep{i}.png"),
                show=args.show,
            )

    # Evaluate Mode 1 (Adversarial Interception)
    m1_results = []
    for i in range(args.episodes):
        res = run_episode(env, agent, mode=1.0)
        m1_results.append(res)
        if (args.plot or args.show) and i < num_plots:
            plot_trajectories(
                res,
                mode=1.0,
                save_path=os.path.join(args.output_dir, f"mode1_ep{i}.png"),
                show=args.show,
            )

    m0_success = np.mean([r["success"] for r in m0_results])
    m0_dist = np.mean([r["dist_final"] for r in m0_results])
    m1_success = np.mean([r["success"] for r in m1_results])
    m1_dist = np.mean([r["dist_final"] for r in m1_results])

    print(f"\n================ Evaluation Results ================")
    print(f"Mode 0 (Goal-Seeking)       : Success Rate: {m0_success*100:5.1f}% | Avg Final Goal Dist: {m0_dist:.2f}m")
    print(f"Mode 1 (Adversarial Crash)  : Success Rate: {m1_success*100:5.1f}% | Avg Final Agent Dist: {m1_dist:.2f}m")
    print(f"====================================================")

    if args.plot or args.show:
        abs_out = os.path.abspath(args.output_dir)
        print(f"\nTrajectory plots saved in: {abs_out}")
        print(f"To open them on macOS, run:  open {args.output_dir}/*.png")



if __name__ == "__main__":
    main()
