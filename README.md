# Dual-Mode Adversarial Hindsight Experience Replay with RLPD on Multi-Agent Safety Gym

This repository implements a high-efficiency multi-agent reinforcement learning system based on **Safety Gym** (`MultiGoal0` setting).

In this environment, two mobile Point robots operate under two dynamically conditioned modes:
1. **Mode 0: Goal-Seeking ($m = 0$)**: The robot navigates safely towards its assigned stationary goal.
2. **Mode 1: Adversarial / Collision ($m = 1$)**: The robot intercepts and collides with the other mobile robot.

A **single, shared policy and LayerNorm critic ensemble** controls all agents. Symmetrical data sharing and a novel **Dual-Mode Hindsight Experience Replay (HER)** relabeler ensure extreme sample efficiency:
- **Accidental Collision Relabeling**: If an agent attempting to reach a goal inadvertently collides with the other robot, the trajectory is relabeled in hindsight as an intentional adversarial pursuit ($m = 1$), receiving the terminal collision bonus.
- **Crash-Free Goal Relabeling**: When an agent navigates without colliding, future achieved coordinates are relabeled in hindsight as the assigned goal ($m = 0$). Segments involving collisions are excluded from goal relabeling to maintain clean navigation policies.
- **Universal Agent Symmetry**: All transitions from Agent 0 and Agent 1 are represented in local egocentric coordinates and pooled symmetrically into a shared replay buffer.
- **RLPD Training**: Built on Soft Actor-Critic (SAC) featuring LayerNorm in the critic ensemble and high Update-To-Data (UTD) ratios to extract maximum learning value from every step.

---

## System Architecture

```
                                  +-----------------------------+
                                  |     Safety Gym MultiGoal0   |
                                  |    (2 Point Robots + Goals) |
                                  +--------------+--------------+
                                                 |
                                     (Actions & Simulation)
                                                 v
                                  +-----------------------------+
                                  |     DualModeEnvWrapper      |
                                  |  - Egocentric 17D State     |
                                  |  - Mode Conditioning (0/1)  |
                                  |  - Progress Shaping Rewards |
                                  +--------------+--------------+
                                                 |
                                         (Rollout Stream)
                                                 v
                                  +-----------------------------+
                                  |   DualModeHERRelabeler      |
                                  |  - Crash -> Adversarial HER |
                                  |  - Non-crash -> Goal HER    |
                                  |  - Symmetrical Agent Merge  |
                                  +--------------+--------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  |     Shared Replay Buffer    |
                                  |  (Supports RLPD dual batch) |
                                  +--------------+--------------+
                                                 |
                                        (High-UTD Batches)
                                                 v
                                  +-----------------------------+
                                  |          RLPD Agent         |
                                  |  - LayerNorm Critic Ensemble|
                                  |  - Squashed Gaussian Policy |
                                  |  - Auto Entropy Tuning      |
                                  +-----------------------------+
```

---

## Observation Space

Each robot receives a 17-dimensional egocentric feature vector that is invariant to global position and orientation:

| Index | Feature | Description |
|:-----:|:--------|:------------|
| 0, 1  | $v_{local}$ | Linear velocity in body-relative frame $(\dot{x}_{ego}, \dot{y}_{ego})$ |
| 2     | $\omega$ | Angular yaw velocity |
| 3, 4  | Compass | Unit heading vector $[\cos \theta, \sin \theta]$ |
| 5, 6  | $\Delta p_{goal}^{ego}$ | Egocentric vector to assigned navigation goal |
| 7     | $d_{goal}$ | Euclidean distance to assigned goal |
| 8, 9  | $\Delta p_{other}^{ego}$ | Egocentric vector to other mobile robot |
| 10, 11| $v_{other}^{ego}$ | Relative velocity of other robot in body frame |
| 12    | $d_{other}$ | Euclidean distance to other robot |
| 13    | $m$ | Mode indicator: `0.0` (Goal-Seeking) or `1.0` (Adversarial) |
| 14, 15| $\Delta p_{target}^{ego}$ | Conditioned target: $(1-m)\Delta p_{goal}^{ego} + m \Delta p_{other}^{ego}$ |
| 16    | $d_{target}$ | Conditioned distance: $(1-m) d_{goal} + m d_{other}$ |

---

## Quickstart

### 1. Installation

```bash
# Using uv or pip
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

### 2. Run Tests

```bash
pytest tests/ -v
```

### 3. Training with Weights & Biases

Train the shared policy using Dual-Mode HER and RLPD, with live logging to Weights & Biases:

```bash
# Cloud logging
python train.py --wandb --wandb_project adversarial-her --exp_name ours_dual_her_rlpd

# Or offline logging (synced later with `wandb sync <dir>`)
WANDB_MODE=offline python train.py --wandb --exp_name ours_dual_her_rlpd
```

---

## Baseline Comparisons

To evaluate the contribution of each algorithmic component, compare your method against these standard baselines:

### 1. Our Method: Dual-Mode HER + RLPD (Goal HER + Adversarial Crash HER)
Relabels accidental crashes as intentional hits and clean trajectories as achieved goals:
```bash
python train.py --wandb --exp_name 01_ours_dual_her_rlpd --her_mode full --utd 2
```

### 2. Ablation: Standard Goal HER Only (No Adversarial Crash Relabeling)
Tests the isolated impact of your novel crash-to-adversarial relabeling mechanism:
```bash
python train.py --wandb --exp_name 02_ablation_goal_her_only --her_mode goal_only --utd 2
```

### 3. Baseline: No HER (Standard Multi-Agent RLPD)
Evaluates sample efficiency when no hindsight relabeling is used:
```bash
python train.py --wandb --exp_name 03_baseline_no_her --her_mode none --utd 2
```

### 4. Baseline: Vanilla SAC (No HER, UTD = 1, No LayerNorm)
Standard Soft Actor-Critic without RLPD enhancements:
```bash
python train.py --wandb --exp_name 04_baseline_vanilla_sac --her_mode none --no_layernorm --utd 1
```

---

## Evaluation & Visualization

Evaluate a trained checkpoint across Mode 0 and Mode 1 and plot bird's-eye trajectories:

```bash
python evaluate.py --checkpoint checkpoints/01_ours_dual_her_rlpd_best.pt --episodes 10 --plot
```

Trajectory visualization figures are saved to `eval_results/mode0_ep*.png` and `eval_results/mode1_ep*.png`.

# AdverserialHER
