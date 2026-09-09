"""Point robot model definitions and MuJoCo XML generator for MultiGoal0."""

from __future__ import annotations
import numpy as np


def generate_multigoal_xml(
    arena_size: float = 2.5,
    agent_radius: float = 0.15,
    goal_radius: float = 0.3,
    floor_size: float = 3.5,
    level: int = 0,
    num_hazards: int = 4,
    hazard_radius: float = 0.2,
) -> str:
    """Generate MuJoCo MJCF XML string for 2 Point robots and 2 Goals.
    
    If level >= 1: Adds static hazards/obstacles in the arena (Level 1).
    """
    wall_thick = 0.2
    limit = arena_size - agent_radius - 0.02

    hazards_xml = ""
    if level >= 1:
        for k in range(num_hazards):
            hazards_xml += f"""
    <!-- Hazard / Obstacle {k} -->
    <body name="hazard{k}" pos="0 0 0.01" mocap="true">
      <geom name="hazard{k}_geom" type="cylinder" size="{hazard_radius} 0.02" material="hazard_mat" contype="0" conaffinity="0"/>
    </body>"""

    xml = f"""<mujoco model="safety_point_multi_goal0">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
  <default>
    <joint armature="0.01" damping="0.1"/>
    <geom condim="3" density="1000" friction="0.8 0.1 0.1"/>
  </default>
  <option integrator="Euler" timestep="0.02"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="2048"/>
    <map stiffness="100"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0 0 0" width="800" height="800"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.95 0.95 0.95" rgb2="0.85 0.85 0.85"/>
    <material name="grid_mat" texture="grid" texrepeat="5 5" reflectance="0.2"/>
    <material name="wall_mat" rgba="0.6 0.6 0.6 1"/>
    <material name="agent0_mat" rgba="0.9 0.15 0.15 1"/>
    <material name="agent1_mat" rgba="0.15 0.3 0.9 1"/>
    <material name="arrow0_mat" rgba="1.0 0.8 0.0 1"/>
    <material name="arrow1_mat" rgba="0.0 0.9 0.9 1"/>
    <material name="goal0_mat" rgba="0.9 0.15 0.15 0.4"/>
    <material name="goal1_mat" rgba="0.15 0.3 0.9 0.4"/>
    <material name="hazard_mat" rgba="0.2 0.8 0.2 0.6"/>
  </asset>

  <worldbody>
    <light directional="true" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2" pos="0 0 5" dir="0 0 -1"/>
    <!-- Floor -->
    <geom name="floor" type="plane" size="{floor_size} {floor_size} 0.1" material="grid_mat" conaffinity="1" contype="1"/>

    <!-- Boundary walls -->
    <geom name="wall_n" type="box" size="{arena_size + wall_thick} {wall_thick} 0.3" pos="0 {arena_size + wall_thick} 0.15" material="wall_mat" conaffinity="1" contype="1"/>
    <geom name="wall_s" type="box" size="{arena_size + wall_thick} {wall_thick} 0.3" pos="0 -{arena_size + wall_thick} 0.15" material="wall_mat" conaffinity="1" contype="1"/>
    <geom name="wall_e" type="box" size="{wall_thick} {arena_size + wall_thick} 0.3" pos="{arena_size + wall_thick} 0 0.15" material="wall_mat" conaffinity="1" contype="1"/>
    <geom name="wall_w" type="box" size="{wall_thick} {arena_size + wall_thick} 0.3" pos="-{arena_size + wall_thick} 0 0.15" material="wall_mat" conaffinity="1" contype="1"/>

    <!-- Goals (Mocap geoms, visual sensors, conaffinity=0 so agents can pass through) -->
    <body name="goal0" pos="-1.2 0.0 0.01" mocap="true">
      <geom name="goal0_geom" type="cylinder" size="{goal_radius} 0.01" material="goal0_mat" contype="0" conaffinity="0"/>
    </body>
    <body name="goal1" pos="1.2 0.0 0.01" mocap="true">
      <geom name="goal1_geom" type="cylinder" size="{goal_radius} 0.01" material="goal1_mat" contype="0" conaffinity="0"/>
    </body>{hazards_xml}


    <!-- Robot 0 (Red Agent) -->
    <body name="agent0" pos="-1.0 -1.0 0.1">
      <camera name="agent0_cam" pos="0 0 0.1" fovy="90" mode="fixed"/>
      <joint name="agent0_x" type="slide" axis="1 0 0" range="-{limit:.2f} {limit:.2f}" limited="true" damping="1.0"/>
      <joint name="agent0_y" type="slide" axis="0 1 0" range="-{limit:.2f} {limit:.2f}" limited="true" damping="1.0"/>
      <joint name="agent0_rot" type="hinge" axis="0 0 1" damping="0.2"/>
      <!-- Main body cylinder -->
      <geom name="agent0_geom" type="cylinder" size="{agent_radius} 0.08" material="agent0_mat" conaffinity="1" contype="1"/>
      <!-- Direction indicator arrow -->
      <geom name="agent0_arrow" type="box" size="0.08 0.02 0.02" pos="0.08 0 0.05" material="arrow0_mat" contype="0" conaffinity="0"/>
    </body>

    <!-- Robot 1 (Blue Agent) -->
    <body name="agent1" pos="1.0 1.0 0.1">
      <camera name="agent1_cam" pos="0 0 0.1" fovy="90" mode="fixed"/>
      <joint name="agent1_x" type="slide" axis="1 0 0" range="-{limit:.2f} {limit:.2f}" limited="true" damping="1.0"/>
      <joint name="agent1_y" type="slide" axis="0 1 0" range="-{limit:.2f} {limit:.2f}" limited="true" damping="1.0"/>
      <joint name="agent1_rot" type="hinge" axis="0 0 1" damping="0.2"/>
      <!-- Main body cylinder -->
      <geom name="agent1_geom" type="cylinder" size="{agent_radius} 0.08" material="agent1_mat" conaffinity="1" contype="1"/>
      <!-- Direction indicator arrow -->
      <geom name="agent1_arrow" type="box" size="0.08 0.02 0.02" pos="0.08 0 0.05" material="arrow1_mat" contype="0" conaffinity="0"/>
    </body>


    <!-- Overhead Camera -->
    <camera name="overhead" pos="0 0 5.5" mode="fixed" fovy="60"/>
  </worldbody>

  <actuator>
    <!-- Robot 0: Forward force + steering torque -->
    <motor name="agent0_drive_x" joint="agent0_x" gear="1.0" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="agent0_drive_y" joint="agent0_y" gear="1.0" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="agent0_steer" joint="agent0_rot" gear="0.5" ctrllimited="true" ctrlrange="-1 1"/>

    <!-- Robot 1: Forward force + steering torque -->
    <motor name="agent1_drive_x" joint="agent1_x" gear="1.0" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="agent1_drive_y" joint="agent1_y" gear="1.0" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="agent1_steer" joint="agent1_rot" gear="0.5" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""
    return xml
