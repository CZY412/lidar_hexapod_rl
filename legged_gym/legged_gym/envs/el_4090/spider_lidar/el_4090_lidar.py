# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from time import time
import numpy as np
import os
import math
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict
from legged_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, quat_from_euler_xyz_tensor, cart2sphere, downsample_spherical_points_vectorized, farthest_point_sampling
from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.el_4090.spider_nomal.el_4090 import EL_4090

from legged_gym.envs.el_4090.spider_lidar.el_4090_lidar_config import El4090LidarCfg

"""
ElSpider LiDAR Environment for Confined Space Navigation
基于激光雷达的六足机器人受限空间强化学习避障运动控制
"""
import sys
# from isaacgym.torch_utils import quat_apply, quat_mul, quat_rotate_inverse
import warp as wp
import trimesh
from legged_gym.utils import GaitScheduler, GaitSchedulerCfg, AsyncGaitSchedulerCfg, AsyncGaitScheduler
from legged_gym.utils.gym_visualizer import GymVisualizer

from LidarSensor.lidar_sensor import LidarSensor
from LidarSensor.sensor_config.lidar_sensor_config import LidarConfig, LidarType

@torch.no_grad()
def get_elair_xsym_obs_act(obs: torch.Tensor = None, actions: torch.Tensor = None, env = None, obs_type: str = "policy") -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply symmetry transformation to observations and actions for the ElSpider robot.
    
    This function augments the dataset by mirroring the robot's left-right sides.
    
    Args:
        obs: Observations tensor [batch, obs_dim]
        actions: Actions tensor [batch, action_dim]
        env: Environment instance (for reference)
        obs_type: Type of observation ("policy" or "critic")
        
    Returns:
        Tuple of transformed observations and actions tensors
    """
    device = obs.device if obs is not None else actions.device
    batch_size = obs.shape[0] if obs is not None else actions.shape[0]
    
    # Original and mirrored observations/actions
    # [batch*2, dim] where first batch is original, second batch is mirrored
    
    if obs is not None:
        # Mirror the observations for ElSpider which has 6 legs
        # For policy observation, the structure is:
        # [0:3] - base_lin_vel (mirror y)
        # [3:6] - base_ang_vel (mirror x, z)
        # [6:9] - projected_gravity (mirror y)
        # [9:12] - commands (mirror y for lin_vel, mirror ang_vel_z)
        # [12:30] - dof_pos (swap left-right sides)
        # [30:48] - dof_vel (swap left-right sides)
        # [48:66] - previous actions (swap left-right sides)
        # [66:253] - height measurements (mirror left-right pattern)
        
        # Create mirrored observations
        obs_mirrored = obs.clone()
        
        # Mirror linear velocity y-component
        obs_mirrored[:, 1] = -obs[:, 1]
        
        # Mirror angular velocity x and z components
        obs_mirrored[:, 3] = -obs[:, 3]
        obs_mirrored[:, 5] = -obs[:, 5]
        
        # Mirror projected gravity y-component
        obs_mirrored[:, 7] = -obs[:, 7]
        
        # Mirror command velocities (y and angular z)
        obs_mirrored[:, 10] = -obs[:, 10]
        obs_mirrored[:, 11] = -obs[:, 11]
        
        # Swap left-right DOF positions - ElSpider has 6 legs with 3 DOFs each
        # Right side DOFs: 0-8, Left side DOFs: 9-17

        # Swap right and left DOF positions
        obs_mirrored[:, 12:21] = obs[:, 21:30]  # Right legs get left leg positions
        obs_mirrored[:, 21:30] = obs[:, 12:21]  # Left legs get right leg positions
        
        # Mirror DOF velocities (30:48) using the same mapping as positions
        obs_mirrored[:, 30:39] = obs[:, 39:48]  # Right legs get left leg velocities
        obs_mirrored[:, 39:48] = obs[:, 30:39]  # Left legs get right leg velocities
        
        # Mirror previous actions (48:66) using the same mapping
        obs_mirrored[:, 48:57] = obs[:, 57:66]  # Right legs get left leg actions
        obs_mirrored[:, 57:66] = obs[:, 48:57]  # Left legs get right leg actions
        
        # Mirror height measurements (66:253) if present
        if obs.shape[1] > 164:
            # The height measurements are in a grid pattern
            # Original grid pattern: measured_points_x × measured_points_y
            # For ElSpider, this is typically 17×11 = 187 points
            
            # We need to mirror the points along the y-axis
            # If we have 17 points in x and 11 in y, the indices form a 17×11 grid
            
            height_measurements_start = 66
            x_points = 17  # Number of points along x-axis (from config)
            y_points = 11  # Number of points along y-axis (from config)
            
            for x in range(x_points):
                for y in range(y_points):
                    # Calculate original and mirrored indices
                    original_idx = height_measurements_start + x*y_points + y
                    mirrored_y = y_points - y - 1  # Flip y coordinate
                    mirrored_idx = height_measurements_start + x*y_points + mirrored_y
                    
                    # Swap the height measurements
                    obs_mirrored[:, original_idx] = obs[:, mirrored_idx]
        
        # Combine original and mirrored observations
        obs_augmented = torch.cat([obs, obs_mirrored], dim=0)
    else:
        obs_augmented = None
    
    if actions is not None:
        # Mirror the actions
        # ElSpider has 18 actions (6 legs × 3 joints)
        # Right legs: 0-8, Left legs: 9-17
        actions_mirrored = actions.clone()
        
        actions_mirrored[:, 0:9] = actions[:, 9:18]  # Right legs get left leg actions
        actions_mirrored[:, 9:18] = actions[:, 0:9]  # Left legs get right leg actions
        
        # Combine original and mirrored actions
        actions_augmented = torch.cat([actions, actions_mirrored], dim=0) if actions is not None else None
    else:
        actions_augmented = None
    
    return obs_augmented, actions_augmented


class El4090Lidar(EL_4090):
    cfg : El4090LidarCfg
     # env Init
    def __init__(self, cfg: El4090LidarCfg, sim_params, physics_engine, sim_device, headless,task_name="el4090_spider"):
        """El4090 robot with LiDAR sensor for confined space navigation.
        
        This class extends El4090 to include:
        - LiDAR sensor integration using OmniPerception
        - LiDAR-based observations for obstacle avoidance
        - Rewards for collision avoidance in confined spaces
        """
        # Initialize LiDAR configuration before calling parent __init__
        self._init_lidar_cfg(cfg)
        
        # Call parent constructor
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # Initialize LiDAR sensor after environment is created
        self._init_lidar_sensor()
        
        print(f"[ElSpiderLidar] Initialized with LiDAR sensor")
        print(f"  - LiDAR type: {self.lidar_cfg.sensor_type}")
        print(f"  - LiDAR obs dim: {self.cfg.env.num_lidar_obs}")
        print(f"  - Total obs dim: {self.cfg.env.num_observations}")
        print(f"  - Goal navigation: {getattr(self.cfg.terrain, 'goal_navigation', False)}")
        if getattr(self.cfg.terrain, 'goal_navigation', False):
            print(f"  - Goal offset Y: {getattr(self.cfg.terrain, 'goal_offset_y', 4.0)}m")


    def _init_lidar_cfg(self, cfg):
        """Initialize LiDAR configuration from environment config."""
        self.lidar_cfg = LidarConfig()
        
        # Set LiDAR type from config if specified
        if hasattr(cfg, 'lidar') and hasattr(cfg.lidar, 'sensor_type'):
            self.lidar_cfg.sensor_type = cfg.lidar.sensor_type
        else:
            # Default to simple grid for lower computational cost
            self.lidar_cfg.sensor_type = LidarType.SIMPLE_GRID
        
        # Configure LiDAR parameters
        if hasattr(cfg, 'lidar'):
            lidar_cfg = cfg.lidar
            self.lidar_cfg.update_frequency = getattr(lidar_cfg, 'update_frequency', 20.0)
            self.lidar_cfg.max_range = getattr(lidar_cfg, 'max_range', 5.0)
            self.lidar_cfg.min_range = getattr(lidar_cfg, 'min_range', 0.1)
            self.lidar_cfg.horizontal_line_num = getattr(lidar_cfg, 'horizontal_line_num', 36)
            self.lidar_cfg.vertical_line_num = getattr(lidar_cfg, 'vertical_line_num', 10)
            self.lidar_cfg.horizontal_fov_deg_min = getattr(lidar_cfg, 'horizontal_fov_deg_min', -180)
            self.lidar_cfg.horizontal_fov_deg_max = getattr(lidar_cfg, 'horizontal_fov_deg_max', 180)
            self.lidar_cfg.vertical_fov_deg_min = getattr(lidar_cfg, 'vertical_fov_deg_min', -15)
            self.lidar_cfg.vertical_fov_deg_max = getattr(lidar_cfg, 'vertical_fov_deg_max', 15)
            
            # LiDAR observation configuration
            self.num_theta_bins = getattr(lidar_cfg, 'num_theta_bins', 12)
            self.num_phi_bins = getattr(lidar_cfg, 'num_phi_bins', 8)
        else:
            # Default configuration for grid LiDAR
            self.lidar_cfg.update_frequency = 20.0
            self.lidar_cfg.max_range = 5.0
            self.lidar_cfg.min_range = 0.1
            self.lidar_cfg.horizontal_line_num = 36
            self.lidar_cfg.vertical_line_num = 10
            self.lidar_cfg.horizontal_fov_deg_min = -180
            self.lidar_cfg.horizontal_fov_deg_max = 180
            self.lidar_cfg.vertical_fov_deg_min = -15
            self.lidar_cfg.vertical_fov_deg_max = 15
            self.num_theta_bins = 12
            self.num_phi_bins = 8
        
        # Set dt to match simulation
        self.lidar_cfg.dt = 0.02  # Will be updated in init
        self.lidar_cfg.pointcloud_in_world_frame = False  # Local frame for observations

    def _init_lidar_sensor(self):
        """Initialize the LiDAR sensor after simulation is created."""
        # Update dt from simulation
        self.lidar_cfg.dt = self.dt
        
        # Create WARP mesh from terrain
        self._create_warp_mesh()
        
        # Create sensor tensor dictionary
        self.warp_tensor_dict = self._create_warp_tensor_dict()
        
        # Initialize LiDAR sensor
        self.lidar_sensor = LidarSensor(
            env=self.warp_tensor_dict,
            env_cfg=None,
            sensor_config=self.lidar_cfg,
            num_sensors=1,
            device=self.device
        )
        
        # Capture render graph for faster execution
        self.lidar_sensor.capture()
        
        # Initialize timing
        self.lidar_update_time = 0.0
        self.lidar_update_interval = 1.0 / self.lidar_cfg.update_frequency

    def _create_warp_mesh(self):
        """Create WARP mesh from terrain for ray casting."""
        wp.init()
        
        # Get terrain mesh vertices and triangles
        if hasattr(self, 'terrain') and self.terrain is not None:
            if hasattr(self.terrain, 'vertices') and self.terrain.vertices is not None:
                vertices = self.terrain.vertices.copy()
                triangles = self.terrain.triangles.copy()
                
                # Apply terrain offset
                if hasattr(self.cfg.terrain, 'border_size'):
                    vertices[:, 0] -= self.cfg.terrain.border_size
                    vertices[:, 1] -= self.cfg.terrain.border_size
            else:
                # Create simple ground plane if no terrain mesh
                vertices = np.array([
                    [-50, -50, 0],
                    [50, -50, 0],
                    [50, 50, 0],
                    [-50, 50, 0]
                ], dtype=np.float32)
                triangles = np.array([
                    [0, 1, 2],
                    [0, 2, 3]
                ], dtype=np.int32)
        else:
            # Create simple ground plane
            vertices = np.array([
                [-50, -50, 0],
                [50, -50, 0],
                [50, 50, 0],
                [-50, 50, 0]
            ], dtype=np.float32)
            triangles = np.array([
                [0, 1, 2],
                [0, 2, 3]
            ], dtype=np.int32)
        
        # Convert to WARP arrays
        vertex_tensor = torch.tensor(vertices, device=self.device, dtype=torch.float32)
        vertex_wp = wp.from_torch(vertex_tensor, dtype=wp.vec3)
        faces_wp = wp.from_numpy(triangles.flatten().astype(np.int32), dtype=wp.int32, device=self.device)
        
        # Create WARP mesh
        self.wp_mesh = wp.Mesh(points=vertex_wp, indices=faces_wp)
        self.mesh_ids = wp.array([self.wp_mesh.id], dtype=wp.uint64)

    def _create_warp_tensor_dict(self):
        """Create tensor dictionary for LiDAR sensor."""
        warp_dict = {}
        
        # Sensor position and orientation tensors
        self.sensor_pos_tensor = torch.zeros(self.num_envs, 3, device=self.device)
        self.sensor_quat_tensor = torch.zeros(self.num_envs, 4, device=self.device)
        
        # LiDAR mounting offset (on top of robot body)
        # Default: 0.15m above body center, facing forward
        if hasattr(self.cfg, 'lidar') and hasattr(self.cfg.lidar, 'sensor_offset'):
            self.sensor_translation_local = torch.tensor(
                self.cfg.lidar.sensor_offset, device=self.device
            )
        else:
            self.sensor_translation_local = torch.tensor([0.0, 0.0, 0.15], device=self.device)
        
        # Sensor rotation offset (no rotation by default)
        if hasattr(self.cfg, 'lidar') and hasattr(self.cfg.lidar, 'sensor_rotation_deg'):
            roll = np.deg2rad(self.cfg.lidar.sensor_rotation_deg[0])
            pitch = np.deg2rad(self.cfg.lidar.sensor_rotation_deg[1])
            yaw = np.deg2rad(self.cfg.lidar.sensor_rotation_deg[2])
            self.sensor_offset_quat_local = quat_from_euler_xyz_tensor(
                torch.tensor([roll], device=self.device),
                torch.tensor([pitch], device=self.device),
                torch.tensor([yaw], device=self.device)
            ).squeeze()
        else:
            self.sensor_offset_quat_local = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        
        # Expand to all environments
        self.sensor_translation = self.sensor_translation_local.repeat(self.num_envs, 1)
        self.sensor_offset_quat = self.sensor_offset_quat_local.repeat(self.num_envs, 1)
        
        # Update initial sensor poses
        self._update_sensor_pose()
        
        # Populate dictionary
        warp_dict['device'] = self.device
        warp_dict['num_envs'] = self.num_envs
        warp_dict['sensor_pos_tensor'] = self.sensor_pos_tensor
        warp_dict['sensor_quat_tensor'] = self.sensor_quat_tensor
        warp_dict['mesh_ids'] = self.mesh_ids
        
        return warp_dict

    def _update_sensor_pose(self):
        """Update LiDAR sensor pose based on robot base pose."""
        # Compute sensor position in world frame
        self.sensor_pos_tensor[:] = self.root_states[:, :3] + quat_apply(
            self.root_states[:, 3:7], self.sensor_translation
        )
        
        # Compute sensor orientation in world frame
        self.sensor_quat_tensor[:] = quat_mul(
            self.root_states[:, 3:7], self.sensor_offset_quat
        )

    def _init_buffers(self):
        """Initialize buffers including LiDAR observation buffers and goal buffers."""
        # Set goal_navigation flag BEFORE super()._init_buffers()
        # because _get_noise_scale_vec is called inside super()._init_buffers()
        self.goal_navigation = getattr(self.cfg.terrain, 'goal_navigation', False)

        super()._init_buffers()

        # LiDAR observation buffers
        num_lidar_obs = self.num_theta_bins * self.num_phi_bins
        self.lidar_obs_buf = torch.zeros(
            self.num_envs, num_lidar_obs, device=self.device, requires_grad=False
        )
        
        # Raw LiDAR data buffers
        total_rays = self.lidar_cfg.horizontal_line_num * self.lidar_cfg.vertical_line_num
        self.lidar_points_buf = torch.zeros(
            self.num_envs, total_rays, 3, device=self.device, requires_grad=False
        )
        self.lidar_dist_buf = torch.zeros(
            self.num_envs, total_rays, device=self.device, requires_grad=False
        )
        self.downsampled_cloud = torch.zeros(
            self.num_envs, 1, total_rays, 3, device=self.device, requires_grad=False
        )
        
        # Minimum distance to obstacles (for rewards)
        self.min_obstacle_dist = torch.ones(
            self.num_envs, device=self.device, requires_grad=False
        ) * self.lidar_cfg.max_range

        # Sector-wise obstacle distances for maneuvering decisions
        self.front_obstacle_dist = torch.ones(
            self.num_envs, device=self.device, requires_grad=False
        ) * self.lidar_cfg.max_range
        self.left_obstacle_dist = torch.ones(
            self.num_envs, device=self.device, requires_grad=False
        ) * self.lidar_cfg.max_range
        self.right_obstacle_dist = torch.ones(
            self.num_envs, device=self.device, requires_grad=False
        ) * self.lidar_cfg.max_range
        
        # Goal navigation buffers
        self.goal_obs_buf = torch.zeros(self.num_envs, 2, device=self.device)  # [angle, dist]
        if self.goal_navigation:
            self.goal_positions = torch.zeros(self.num_envs, 3, device=self.device)  # xyz goal
            self.goal_distance = torch.zeros(self.num_envs, device=self.device)
            self.prev_goal_distance = torch.zeros(self.num_envs, device=self.device)
            self.goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.goal_offset_x = getattr(self.cfg.terrain, 'goal_offset_x', 0.0)
            self.goal_offset_y = getattr(self.cfg.terrain, 'goal_offset_y', 4.0)
            waypoint_offsets = getattr(self.cfg.terrain, 'goal_waypoints', None)
            if waypoint_offsets is not None and len(waypoint_offsets) > 0:
                self.goal_waypoints = torch.tensor(waypoint_offsets, dtype=torch.float, device=self.device)
                self.num_goal_waypoints = self.goal_waypoints.shape[0]
            else:
                self.goal_waypoints = None
                self.num_goal_waypoints = 0
            self.current_goal_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.waypoint_reach_threshold = getattr(
                self.cfg.rewards,
                'waypoint_reach_threshold',
                getattr(self.cfg.rewards, 'goal_reach_threshold', 1.0),
            )

    def _debug_info(self):
        """Print debug information for the specified environment(s)"""
        if not self.cfg.env.debug_mode:
            return
        
        self.debug_step_counter += 1
        if self.debug_step_counter % self.cfg.env.debug_interval != 0:
            return
        
        env_id = self.cfg.env.debug_env_id
        
        # Determine which environments to print
        if env_id == -1:
            # Print all environments
            env_ids_to_print = list(range(self.num_envs))
        else:
            # Print specified environment
            if env_id >= self.num_envs:
                return
            env_ids_to_print = [env_id]
        
        print("\n" + "="*80)
        print(f"DEBUG INFO - Step {self.common_step_counter} | Printing {len(env_ids_to_print)} environment(s)")
        print("="*80)
        
        for env_idx in env_ids_to_print:
            print(f"\n{'─'*80}")
            print(f"Environment {env_idx} | Episode Length: {self.episode_length_buf[env_idx].item()}")
            print(f"{'─'*80}")
            
            # Base state
            print(f"\n[Base State]")
            print(f"  Position:     [{self.base_pos[env_idx, 0]:.3f}, {self.base_pos[env_idx, 1]:.3f}, {self.base_pos[env_idx, 2]:.3f}]")
            print(f"  Base Height:  {self.base_pos[env_idx, 2]:.3f} m")

            # Contact info and forces
            contact = self.contact_forces[env_idx, self.feet_indices, 2] > 1.
            contact_forces = self.contact_forces[env_idx, self.feet_indices, :]
            contact_forces_z = contact_forces[:, 2]
            max_contact_force = torch.max(contact_forces_z).item()
            max_force_idx = torch.argmax(contact_forces_z).item()
            
            # Foot names for better readability
            foot_names = ['LB', 'LF', 'LM', 'RB', 'RF', 'RM']
            
            print(f"\n[Contact Info]")
            print(f"  Feet Contact: {contact.cpu().numpy()}")
            print(f"  Contact Forces (X,Y,Z) [N]:")
            for i, name in enumerate(foot_names):
                fx = contact_forces[i, 0].item()
                fy = contact_forces[i, 1].item()
                fz = contact_forces[i, 2].item()
                f_total = torch.norm(contact_forces[i]).item()
                contact_str = "✓" if contact[i].item() else "✗"
                print(f"    {name}: [{fx:7.2f}, {fy:7.2f}, {fz:7.2f}] (Total: {f_total:7.2f}) {contact_str}")
            print(f"  Max Contact Force: {max_contact_force:.2f} N (Foot {foot_names[max_force_idx]})")
            print(f"  Total Ground Force: {torch.sum(contact_forces_z).item():.2f} N")
            print(f"  Feet Air Time: {self.feet_air_time[env_idx].cpu().numpy()}")
            
        
        print("\n" + "="*80 + "\n")
    
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, :3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_lin_acc[:] = self.base_lin_acc[:] * self.acc_ema + (1 - self.acc_ema) * \
            quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.base_ang_acc[:] = self.base_ang_acc[:] * self.acc_ema + (1 - self.acc_ema) * \
            quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13] - self.last_root_vel[:, 3:]) / self.dt
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self._post_physics_step_callback()



        """Update after physics step, including LiDAR sensor and goal tracking."""
        # Update sensor pose before parent's post_physics_step
        self._update_sensor_pose()
        
        # Update LiDAR sensor
        self.lidar_update_time += self.dt
        if self.lidar_update_time >= self.lidar_update_interval:
            self._update_lidar()
            self.lidar_update_time = 0.0
        
        # Update goal distance tracking (before rewards are computed)
        if self.goal_navigation:
            self._update_goal_tracking()



        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        # Call debug info after all computations are done
        self._debug_info()        

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def _post_physics_step_callback(self):
        """Override to add goal-directed heading command."""
        # Resample commands on schedule
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)
                   == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        
        # Goal-directed heading: override heading command to point toward goal
        if self.goal_navigation and getattr(self.cfg.commands, 'goal_directed', False):
            # Compute heading angle from robot to goal
            goal_vec = self.goal_positions[:, :2] - self.root_states[:, :2]
            goal_heading = torch.atan2(goal_vec[:, 1], goal_vec[:, 0])
            
            # Set heading command to goal direction
            self.commands[:, 3] = goal_heading
            
            # Convert heading to angular velocity command (P controller)
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            heading_error = goal_heading - heading
            # Wrap to [-pi, pi]
            heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
            self.commands[:, 2] = torch.clip(0.8 * heading_error, -1., 1.)
            
            # Forward velocity: obstacle-aware goal speed
            # When facing goal: higher speed; near obstacles: slow down or back up
            max_vel = self.command_ranges["lin_vel_x"][1]  # 1.2 m/s
            min_vel = 0.15
            
            # facing_factor: 1.0 when facing goal, 0.0 when perpendicular, -1 when away
            cos_heading = torch.cos(heading_error)
            # Map cos_heading from [-1, 1] to [min_vel, max_vel]
            speed_factor = (cos_heading + 1.0) / 2.0  # 0~1
            goal_speed = min_vel + (max_vel - min_vel) * speed_factor

            # Obstacle-aware speed scaling using LiDAR sector distances
            safe_dist = getattr(self.cfg.rewards, 'safe_obstacle_dist', 0.5)
            danger_dist = getattr(self.cfg.rewards, 'danger_obstacle_dist', 0.15)
            obs_speed_scale = torch.clamp(
                (self.min_obstacle_dist - danger_dist) / (safe_dist - danger_dist + 1e-6),
                0.0, 1.0
            )

            front_pressure = torch.clamp(
                (safe_dist - self.front_obstacle_dist) / (safe_dist - danger_dist + 1e-6),
                0.0,
                1.0,
            )

            # If the front is blocked, back off a bit instead of insisting forward.
            backoff_speed = -0.02 * front_pressure
            forward_speed = goal_speed * torch.clamp(obs_speed_scale, 0.35, 1.0) * (1.0 - 0.5 * front_pressure)
            self.commands[:, 0] = torch.where(front_pressure > 0.92, backoff_speed, forward_speed)
            
            # Lateral velocity: turn toward the clearer side when blocked.
            side_clearance = torch.clamp(
                (self.right_obstacle_dist - self.left_obstacle_dist) / (safe_dist + 1e-6),
                -1.0,
                1.0,
            )
            self.commands[:, 1] = torch.clip(
                -0.3 * torch.sin(heading_error) + 0.20 * side_clearance * front_pressure,
                -0.3,
                0.3,
            )
        else:
            # Default heading command conversion
            if self.cfg.commands.heading_command:
                forward = quat_apply(self.base_quat, self.forward_vec)
                heading = torch.atan2(forward[:, 1], forward[:, 0])
                self.commands[:, 2] = torch.clip(0.5 * (self.commands[:, 3] - heading), -1., 1.)
        
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _update_lidar(self):
        """Update LiDAR sensor and process observations."""
        # Get raw LiDAR data
        lidar_points, lidar_dist = self.lidar_sensor.update()

        # Reshape data robustly using actual sensor output shape
        lidar_points = lidar_points.contiguous().view(self.num_envs, -1, 3)
        lidar_dist = lidar_dist.contiguous().view(self.num_envs, -1)

        if lidar_points.shape[1] > 0:
            self.downsampled_cloud = farthest_point_sampling(
                lidar_points.unsqueeze(1), sample_size=1
            )
        else:
            self.downsampled_cloud = torch.zeros(
                self.num_envs, 1, 1, 3, device=self.device, requires_grad=False
            )

        # Keep fixed-size buffers by truncating or padding with safe defaults
        total_rays = self.lidar_cfg.horizontal_line_num * self.lidar_cfg.vertical_line_num
        used_rays = min(total_rays, lidar_points.shape[1])

        self.lidar_points_buf.zero_()
        self.lidar_dist_buf.fill_(self.lidar_cfg.max_range)
        self.lidar_points_buf[:, :used_rays, :] = lidar_points[:, :used_rays, :]
        self.lidar_dist_buf[:, :used_rays] = lidar_dist[:, :used_rays]
        
        # Compute minimum obstacle distance (vectorized, no Python loop)
        # Treat invalid returns as no-hit max-range so they don't trigger false collision
        valid_hit = (self.lidar_dist_buf > self.lidar_cfg.min_range) & (
            self.lidar_dist_buf < self.lidar_cfg.max_range
        )
        clamped_dist = torch.where(
            valid_hit,
            self.lidar_dist_buf,
            torch.full_like(self.lidar_dist_buf, self.lidar_cfg.max_range)
        )
        self.min_obstacle_dist[:] = clamped_dist.min(dim=1)[0]
        
        # Convert to spherical coordinates and downsample for observation
        sphere_points = cart2sphere(self.lidar_points_buf.view(-1, 3)).view(self.num_envs, -1, 3)
        downsampled = downsample_spherical_points_vectorized(
            sphere_points, self.num_theta_bins, self.num_phi_bins, self.lidar_cfg.max_range
        )

        # Sector distances: front / left / right. These help the policy learn
        # when it should back up or sidestep instead of pushing straight ahead.
        bin_r = downsampled[:, :, 0]
        bin_theta = downsampled[:, :, 1]
        bin_phi = downsampled[:, :, 2]
        sector_max = self.lidar_cfg.max_range

        front_mask = (torch.abs(bin_theta) <= 0.52) & (torch.abs(bin_phi) <= 0.35)
        left_mask = (bin_theta > 0.52) & (bin_theta <= 2.62) & (torch.abs(bin_phi) <= 0.35)
        right_mask = (bin_theta < -0.52) & (bin_theta >= -2.62) & (torch.abs(bin_phi) <= 0.35)

        self.front_obstacle_dist[:] = torch.where(
            front_mask, bin_r, torch.full_like(bin_r, sector_max)
        ).min(dim=1)[0]
        self.left_obstacle_dist[:] = torch.where(
            left_mask, bin_r, torch.full_like(bin_r, sector_max)
        ).min(dim=1)[0]
        self.right_obstacle_dist[:] = torch.where(
            right_mask, bin_r, torch.full_like(bin_r, sector_max)
        ).min(dim=1)[0]
        
        # Use normalized distance as observation (0 = close, 1 = far/no hit)
        self.lidar_obs_buf[:] = downsampled[:, :, 0].clamp(0, self.lidar_cfg.max_range) / self.lidar_cfg.max_range

    def _update_goal_tracking(self):
        """Update goal distance and goal observations each step."""
        # Save previous distance
        self.prev_goal_distance[:] = self.goal_distance.clone()

        # Compute current distance to active goal / waypoint (XY plane)
        self.goal_distance[:] = torch.norm(
            self.root_states[:, :2] - self.goal_positions[:, :2], dim=1
        )

        if self.goal_waypoints is not None and self.num_goal_waypoints > 0:
            reached_waypoint = self.goal_distance < self.waypoint_reach_threshold
            final_waypoint = self.current_goal_idx >= (self.num_goal_waypoints - 1)
            advance_waypoint = reached_waypoint & (~final_waypoint)

            if torch.any(advance_waypoint):
                advance_env_ids = advance_waypoint.nonzero(as_tuple=False).flatten()
                self.current_goal_idx[advance_env_ids] += 1
                self._set_current_goal_positions(advance_env_ids)
                self.goal_distance[advance_env_ids] = torch.norm(
                    self.root_states[advance_env_ids, :2] - self.goal_positions[advance_env_ids, :2],
                    dim=1,
                )
                self.prev_goal_distance[advance_env_ids] = self.goal_distance[advance_env_ids].clone()
                reached_waypoint[advance_env_ids] = False

            self.goal_reached[:] = reached_waypoint & final_waypoint
        else:
            # Check if goal reached
            goal_threshold = getattr(self.cfg.rewards, 'goal_reach_threshold', 1.0)
            self.goal_reached[:] = self.goal_distance < goal_threshold
        
        # Compute goal observation: [direction_angle, normalized_distance]
        # Direction to goal in robot's local frame
        goal_vec_world = self.goal_positions[:, :2] - self.root_states[:, :2]  # (N, 2)
        goal_vec_local = self._world_to_local_2d(goal_vec_world)  # (N, 2)
        
        # Angle to goal (in local frame): atan2(local_y, local_x)
        goal_angle = torch.atan2(goal_vec_local[:, 1], goal_vec_local[:, 0])  # (-pi, pi)
        
        # Normalized distance (0 = at goal, 1 = max distance)
        goal_max_dist = getattr(self.cfg.rewards, 'goal_max_distance', 8.0)
        goal_dist_normalized = torch.clamp(self.goal_distance / goal_max_dist, 0, 1)
        
        self.goal_obs_buf[:, 0] = goal_angle / 3.14159  # Normalize to [-1, 1]
        self.goal_obs_buf[:, 1] = goal_dist_normalized
    
    def compute_observations(self):
        """Compute observations including LiDAR data and goal information."""
        # Base observations (same as ElSpider)
        base_obs = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)
        
        # Add height measurements if configured
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1, 1.
            ) * self.obs_scales.height_measurements
            base_obs = torch.cat((base_obs, heights), dim=-1)
        
        # Add LiDAR observations
        obs_parts = [base_obs, self.lidar_obs_buf]
        
        # Keep fixed observation dimension by always reserving goal observation slots
        obs_parts.append(self.goal_obs_buf)
        
        self.obs_buf = torch.cat(obs_parts, dim=-1)
        
        # Add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _get_noise_scale_vec(self, cfg):
        """Get noise scale vector including LiDAR and goal observation noise."""
        noise_vec = torch.zeros(self.cfg.env.num_observations, device=self.device)
        self.add_noise = cfg.noise.add_noise
        noise_scales = cfg.noise.noise_scales
        noise_level = cfg.noise.noise_level
        
        # Base proprioception noise (same as ElSpider parent)
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.  # commands
        noise_vec[12:30] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[30:48] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[48:66] = 0.  # previous actions

        proprio_obs_dim = 66
        height_obs_dim = 0
        
        # Height measurements noise
        if self.cfg.terrain.measure_heights:
            height_obs_dim = len(self.cfg.terrain.measured_points_x) * len(self.cfg.terrain.measured_points_y)
            height_end = proprio_obs_dim + height_obs_dim
            noise_vec[proprio_obs_dim:height_end] = (
                noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
            )
        
        # LiDAR observation noise
        num_lidar_obs = getattr(self.cfg.env, 'num_lidar_obs', self.num_theta_bins * self.num_phi_bins)
        lidar_noise = getattr(noise_scales, 'lidar', 0.05) * noise_level
        lidar_start = proprio_obs_dim + height_obs_dim
        lidar_end = lidar_start + num_lidar_obs
        noise_vec[lidar_start:lidar_end] = lidar_noise
        
        # Goal observation noise (small)
        if self.goal_navigation:
            noise_vec[lidar_end:lidar_end+2] = 0.02 * noise_level
        
        return noise_vec

    def _reset_root_states(self, env_ids):
        """Reset root states with a goal-facing spawn bias for confined navigation."""
        super()._reset_root_states(env_ids)

        if len(env_ids) == 0 or not self.goal_navigation:
            return

        if self.custom_origins and getattr(self.cfg.terrain, 'corridor_uniform_width', False):
            x_jitter = torch.empty(len(env_ids), device=self.device).uniform_(-0.12, 0.12)
            y_jitter = torch.empty(len(env_ids), device=self.device).uniform_(-0.25, 0.25)
            self.root_states[env_ids, 0] = self.env_origins[env_ids, 0] + x_jitter
            self.root_states[env_ids, 1] = self.env_origins[env_ids, 1] + y_jitter

        yaw = torch.full((len(env_ids),), np.pi * 0.5, device=self.device)
        yaw += torch.empty(len(env_ids), device=self.device).uniform_(-0.15, 0.15)
        zeros = torch.zeros_like(yaw)
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz_tensor(zeros, zeros, yaw)

        self.root_states[env_ids, 7:13] = torch.empty((len(env_ids), 6), device=self.device).uniform_(-0.05, 0.05)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

    def reset_idx(self, env_ids):
        """Reset environments including LiDAR sensor and goal positions."""
        # Log goal stats before reset (while data is still valid)
        if self.goal_navigation and len(env_ids) > 0 and self.init_done and "episode" in self.extras:
            goal_reached_ratio = self.goal_reached[env_ids].float().mean().item()
            mean_goal_dist = self.goal_distance[env_ids].mean().item()
            self.extras["episode"]["goal_reached_ratio"] = goal_reached_ratio
            self.extras["episode"]["mean_goal_distance"] = mean_goal_dist
        
        super().reset_idx(env_ids)
        
        # Reset LiDAR buffers for reset environments
        if len(env_ids) > 0:
            self.lidar_obs_buf[env_ids] = 0.0
            self.min_obstacle_dist[env_ids] = self.lidar_cfg.max_range
            
            # Reset LiDAR sensor for specific environments
            if hasattr(self, 'lidar_sensor'):
                self.lidar_sensor.reset(env_ids)
            
            # Set goal positions for reset environments
            if self.goal_navigation:
                self._set_goal_positions(env_ids)

    def _set_goal_positions(self, env_ids):
        """Set goal positions for given environments.
        Goal is placed at +Y offset from env_origin (far end of corridor).
        """
        if self.goal_waypoints is not None and self.num_goal_waypoints > 0:
            self.current_goal_idx[env_ids] = 0
            self._set_current_goal_positions(env_ids)
            self.goal_distance[env_ids] = torch.norm(
                self.root_states[env_ids, :2] - self.goal_positions[env_ids, :2], dim=1
            )
            self.prev_goal_distance[env_ids] = self.goal_distance[env_ids].clone()
            self.goal_reached[env_ids] = False
            return

        # Goal position: offset X/Y from origin, same Z as origin
        self.goal_positions[env_ids, 0] = self.env_origins[env_ids, 0] + self.goal_offset_x
        self.goal_positions[env_ids, 1] = self.env_origins[env_ids, 1] + self.goal_offset_y  # Forward Y
        self.goal_positions[env_ids, 2] = self.env_origins[env_ids, 2]  # Same Z
        
        # Add a small random variation only when the terrain is not a straight corridor.
        # In corridor-only training, keep the goal centered so the policy learns to move forward.
        corridor_only = getattr(self.cfg.terrain, 'corridor_only', False)
        if not corridor_only:
            self.goal_positions[env_ids, 0] += (2.0 * torch.rand(len(env_ids), device=self.device) - 1.0) * 0.5
        
        # Initialize goal distances
        self.goal_distance[env_ids] = torch.norm(
            self.root_states[env_ids, :2] - self.goal_positions[env_ids, :2], dim=1
        )
        self.prev_goal_distance[env_ids] = self.goal_distance[env_ids].clone()
        self.goal_reached[env_ids] = False

    def _set_current_goal_positions(self, env_ids):
        """Set the active waypoint target for the selected environments."""
        if self.goal_waypoints is None or self.num_goal_waypoints == 0 or len(env_ids) == 0:
            return

        waypoint_offsets = self.goal_waypoints[self.current_goal_idx[env_ids]]
        self.goal_positions[env_ids, 0] = self.env_origins[env_ids, 0] + waypoint_offsets[:, 0]
        self.goal_positions[env_ids, 1] = self.env_origins[env_ids, 1] + waypoint_offsets[:, 1]
        self.goal_positions[env_ids, 2] = self.env_origins[env_ids, 2]
    
    def _world_to_local_2d(self, vec_world):
        """Convert 2D world vectors to robot's local frame using yaw rotation."""
        # Get robot yaw from quaternion
        forward = quat_apply(self.base_quat, self.forward_vec)
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        
        # Rotate to local frame
        cos_yaw = torch.cos(-yaw)
        sin_yaw = torch.sin(-yaw)
        local_x = cos_yaw * vec_world[:, 0] - sin_yaw * vec_world[:, 1]
        local_y = sin_yaw * vec_world[:, 0] + cos_yaw * vec_world[:, 1]
        return torch.stack([local_x, local_y], dim=1)

    def _update_terrain_curriculum(self, env_ids):
        """Override terrain curriculum for goal-directed confined navigation.
        
        Criteria:
        - Move UP: reached goal OR survived >50% AND walked >2m toward goal
        - Move DOWN: survived <15% of episode (died quickly)
        """
        if not self.init_done:
            return
        
        # Compute distance walked from spawn
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        
        # Survival ratio
        survival_ratio = self.episode_length_buf[env_ids].float() / self.max_episode_length
        
        if self.goal_navigation:
            # Goal reached is the strongest signal for moving up
            reached_goal = self.goal_reached[env_ids]
            # Secondary: made significant progress toward goal (covered >60% of initial distance)
            initial_dist = self.goal_offset_y  # ~4.0m
            progress_ratio = 1.0 - self.goal_distance[env_ids] / (initial_dist + 1e-6)
            good_progress = (survival_ratio > 0.4) & (progress_ratio > 0.6)
            move_up = reached_goal | good_progress
        else:
            # Fallback: survival-based
            move_up = (survival_ratio > 0.4) & (distance > 1.0)
        
        move_down = (survival_ratio < 0.15) & (~move_up)
        
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0)
        )
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def check_termination(self):
        """Check termination conditions including collision and goal reached."""
        super().check_termination()
        
        # Get collision parameters from config
        if hasattr(self.cfg.rewards, 'collision_threshold'):
            collision_threshold = self.cfg.rewards.collision_threshold
        else:
            collision_threshold = 0.08  # Default 8cm
        
        # Get protection steps
        if hasattr(self.cfg.rewards, 'collision_termination_after_steps'):
            min_steps = self.cfg.rewards.collision_termination_after_steps
        else:
            min_steps = 10
        
        # Terminate due to collision after protection period
        collision = self.min_obstacle_dist < collision_threshold
        collision_termination = collision & (self.episode_length_buf > min_steps)
        self.reset_buf |= collision_termination
        
        # Terminate (success) when goal is reached (can be disabled for walk-only pretraining)
        terminate_on_goal_reached = getattr(self.cfg.rewards, 'terminate_on_goal_reached', True)
        if self.goal_navigation and terminate_on_goal_reached:
            self.reset_buf |= self.goal_reached

    def _draw_debug_vis(self):
        """Draw debug visualization including LiDAR points and goal markers."""
        super()._draw_debug_vis()
        
        # Draw LiDAR points
        if not self.headless and hasattr(self, 'lidar_points_buf'):
            self._draw_lidar_points()
        
        # Draw goal markers
        if not self.headless and self.goal_navigation:
            self._draw_goal_markers()
    
    def _draw_goal_markers(self):
        """Visualize goal positions as red spheres."""
        if not hasattr(self, 'viewer') or self.viewer is None:
            return
        
        # Draw goal for first few environments
        for env_idx in range(min(4, self.num_envs)):
            goal_pos = self.goal_positions[env_idx].cpu().numpy()
            sphere = gymutil.WireframeSphereGeometry(0.15, 8, 8, None, color=(1, 0, 0))
            pose = gymapi.Transform(gymapi.Vec3(goal_pos[0], goal_pos[1], goal_pos[2] + 0.3), r=None)
            gymutil.draw_lines(sphere, self.gym, self.viewer, self.envs[env_idx], pose)

    def _draw_lidar_points(self):
        """Visualize LiDAR point clouds for all (or selected number of) environments."""
        if not hasattr(self, 'viewer') or self.viewer is None:
            return

        max_envs_to_draw = int(getattr(self.cfg.viewer, 'lidar_vis_num_envs', self.num_envs))
        max_envs_to_draw = max(1, min(max_envs_to_draw, self.num_envs))
        max_points = int(getattr(self.cfg.viewer, 'lidar_vis_max_points', 180))
        near_geom = gymutil.WireframeSphereGeometry(0.012, 4, 4, None, color=(1, 0, 0))
        far_geom = gymutil.WireframeSphereGeometry(0.012, 4, 4, None, color=(0, 1, 0))
        near_threshold = 0.6

        for env_idx in range(max_envs_to_draw):
            points_local = self.lidar_points_buf[env_idx]
            dists = self.lidar_dist_buf[env_idx]
            valid_mask = (dists > self.lidar_cfg.min_range) & (dists < self.lidar_cfg.max_range)
            if not torch.any(valid_mask):
                continue

            points_local = points_local[valid_mask]
            dists = dists[valid_mask]

            if points_local.shape[0] > max_points:
                idx = torch.linspace(0, points_local.shape[0] - 1, max_points, device=self.device).long()
                points_local = points_local[idx]
                dists = dists[idx]

            sensor_pos = self.sensor_pos_tensor[env_idx]
            sensor_quat = self.sensor_quat_tensor[env_idx]
            sensor_quat_expand = sensor_quat.unsqueeze(0).expand(points_local.shape[0], -1)
            world_points = sensor_pos.unsqueeze(0) + quat_apply(sensor_quat_expand, points_local)

            for point_idx in range(world_points.shape[0]):
                pos = world_points[point_idx]
                geom = near_geom if dists[point_idx] < near_threshold else far_geom
                pose = gymapi.Transform(gymapi.Vec3(float(pos[0]), float(pos[1]), float(pos[2])), r=None)
                gymutil.draw_lines(geom, self.gym, self.viewer, self.envs[env_idx], pose)

    def create_viewer(self):
        # create viewer
        if self.headless == True:
            self.viewer = None
            print("Running in headless mode")
        else:
            self.debug_viz = True
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                print("*** Failed to create viewer")
                quit()
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT") # 按 Esc 关闭仿真窗口。
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync") # 焦点在仿真与显示之间切换
            
            self.vis = GymVisualizer(self.gym, self.sim, self.viewer, self.envs)

    # ============== Reward Functions ==============

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        self.feet_contact_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        self.feet_contact_time *= contact_filt
        return rew_airTime
 
    def _sync_reward_func(self, foot_0: int, foot_1: int, max_err=2) -> torch.Tensor:
        """Penalize desynchronization of two feet."""
        air_time = self.feet_air_time
        contact_time = self.feet_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=max_err**2)
        return se_air + se_contact
    
    def _async_reward_func(self, foot_0: int, foot_1: int, max_err=2) -> torch.Tensor:
        """Penalize synchronization of two feet."""
        air_time = self.feet_air_time
        contact_time = self.feet_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=max_err**2)
        return se_act_0 + se_act_1

    def _reward_feet_sync(self):
        """
        Penalize desynchronization within each tripod group by summing all pair-wise sync errors.
        Group 1: LF (1), LB (0), RM (5)
        Group 2: RF (4), RB (3), LM (2)
        """
        # Pairs in Group 1
        sync_g1 = self._sync_reward_func(1, 0) + self._sync_reward_func(1, 5) + self._sync_reward_func(0, 5)
        
        # Pairs in Group 2
        sync_g2 = self._sync_reward_func(4, 3) + self._sync_reward_func(4, 2) + self._sync_reward_func(3, 2)

        # Total sync penalty is the sum of penalties from both groups
        sync_reward = sync_g1 + sync_g2
        # Only apply reward when moving
        if self.cfg.commands.heading_command:
            move_condition = torch.logical_or(torch.norm(self.commands[:, :2], dim=1) > 0.1, 
                                              torch.abs(self.commands[:, 3]) > 0.1)
        else:
            move_condition = torch.logical_or(torch.norm(self.commands[:, :2], dim=1) > 0.1, 
                                              torch.abs(self.commands[:, 2]) > 0.1)
        
        return sync_reward * move_condition

    def _reward_feet_async(self):
        """
        Penalize synchronization between the two tripod groups by summing all pair-wise async errors.

        """
        async_reward = 0
        # Sum of async penalties for all pairs between Group 1 and Group 2
        for foot_g1 in self.tripod_group1_indices:
            for foot_g2 in self.tripod_group2_indices:
                async_reward += self._async_reward_func(foot_g1, foot_g2)

        # Only apply reward when moving
        if self.cfg.commands.heading_command:
            move_condition = torch.logical_or(torch.norm(self.commands[:, :2], dim=1) > 0.1, 
                                              torch.abs(self.commands[:, 3]) > 0.1)
        else:
            move_condition = torch.logical_or(torch.norm(self.commands[:, :2], dim=1) > 0.1, 
                                              torch.abs(self.commands[:, 2]) > 0.1)

        return async_reward * move_condition
    
    def _reward_contact_force_balance(self):
        """
        核心惩罚逻辑：惩罚接触力方差。
        惩罚同一三角支撑组内接触力的方差。
        这会鼓励处于同一支撑相的腿均匀地分担负载。
        """
        # 获取所有脚底的法向接触力（Z轴方向）
        normal_forces = self.contact_forces[:, self.feet_indices, 2]
        
        # 创建一个蒙版，标记哪些脚正在与地面接触（力大于1.0）
        contact_mask = (normal_forces > 1.0).float()

        # --- 处理第一组腿 ---
        # 获取第一组腿的接触力
        forces_g1 = normal_forces[:, self.tripod_group1_indices]
        # 获取第一组腿的接触蒙版
        mask_g1 = contact_mask[:, self.tripod_group1_indices]
        # 将未接触地面的腿的力置零，以便它们不影响均值和方差的计算
        masked_forces_g1 = forces_g1 * mask_g1
        # 计算第一组中接触地面的腿的数量
        num_contacting_g1 = torch.sum(mask_g1, dim=1)
        # 只有当接触地面的腿数大于1时，计算方差才有意义
        is_valid_g1 = (num_contacting_g1 > 1).float()
        # 计算接触腿的平均力（分母加一个小数防止除以零）
        mean_g1 = torch.sum(masked_forces_g1, dim=1) / (num_contacting_g1 + 1e-6)
        # 计算接触腿的力的方差： Variance = mean( (x - mean)^2 )
        variance_g1 = torch.sum(torch.square(masked_forces_g1 - mean_g1.unsqueeze(1)) * mask_g1, dim=1) / (num_contacting_g1 + 1e-6)
        
        # --- 处理第二组腿 ---
        # 获取第二组腿的接触力
        forces_g2 = normal_forces[:, self.tripod_group2_indices]
        # 获取第二组腿的接触蒙版
        mask_g2 = contact_mask[:, self.tripod_group2_indices]
        # 将未接触地面的腿的力置零
        masked_forces_g2 = forces_g2 * mask_g2
        # 计算第二组中接触地面的腿的数量
        num_contacting_g2 = torch.sum(mask_g2, dim=1)
        # 只有当接触地面的腿数大于1时，计算方差才有意义
        is_valid_g2 = (num_contacting_g2 > 1).float()
        # 计算接触腿的平均力
        mean_g2 = torch.sum(masked_forces_g2, dim=1) / (num_contacting_g2 + 1e-6)
        # 计算接触腿的力的方差
        variance_g2 = torch.sum(torch.square(masked_forces_g2 - mean_g2.unsqueeze(1)) * mask_g2, dim=1) / (num_contacting_g2 + 1e-6)

        # 总惩罚是两组方差的总和，仅在对应组有效（接触腿数>1）时计算
        total_variance_penalty = variance_g1 * is_valid_g1 + variance_g2 * is_valid_g2
        
        return total_variance_penalty

    def _reward_shank_vertical(self):
  
        # 获取小腿的刚体状态: [num_envs, num_shanks, 13]
        # 13维度: pos(3), quat(4), lin_vel(3), ang_vel(3)
        shank_states = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.shank_indices, :]
        foot_states = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, :]

        x_error = shank_states[:, :, 0] - foot_states[:, :, 0]
        y_error = shank_states[:, :, 1] - foot_states[:, :, 1]
        
        # 计算小腿方向向量在XY平面的投影长度    
        horizontal_dist = torch.sum(torch.sqrt(x_error**2 + y_error**2), dim=1)
        

        return horizontal_dist

    
    def _reward_haa_tripod_symmetry(self):
        """
        奖励三角步态的HAA对称性。
        
        要求:
        - Group 1 (LF, RM, LB): 组内HAA角度应该相同
        - Group 2 (RF, LM, RB): 组内HAA角度应该相同
        - 两组之间HAA角度应该相反
        
        DOF顺序 (18个DOF, 每条腿3个关节):
        LB: HAA(0), HFE(1), KFE(2)
        LF: HAA(3), HFE(4), KFE(5)
        LM: HAA(6), HFE(7), KFE(8)
        RB: HAA(9), HFE(10), KFE(11)
        RF: HAA(12), HFE(13), KFE(14)
        RM: HAA(15), HFE(16), KFE(17)
        
        Returns:
            torch.Tensor: 惩罚值,当HAA不对称时返回正值
        """
        # 获取各腿的HAA角度
        LB_HAA = self.dof_pos[:, 0]   # LB
        LF_HAA = self.dof_pos[:, 3]   # LF
        LM_HAA = self.dof_pos[:, 6]   # LM
        RB_HAA = self.dof_pos[:, 9]   # RB
        RF_HAA = self.dof_pos[:, 12]  # RF
        RM_HAA = self.dof_pos[:, 15]  # RM
        
        # Group 1 (LF, RM, LB): 组内应该角度相同
        # 计算组内两两之间的角度差的平方
        g1_penalty = torch.square(LF_HAA - RM_HAA) + \
                     torch.square(LF_HAA - LB_HAA) + \
                     torch.square(RM_HAA - LB_HAA)
        
        # Group 2 (RF, LM, RB): 组内应该角度相同
        g2_penalty = torch.square(RF_HAA - LM_HAA) + \
                     torch.square(RF_HAA - RB_HAA) + \
                     torch.square(LM_HAA - RB_HAA)
        
        # 两组之间应该相反: G1的平均值 + G2的平均值 ≈ 0
        # 计算两组的平均HAA角度
        g1_mean = (LF_HAA + RM_HAA + LB_HAA) / 3.0
        g2_mean = (RF_HAA + LM_HAA + RB_HAA) / 3.0
        
        # 两组平均值应该相反(和接近0)
        inter_group_penalty = torch.square(g1_mean + g2_mean)
        
        # 总惩罚
        total_penalty = g1_penalty + g2_penalty + inter_group_penalty
        
        return total_penalty
    


    def _reward_stand_on_six_legs(self):
        # 低命令下：鼓励六条腿全部着地

        lin_cmd_small = torch.norm(self.commands[:, :2], dim=1) < self.speed_min
        if self.cfg.commands.heading_command:
            yaw_or_heading_small = torch.abs(self.commands[:, 3]) < self.speed_min
        else:
            yaw_or_heading_small = torch.abs(self.commands[:, 2]) < self.speed_min
        small_command_mask = torch.logical_and(lin_cmd_small, yaw_or_heading_small)

        # 足端接触（法向力阈值 1N）
        foot_contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        num_feet_in_contact = torch.sum(foot_contact.float(), dim=1)

        # 惩罚未着地的脚数量；六足都着地时为 0
        missing_contact_penalty = len(self.feet_indices) - num_feet_in_contact

        return missing_contact_penalty * small_command_mask.float()


    # ============== 原ElSpiderLidar奖励 ==============

    def _reward_obstacle_avoidance(self):
        """Reward for maintaining safe distance from obstacles."""
        # Reward increases with distance from obstacles
        safe_dist = getattr(self.cfg.rewards, 'safe_obstacle_dist', 0.5)
        
        # Compute reward based on minimum distance
        dist_reward = torch.clamp(self.min_obstacle_dist / safe_dist, 0, 1)
        return dist_reward

    def _reward_collision_penalty(self):
        """Penalty for getting too close to obstacles.
        Returns positive value (higher = closer to obstacle).
        Should be used with NEGATIVE reward scale.
        """
        danger_dist = getattr(self.cfg.rewards, 'danger_obstacle_dist', 0.3)
        
        # Exponential penalty for being too close
        # Returns positive value: large when close, ~0 when far
        penalty = torch.exp(-self.min_obstacle_dist / danger_dist + 1) - 1
        penalty = torch.clamp(penalty, 0, 10)
        return penalty

    def _reward_exploration(self):
        """Reward for exploring (moving) while avoiding obstacles."""
        # Reward any movement (not just forward) when safe from obstacles
        move_speed = torch.norm(self.base_lin_vel[:, :2], dim=1)
        safe_dist = getattr(self.cfg.rewards, 'safe_obstacle_dist', 0.5)
        
        # Scale movement reward by safety — encourage moving only when safe
        safety_factor = torch.clamp(self.min_obstacle_dist / safe_dist, 0, 1)
        
        # Also consider command tracking: reward moving in commanded direction
        cmd_speed = torch.norm(self.commands[:, :2], dim=1)
        has_command = (cmd_speed > 0.1).float()
        exploration_reward = move_speed * safety_factor * has_command
        return torch.clamp(exploration_reward, 0, 1)

    def _reward_goal_reaching(self):
        """Reward for moving toward goal: velocity projected onto goal direction.
        This directly rewards the component of robot velocity pointing toward the goal.
        
        IMPORTANT: goal_dir is in WORLD frame, so we must use WORLD-frame velocity
        (root_states[:, 7:9]) NOT local-frame base_lin_vel.
        """
        if not self.goal_navigation:
            return torch.zeros(self.num_envs, device=self.device)
        
        # Direction to goal (unit vector in XY plane, WORLD frame)
        goal_vec = self.goal_positions[:, :2] - self.root_states[:, :2]
        goal_dir = goal_vec / (torch.norm(goal_vec, dim=1, keepdim=True) + 1e-6)
        
        # Robot velocity in WORLD frame (root_states[:, 7:9] = world linear vel XY)
        world_vel_xy = self.root_states[:, 7:9]
        
        # Project world-frame velocity onto world-frame goal direction
        # Positive = moving toward goal, negative = moving away
        vel_toward_goal = torch.sum(world_vel_xy * goal_dir, dim=1)
        
        # Reward: proportional to velocity toward goal, capped at command speed
        reward = torch.clamp(vel_toward_goal, -0.5, 1.5)
        
        # Bonus multiplier when close to goal (last 2m): encourage final approach
        close_bonus = torch.where(
            self.goal_distance < 2.0,
            1.0 + (2.0 - self.goal_distance) * 0.5,  # up to 2x reward when very close
            torch.ones_like(self.goal_distance)
        )
        
        return reward * close_bonus

    def _reward_goal_progress(self):
        """Dense reward for reducing distance to goal each step.
        Positive when approaching goal, negative when moving away.
        """
        if not self.goal_navigation:
            return torch.zeros(self.num_envs, device=self.device)

        progress = self.prev_goal_distance - self.goal_distance
        return torch.clamp(progress, -0.05, 0.10)
    
    def _reward_goal_bonus(self):
        """Large bonus reward for reaching the goal."""
        if not self.goal_navigation:
            return torch.zeros(self.num_envs, device=self.device)
        
        return self.goal_reached.float()

    def _reward_corridor_centering(self):
        """Reward staying near the corridor centerline during confined navigation."""
        if not self.goal_navigation:
            return torch.zeros(self.num_envs, device=self.device)

        corridor_width = getattr(self.cfg.terrain, 'corridor_width_override', 1.6)
        half_width = max(float(corridor_width) * 0.5, 1e-3)
        center_offset = torch.abs(self.root_states[:, 0] - self.env_origins[:, 0])

        centering_reward = 1.0 - center_offset / half_width
        return torch.clamp(centering_reward, 0.0, 1.0)

    def _reward_obstacle_maneuvering(self):
        """Reward active maneuvering when obstacles are near.

        This encourages the policy to turn or sidestep instead of only pushing
        forward when the corridor is blocked.
        """
        if not self.goal_navigation:
            return torch.zeros(self.num_envs, device=self.device)

        safe_dist = getattr(self.cfg.rewards, 'safe_obstacle_dist', 0.5)
        danger_dist = getattr(self.cfg.rewards, 'danger_obstacle_dist', 0.15)

        # 0 when far from obstacles, 1 when within the danger zone.
        obstacle_pressure = torch.clamp(
            (safe_dist - self.min_obstacle_dist) / (safe_dist - danger_dist + 1e-6),
            0.0,
            1.0,
        )

        # Encourage turning and small lateral motion around blocked obstacles.
        yaw_activity = torch.abs(self.base_ang_vel[:, 2])
        lateral_activity = torch.abs(self.base_lin_vel[:, 1])
        maneuver_activity = 0.6 * yaw_activity + 0.4 * lateral_activity

        return torch.clamp(maneuver_activity * obstacle_pressure, 0.0, 1.0)

    def _reward_retreat(self):
        """Reward backing up when the front sector is blocked.

        This helps the policy learn to actively step away from a small pillar
        instead of pausing in place or trying to climb over it.
        """
        if not self.goal_navigation:
            return torch.zeros(self.num_envs, device=self.device)

        safe_dist = getattr(self.cfg.rewards, 'safe_obstacle_dist', 0.5)
        danger_dist = getattr(self.cfg.rewards, 'danger_obstacle_dist', 0.15)
        front_pressure = torch.clamp(
            (safe_dist - self.front_obstacle_dist) / (safe_dist - danger_dist + 1e-6),
            0.0,
            1.0,
        )

        # local x velocity < 0 means moving backward in the robot frame
        retreat_speed = torch.clamp(-self.base_lin_vel[:, 0], 0.0, 0.25)
        return torch.clamp(retreat_speed * front_pressure, 0.0, 1.0)
    
    def _reward_goal_heading(self):
        """Reward for facing toward goal AND moving forward.
        Only rewards heading alignment when the robot is actually walking.
        Prevents 'face goal and stand still' local optimum.
        """
        if not self.goal_navigation:
            return torch.zeros(self.num_envs, device=self.device)
        
        # Compute angle between robot forward and goal direction
        goal_vec = self.goal_positions[:, :2] - self.root_states[:, :2]
        goal_dir = goal_vec / (torch.norm(goal_vec, dim=1, keepdim=True) + 1e-6)
        
        forward = quat_apply(self.base_quat, self.forward_vec)
        forward_2d = forward[:, :2]
        forward_2d = forward_2d / (torch.norm(forward_2d, dim=1, keepdim=True) + 1e-6)
        
        # Dot product: 1 when facing goal, -1 when facing away
        cos_angle = torch.sum(forward_2d * goal_dir, dim=1)
        heading_reward = torch.clamp((cos_angle + 1.0) / 2.0, 0, 1)
        
        # Gate by movement speed: only reward heading when robot is walking
        # Use WORLD-frame velocity for consistency (root_states[:, 7:9])
        move_speed = torch.norm(self.root_states[:, 7:9], dim=1)
        moving_mask = torch.clamp(move_speed / 0.3, 0, 1)  # ramp: 0 at 0m/s, 1 at 0.3m/s+
        
        return heading_reward * moving_mask    



    
    

# Bodies:  ['base_link', 'LB_HIP', 'LB_THIGH', 'LB_SHANK', 'LB_FOOT', 'LF_HIP', 'LF_THIGH', 'LF_SHANK', 'LF_FOOT', 
# 'LM_HIP', 'LM_THIGH', 'LM_SHANK', 'LM_FOOT', 'RB_HIP', 'RB_THIGH', 'RB_SHANK', 'RB_FOOT', 
# 'RF_HIP', 'RF_THIGH', 'RF_SHANK', 'RF_FOOT', 'RM_HIP', 'RM_THIGH', 'RM_SHANK', 'RM_FOOT']