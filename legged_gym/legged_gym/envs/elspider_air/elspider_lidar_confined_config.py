# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# Extended for LiDAR-based confined space navigation

"""
Configuration for ElSpider LiDAR Confined Space Navigation Task
基于激光雷达的六足机器人受限空间避障运动控制配置
"""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from dataclasses import dataclass, field

SAME_DIM_POLICY_HIDDEN_DIMS = [128, 64, 32]
SAME_DIM_INIT_NOISE_STD = 0.35


class ElSpiderLidarConfinedCfg(LeggedRobotCfg):
    """Configuration for ElSpider with LiDAR in confined spaces."""
    
    class env(LeggedRobotCfg.env):
        num_envs = 512  # Reduced to fit GPU memory for confined terrain initialization
        num_actions = 18  # 6 legs × 3 joints
        
        # Base observations: 3+3+3+3+18+18+18 = 66
        # LiDAR observations: 12×8 = 96
        # Goal observations: 2 (direction_angle, normalized_distance)
        num_lidar_obs = 96  # num_theta_bins × num_phi_bins
        num_goal_obs = 2    # goal direction angle + normalized distance
        num_observations = 66 + 96 + 2  # 164 total
        
        episode_length_s = 24  # Longer episode to reach goal

    class sim(LeggedRobotCfg.sim):
        class physx(LeggedRobotCfg.sim.physx):
            max_gpu_contact_pairs = 2**24
            default_buffer_size_multiplier = 8
            
    class lidar:
        """LiDAR sensor configuration."""
        sensor_type = "simple_grid"  # Options: simple_grid, avia, mid360, etc.
        
        # Sensor update frequency
        update_frequency = 20.0  # Hz
        
        # Range settings
        max_range = 5.0  # meters
        min_range = 0.1  # meters
        
        # Grid LiDAR settings
        horizontal_line_num = 48  # More horizontal rays to catch thin columns
        vertical_line_num = 12   # More vertical rays to catch low obstacles
        horizontal_fov_deg_min = -180  # Horizontal FOV min (degrees)
        horizontal_fov_deg_max = 180   # Horizontal FOV max (degrees)
        vertical_fov_deg_min = -30     # Vertical FOV min (degrees)
        vertical_fov_deg_max = 10      # Vertical FOV max (degrees)
        
        # Observation downsampling
        num_theta_bins = 12  # Azimuth bins for observation
        num_phi_bins = 8     # Elevation bins for observation
        
        # Sensor mounting position (relative to robot base frame)
        sensor_offset = [0.3, 0.0, 0.35]  # [x, y, z] in meters
        sensor_rotation_deg = [3.14, 0.0, 0.0]  # [roll, pitch, yaw] in degrees

    class terrain(LeggedRobotCfg.terrain):
        """Terrain configuration for confined spaces."""
        mesh_type = 'confined_trimesh'  # Use confined terrain with ceiling
        
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        
        # Height measurement settings
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1,
                           0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Confined terrain settings
        terrain_length = 10.0  # Corridor length
        terrain_width = 10.0
        num_rows = 8   # More difficulty levels for corridor progression
        num_cols = 4   # Terrain type columns
        
        # Final confined task: pillar/column pile only.
        # Type order in TerrainConfined: [corridor, timber, column, maze, barrier, gap, corridor-fallback]
        confined_terrain_proportions = [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00]
        
        # Spawn area size for robot placement
        spawn_area_size = 1.0  # Smaller central free area to reduce local hovering near center
        spawn_area_flat = True
        
        # Pillar pile difficulty: passable gaps for ElSpider while requiring active routing.
        difficulty_scale = 0.16
        corridor_only = False
        column_spacing_override = 0.95
        column_density_override = 0.22
        column_radius_override = 0.09
        column_height_override = 300.0
        hanging_length_override = 0.10
        
        slope_treshold = 0.75  # Slopes above this threshold will be corrected
        
        # Goal navigation settings
        goal_navigation = True  # Enable start→goal navigation mode
        goal_offset_y = 4.8     # Put goal near the far end without hitting the boundary

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.4]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "RF_HAA": 0.0,
            "RM_HAA": 0.0,
            "RB_HAA": 0.0,
            "LF_HAA": 0.0,
            "LM_HAA": 0.0,
            "LB_HAA": 0.0,

            "RF_HFE": 0.2,
            "RM_HFE": 0.2,
            "RB_HFE": 0.2,
            "LF_HFE": 0.2,
            "LM_HFE": 0.2,
            "LB_HFE": 0.2,

            "RF_KFE": 0.3,
            "RM_KFE": 0.3,
            "RB_KFE": 0.3,
            "LF_KFE": 0.3,
            "LM_KFE": 0.3,
            "LB_KFE": 0.3,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters
        stiffness = {'HAA': 60., 'HFE': 60., 'KFE': 60.}  # [N*m/rad]
        damping = {'HAA': 0.8, 'HFE': 0.8, 'KFE': 0.8}    # [N*m*s/rad]

        action_scale = 0.25
        decimation = 4

        use_actuator_network = False
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/el_mini/urdf/el_mini.urdf"
        name = "elspider_air"
        foot_name = "FOOT"
        penalize_contacts_on = ["THIGH", "HIP"]
        terminate_after_contacts_on = ["trunk"]
        self_collisions = 0  # 1 to disable, 0 to enable
        flip_visual_attachments = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.3, 1.25]
        randomize_base_mass = True
        added_mass_range = [-5., 5.]
        push_robots = True
        push_interval_s = 3
        max_push_vel_xy = 1.0

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0
        
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            dof_pos = 0.01
            dof_vel = 1.5
            height_measurements = 0.1
            lidar = 0.05  # LiDAR observation noise

    class rewards(LeggedRobotCfg.rewards):
        base_height_target = 0.29
        max_contact_force = 500.
        only_positive_rewards = False  # Allow negative rewards for collision
        
        # Obstacle avoidance parameters
        safe_obstacle_dist = 0.60    # Keep a conservative but feasible clearance in pillar piles
        danger_obstacle_dist = 0.18  # Start danger penalty close to true collision envelope
        collision_threshold = 0.03  # REDUCED from 0.05: only terminate on actual collision (3cm)
        
        # Termination protection - generous grace period
        collision_termination_after_steps = 200  # INCREASED from 50: let robot survive much longer
        allow_initial_contact_steps = 30  # Grace period at episode start
        
        # Multi-stage rewards disabled
        multi_stage_rewards = False
        reward_stage_threshold = 80.0
        reward_min_stage = 0
        reward_max_stage = 2
        
        # Goal navigation reward parameters
        goal_reach_threshold = 0.9    # Tighten goal reach criterion for the final task
        goal_max_distance = 6.5       # Max expected distance to goal [meters] (for normalization)

        class scales(LeggedRobotCfg.rewards.scales):
            # Standard locomotion rewards
            termination = -2.0         # Penalize episode termination
            tracking_lin_vel = 1.2     # Align with learned walk-flat gait
            tracking_ang_vel = 0.5
            lin_vel_z = -3.0
            ang_vel_xy = -0.2
            orientation = -1.8
            torques = -0.0001
            dof_vel = -0.00045
            dof_acc = -5e-8
            base_height = -0.8
            feet_air_time = 0.35
            collision = -1.0
            feet_stumble = -0.0
            action_rate = -0.006
            stand_still = -0.1
            dof_pos_limits = -1.0
            feet_slip = -0.0
            feet_contact_forces = -0.2
            shank_perp2ground = -0.12
            
            # Confined space specific rewards
            obstacle_avoidance = 0.28   # Keep obstacle margin without dominating gait
            collision_penalty = -0.20   # Penalize danger but avoid over-conservative policy
            corridor_centering = 0.10   # Reduced because this task is no longer corridor-only
            exploration = 0.0           # Disabled: goal system handles movement

            # Active obstacle negotiation rewards
            obstacle_maneuvering = 0.12
            retreat = 0.05
            
            # Goal-directed navigation rewards
            goal_reaching = 6.5         # Balance objective against gait preservation
            goal_progress = 4.5         # Increase dense forward incentive
            goal_bonus = 16.0           # Terminal objective remains meaningful
            goal_heading = 0.8          # Heading guidance, gated by movement speed
            
            # Gait rewards
            gait_2_step = -0.25

        class async_gait_scheduler:
            dof_align = 0.5
            dof_nominal_pos = [0.1, 0.2]
            reward_foot_z_align = [0.2, 0.05]

    class commands(LeggedRobotCfg.commands):
        curriculum = True  # Enable: start with slow commands, increase over time
        max_curriculum = 1.0
        num_commands = 4
        resampling_time = 10.0  # Longer resampling: goal provides consistent direction
        heading_command = True  # Will be overridden to use goal heading
        goal_directed = True    # Use goal position to generate heading commands
        
        class ranges:
            lin_vel_x = [0.15, 0.9]   # Align with walk/nav flat command envelope
            lin_vel_y = [-0.25, 0.25]
            ang_vel_yaw = [-0.7, 0.7]
            heading = [-3.14, 3.14]   # Will be overridden by goal heading


class ElSpiderLidarConfinedCfgPPO(LeggedRobotCfgPPO):
    """PPO training configuration for ElSpider LiDAR confined space task."""
    
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.005         # Slightly more exploration for robust obstacle handling
        learning_rate = 7e-4          # Smoother policy updates for gait stability
        num_learning_epochs = 5
        gamma = 0.99
        lam = 0.95
        num_mini_batches = 4
        desired_kl = 0.008            # Tighter updates for stability
        schedule = 'adaptive'         # Use adaptive LR schedule based on KL divergence

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.25
        actor_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        critic_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        activation = 'elu'

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'elspider_lidar_confined'
        load_run = -1
        max_iterations = 22600  # Extended: resuming from checkpoint with boosted rewards
        
        # Multi-stage rewards disabled (env config controls this)
        multi_stage_rewards = False
        
        # Checkpointing
        save_interval = 100
        
        # Logging
        log_interval = 10


# Alternative configuration with simpler LiDAR for faster training
class ElSpiderLidarConfinedSimpleCfg(ElSpiderLidarConfinedCfg):
    """Simplified configuration with reduced LiDAR resolution."""
    
    class env(ElSpiderLidarConfinedCfg.env):
        # Reduced LiDAR observations: 8×6 = 48
        num_lidar_obs = 48
        num_observations = 66 + 48 + 2  # 116 total

    class lidar(ElSpiderLidarConfinedCfg.lidar):
        # Fewer rays for faster computation
        horizontal_line_num = 24
        vertical_line_num = 6
        
        # Smaller observation bins
        num_theta_bins = 8
        num_phi_bins = 6


class ElSpiderLidarConfinedSimpleCfgPPO(ElSpiderLidarConfinedCfgPPO):
    """PPO config for simplified LiDAR task."""
    
    class runner(ElSpiderLidarConfinedCfgPPO.runner):
        experiment_name = 'elspider_lidar_confined_simple'


# Configuration for timber pile terrain only
class ElSpiderLidarTimberPileCfg(ElSpiderLidarConfinedCfg):
    """Configuration specifically for timber pile terrain."""
    
    class terrain(ElSpiderLidarConfinedCfg.terrain):
        # Only timber pile terrain: [corridor, timber, column, maze, barrier, gap, corridor-fallback]
        confined_terrain_proportions = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        difficulty_scale = 0.8  # Slightly easier
        timber_spacing_override = 1.35  # Increase spacing so the robot has more room to pass
        goal_offset_x = 1.5   # Add lateral offset so the policy must learn to turn
        goal_offset_y = 5.0   # Push the goal farther down the terrain

    class commands(ElSpiderLidarConfinedCfg.commands):
        class ranges(ElSpiderLidarConfinedCfg.commands.ranges):
            lin_vel_x = [0.10, 0.65]   # Keep forward motion modest to preserve walk-flat posture
            lin_vel_y = [-0.25, 0.25]   # Sideways motion stays available, but not excessive
            ang_vel_yaw = [-0.60, 0.60] # Enough turning to route around timber without twisting too hard

    class rewards(ElSpiderLidarConfinedCfg.rewards):
        max_contact_force = 500.
        base_height_target = 0.34
        only_positive_rewards = False
        tracking_sigma = 0.25

        class scales(ElSpiderLidarConfinedCfg.rewards.scales):
            tracking_lin_vel = 3.0     # Align with elair_nav_timberpile posture/locomotion balance
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -5.0
            torques = -0.00001
            dof_vel = -0.0
            dof_acc = -0.5e-8
            base_height = 0.0
            base_foot_height = -8.0
            feet_slip = -0.4
            feet_air_time = 0.8
            collision = -0.5
            feet_stumble = -0.4
            feet_stumble_liftup = 1.0
            action_rate = -0.001
            stand_still = -0.0
            dof_pos_limits = -1.0
            gait_2_step = -1.0

            obstacle_avoidance = 0.30  # Reward clearance, but not so much that gait gets distorted
            collision_penalty = -0.80  # Penalize close passes while keeping motion fluid
            corridor_centering = 0.12  # Mild centering helps stable routing through the pile

            goal_heading = 0.40        # Keep goal alignment, but avoid over-forcing turns
            goal_progress = 2.5        # Lower dense forward incentive to preserve gait
            goal_reaching = 4.2        # Keep the goal attractive, but secondary to stable walking
            goal_bonus = 9.0           # Enough terminal incentive without greedy posture changes
            collision = -4.0           # Keep hard-contact cost high


class ElSpiderLidarTimberPileCfgPPO(ElSpiderLidarConfinedCfgPPO):
    """PPO config for timber pile task."""

    class policy(ElSpiderLidarConfinedCfgPPO.policy):
        actor_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        critic_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        activation = 'elu'
        init_noise_std = 0.18

    class algorithm(ElSpiderLidarConfinedCfgPPO.algorithm):
        entropy_coef = 0.002
        learning_rate = 5e-4
        desired_kl = 0.006
    
    class runner(ElSpiderLidarConfinedCfgPPO.runner):
        experiment_name = 'elspider_lidar_timber_pile'


# Configuration for tunnel terrain only
class ElSpiderLidarTunnelCfg(ElSpiderLidarConfinedCfg):
    """Configuration specifically for tunnel terrain."""
    
    class terrain(ElSpiderLidarConfinedCfg.terrain):
        # Only tunnel terrain
        confined_terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class ElSpiderLidarTunnelCfgPPO(ElSpiderLidarConfinedCfgPPO):
    """PPO config for tunnel task."""
    
    class runner(ElSpiderLidarConfinedCfgPPO.runner):
        experiment_name = 'elspider_lidar_tunnel'


class ElSpiderLidarCaveCfg(ElSpiderLidarConfinedCfg):
    """LiDAR cave task using mesh terrain from cave.obj."""

    class terrain(ElSpiderLidarConfinedCfg.terrain):
        mesh_type = 'trimesh'
        use_terrain_obj = True
        terrain_file = 'resources/terrains/confined/cave.obj'

        curriculum = False
        max_init_terrain_level = 0
        num_rows = 1
        num_cols = 1

        # Spawn inside the cave bounds, but near the floor and away from walls.
        terrain_length = 8.0
        terrain_width = 8.0
        random_origins = False
        origin_generation_max_attempts = 10000
        spawn_origin_x = 5.5
        spawn_origin_y = -5.0
        # spawn_origin_x = 0.5
        # spawn_origin_y = -5.0
        spawn_height_offset = 0.45
        spawn_height_clearance = 0.18

        # Not used by trimesh+TerrainObj, but left explicit for readability.
        confined_terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        goal_navigation = True
        # Put the goal farther along the cave and slightly offset in Y so the policy must turn.
        goal_offset_x = -18.0
        goal_offset_y = -11.5
        goal_waypoints = [
            # (2.0, -1.5),
            # (-1.0, -4.0),
            (-4.5, -7.0),
            # (-10.0, -11.0),
            (-18.0, -11.5),
        ]

    class commands(ElSpiderLidarConfinedCfg.commands):
        class ranges(ElSpiderLidarConfinedCfg.commands.ranges):
            lin_vel_x = [0.0, 0.28]
            lin_vel_y = [-0.45, 0.45]
            ang_vel_yaw = [-1.4, 1.4]

    class rewards(ElSpiderLidarConfinedCfg.rewards):
        base_height_target = 0.30
        safe_obstacle_dist = 0.85
        danger_obstacle_dist = 0.30
        collision_threshold = 0.045
        collision_termination_after_steps = 8
        waypoint_reach_threshold = 0.85
        goal_max_distance = 9.5

        class scales(ElSpiderLidarConfinedCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            obstacle_avoidance = 0.55
            obstacle_maneuvering = 0.90
            retreat = 0.45
            corridor_centering = 0.10
            collision_penalty = -2.0
            goal_heading = 0.80
            goal_progress = 1.4
            goal_reaching = 3.8
            goal_bonus = 8.0
            collision = -10.0


class ElSpiderLidarCaveCfgPPO(ElSpiderLidarConfinedCfgPPO):
    """PPO config for cave task."""

    class policy(ElSpiderLidarConfinedCfgPPO.policy):
        actor_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        critic_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        activation = 'elu'
        init_noise_std = 0.20

    class algorithm(ElSpiderLidarConfinedCfgPPO.algorithm):
        entropy_coef = 0.002
        learning_rate = 5e-4
        desired_kl = 0.007

    class runner(ElSpiderLidarConfinedCfgPPO.runner):
        experiment_name = 'elspider_lidar_cave'


class ElSpiderLidarFlatPretrainCfg(ElSpiderLidarConfinedCfg):
    """Flat pretraining task with exactly the same obs/action dimensions as confined task.

    This task is used for stage-1 pretraining, then weights can be resumed on
    `elspider_lidar_confined` directly without network shape mismatch.
    """

    class terrain(ElSpiderLidarConfinedCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        measure_heights = False
        goal_navigation = False
        goal_offset_y = 2.0

    class commands(ElSpiderLidarConfinedCfg.commands):
        class ranges(ElSpiderLidarConfinedCfg.commands.ranges):
            lin_vel_x = [0.0, 0.8]
            lin_vel_y = [-0.2, 0.2]
            ang_vel_yaw = [-0.8, 0.8]

    class domain_rand(ElSpiderLidarConfinedCfg.domain_rand):
        randomize_friction = False
        randomize_base_mass = False
        push_robots = False

    class rewards(ElSpiderLidarConfinedCfg.rewards):
        class scales(ElSpiderLidarConfinedCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            orientation = -0.25
            base_height = -1.0
            action_rate = -0.01
            feet_slip = -0.15
            collision = -0.5
            obstacle_avoidance = 0.0
            collision_penalty = 0.0
            goal_reaching = 10.0
            goal_progress = 6.0
            goal_bonus = 25.0
            stand_still = -0.35


class ElSpiderLidarFlatPretrainCfgPPO(ElSpiderLidarConfinedCfgPPO):
    """PPO config for flat pretraining with same network structure."""

    class algorithm(ElSpiderLidarConfinedCfgPPO.algorithm):
        entropy_coef = 0.0003
        desired_kl = 0.008

    class runner(ElSpiderLidarConfinedCfgPPO.runner):
        experiment_name = 'elspider_lidar_flat_pretrain'
        max_iterations = 8000


class ElSpiderLidarPoseAdaptSameDimCfg(ElSpiderLidarConfinedCfg):
    """Stage-A: pose/posture adaptation style task (same obs/action dimensions)."""

    class terrain(ElSpiderLidarConfinedCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        measure_heights = False
        goal_navigation = False

    class commands(ElSpiderLidarConfinedCfg.commands):
        goal_directed = False
        heading_command = False
        curriculum = False
        resampling_time = 4.0

        class ranges(ElSpiderLidarConfinedCfg.commands.ranges):
            lin_vel_x = [-0.3, 0.3]
            lin_vel_y = [-0.2, 0.2]
            ang_vel_yaw = [-0.3, 0.3]

    class domain_rand(ElSpiderLidarConfinedCfg.domain_rand):
        randomize_friction = False
        randomize_base_mass = False
        push_robots = False

    class rewards(ElSpiderLidarConfinedCfg.rewards):
        base_height_target = 0.34
        only_positive_rewards = True
        goal_reach_threshold = 0.20
        terminate_on_goal_reached = False

        class scales(ElSpiderLidarConfinedCfg.rewards.scales):
            tracking_lin_vel = 0.4
            tracking_ang_vel = 0.2
            orientation = -6.0
            base_height = -10.0
            action_rate = -0.002
            stand_still = -0.05
            collision = -0.3
            gait_2_step = -2.0
            obstacle_avoidance = 0.0
            collision_penalty = 0.0
            goal_reaching = 0.0
            goal_progress = 0.0
            goal_bonus = 0.0
            goal_heading = 0.0


class ElSpiderLidarPoseAdaptSameDimCfgPPO(ElSpiderLidarConfinedCfgPPO):
    """PPO config for pose/posture adaptation style stage."""

    class policy(ElSpiderLidarConfinedCfgPPO.policy):
        actor_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        critic_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        activation = 'elu'
        init_noise_std = 0.25

    class algorithm(ElSpiderLidarConfinedCfgPPO.algorithm):
        entropy_coef = 0.002
        desired_kl = 0.01

    class runner(ElSpiderLidarConfinedCfgPPO.runner):
        experiment_name = 'elspider_lidar_pose_adapt_same_dim'
        max_iterations = 1500


class ElSpiderLidarFlatSkillSameDimCfg(ElSpiderLidarConfinedCfg):
    """Stage-B: flat locomotion style task (same obs/action dimensions)."""

    class terrain(ElSpiderLidarConfinedCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        measure_heights = False
        goal_navigation = False

    class commands(ElSpiderLidarConfinedCfg.commands):
        goal_directed = False
        heading_command = False
        curriculum = False
        max_curriculum = 1.0
        resampling_time = 4.0

        class ranges(ElSpiderLidarConfinedCfg.commands.ranges):
            lin_vel_x = [0.15, 0.9]
            lin_vel_y = [-0.25, 0.25]
            ang_vel_yaw = [-0.7, 0.7]

    class domain_rand(ElSpiderLidarConfinedCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.3, 1.25]
        randomize_base_mass = False
        added_mass_range = [-5., 5.]
        push_robots = False
        push_interval_s = 3
        max_push_vel_xy = 1.

    class control(ElSpiderLidarConfinedCfg.control):
        stiffness = {'HAA': 60., 'HFE': 60., 'KFE': 60.}
        damping = {'HAA': 0.8, 'HFE': 0.8, 'KFE': 0.8}
        action_scale = 0.25
        decimation = 4
        use_actuator_network = False
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"

    class rewards(ElSpiderLidarConfinedCfg.rewards):
        base_height_target = 0.24
        only_positive_rewards = False
        multi_stage_rewards = True
        reward_stage_threshold = 2.0
        reward_min_stage = 0
        reward_max_stage = 1

        class scales(ElSpiderLidarConfinedCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -3.0
            ang_vel_xy = -0.2
            orientation = [-5.0, -3.0]
            torques = -0.0001
            dof_vel = [-0.0002, -0.0004]
            dof_acc = [-5e-8, -1.5e-7]
            base_height = [-2.0, -0.4]
            feet_slip = [-0.0, -0.2]
            feet_air_time = [0.5, 0.1]
            gait_2_step = [-1.0, -0.0]
            collision = -1.0
            feet_stumble = [-0.0, -0.2]
            action_rate = [-0.005, -0.005]
            stand_still = 0.0
            stand_still2 = -0.6
            dof_pos_limits = -1.0
            feet_contact_forces = [-0.2, -0.5]
            shank_perp2ground = -0.05
            obstacle_avoidance = 0.0
            collision_penalty = 0.0
            goal_reaching = 0.0
            goal_progress = 0.0
            goal_bonus = 0.0
            goal_heading = 0.0
            async_gait_scheduler = [-0.4, -0.4, -0.4]


class ElSpiderLidarFlatSkillSameDimCfgPPO(ElSpiderLidarConfinedCfgPPO):
    """PPO config for flat locomotion style stage."""

    class policy(ElSpiderLidarConfinedCfgPPO.policy):
        actor_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        critic_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        activation = 'elu'
        init_noise_std = 0.25

    class algorithm(ElSpiderLidarConfinedCfgPPO.algorithm):
        entropy_coef = 0.005
        desired_kl = 0.01

    class runner(ElSpiderLidarConfinedCfgPPO.runner):
        experiment_name = 'elspider_lidar_flat_same_dim'
        max_iterations = 3000
        multi_stage_rewards = True


class ElSpiderLidarMixedTerrainSameDimCfg(ElSpiderLidarConfinedCfg):
    """Stage-C: mixed terrain style task (same obs/action dimensions)."""

    class terrain(ElSpiderLidarConfinedCfg.terrain):
        mesh_type = 'trimesh'
        curriculum = True
        measure_heights = False
        goal_navigation = False
        max_init_terrain_level = 0
        terrain_length = 4.0
        terrain_width = 4.0
        num_rows = 4
        num_cols = 4
        terrain_proportions = [0.1, 0.1, 0.3, 0.3, 0.2]

    class commands(ElSpiderLidarConfinedCfg.commands):
        goal_directed = False
        heading_command = False
        curriculum = True
        max_curriculum = 1.0
        resampling_time = 4.0

        class ranges(ElSpiderLidarConfinedCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.4, 0.4]
            ang_vel_yaw = [-0.6, 0.6]

    class domain_rand(ElSpiderLidarConfinedCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.5]
        randomize_base_mass = True
        added_mass_range = [-5.0, 5.0]
        push_robots = False

    class rewards(ElSpiderLidarConfinedCfg.rewards):
        base_height_target = 0.34
        only_positive_rewards = True
        multi_stage_rewards = True
        reward_stage_threshold = 6.0
        reward_min_stage = 0
        reward_max_stage = 1

        class scales(ElSpiderLidarConfinedCfg.rewards.scales):
            termination = -5.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -5.0
            torques = -0.00001
            dof_acc = -5e-8
            base_height = -8.0
            feet_slip = [-0.0, -0.4]
            feet_air_time = 0.8
            gait_2_step = -5.0
            collision = -1.0
            action_rate = -0.001
            stand_still = -0.05
            dof_pos_limits = -1.0
            obstacle_avoidance = 0.0
            collision_penalty = 0.0
            goal_reaching = 0.0
            goal_progress = 0.0
            goal_bonus = 0.0
            goal_heading = 0.0


class ElSpiderLidarMixedTerrainSameDimCfgPPO(ElSpiderLidarConfinedCfgPPO):
    """PPO config for mixed terrain style stage."""

    class policy(ElSpiderLidarConfinedCfgPPO.policy):
        actor_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        critic_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        activation = 'elu'
        init_noise_std = SAME_DIM_INIT_NOISE_STD

    class algorithm(ElSpiderLidarConfinedCfgPPO.algorithm):
        entropy_coef = 0.004
        desired_kl = 0.01

    class runner(ElSpiderLidarConfinedCfgPPO.runner):
        experiment_name = 'elspider_lidar_mixed_terrains_same_dim'
        max_iterations = 4000
        multi_stage_rewards = True


class ElSpiderLidarNavBarrierSameDimCfg(ElSpiderLidarConfinedCfg):
    """Stage-D: barrier obstacle-avoidance style task (same obs/action dimensions)."""

    class env(ElSpiderLidarConfinedCfg.env):
        num_envs = 512

    class sim(ElSpiderLidarConfinedCfg.sim):
        class physx(ElSpiderLidarConfinedCfg.sim.physx):
            max_gpu_contact_pairs = 2**25
            default_buffer_size_multiplier = 12

    class terrain(ElSpiderLidarConfinedCfg.terrain):
        mesh_type = 'confined_trimesh'
        curriculum = True
        max_init_terrain_level = 0
        num_rows = 1
        num_cols = 1
        difficulty_scale = 0.6
        spawn_area_size = 6.0
        goal_navigation = False
        goal_offset_y = 3.0
        confined_terrain_proportions = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

    class commands(ElSpiderLidarConfinedCfg.commands):
        goal_directed = False
        heading_command = False
        curriculum = False
        resampling_time = 4.0

        class ranges(ElSpiderLidarConfinedCfg.commands.ranges):
            lin_vel_x = [0.1, 0.7]
            lin_vel_y = [-0.15, 0.15]
            ang_vel_yaw = [-0.6, 0.6]

    class rewards(ElSpiderLidarConfinedCfg.rewards):
        base_height_target = 0.34
        safe_obstacle_dist = 0.6
        danger_obstacle_dist = 0.22
        collision_threshold = 0.05
        goal_reach_threshold = 0.25
        goal_max_distance = 10.0
        terminate_on_goal_reached = False

        class scales(ElSpiderLidarConfinedCfg.rewards.scales):
            tracking_lin_vel = 0.8
            tracking_ang_vel = 0.4
            collision = -2.0
            obstacle_avoidance = 1.2
            collision_penalty = -2.0
            stand_still = -0.3
            goal_reaching = 0.0
            goal_progress = 0.0
            goal_bonus = 0.0
            goal_heading = 0.0


class ElSpiderLidarNavBarrierSameDimCfgPPO(ElSpiderLidarConfinedCfgPPO):
    """PPO config for barrier navigation style stage."""

    class policy(ElSpiderLidarConfinedCfgPPO.policy):
        actor_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        critic_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        activation = 'elu'
        init_noise_std = SAME_DIM_INIT_NOISE_STD

    class algorithm(ElSpiderLidarConfinedCfgPPO.algorithm):
        entropy_coef = 0.003
        desired_kl = 0.01

    class runner(ElSpiderLidarConfinedCfgPPO.runner):
        experiment_name = 'elspider_lidar_nav_barrier_same_dim'
        max_iterations = 5000


class ElSpiderLidarWalkFlatSameDimCfg(ElSpiderLidarConfinedCfg):
    """Stage-1: pure locomotion + posture control on flat terrain (same dimensions)."""

    class init_state(ElSpiderLidarConfinedCfg.init_state):
        pos = [0.0, 0.0, 0.35]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "RF_HAA": 0.0,
            "RM_HAA": 0.0,
            "RB_HAA": 0.0,
            "LF_HAA": 0.0,
            "LM_HAA": 0.0,
            "LB_HAA": 0.0,

            "RF_HFE": 0.2,
            "RM_HFE": 0.2,
            "RB_HFE": 0.2,
            "LF_HFE": 0.2,
            "LM_HFE": 0.2,
            "LB_HFE": 0.2,

            "RF_KFE": 0.3,
            "RM_KFE": 0.3,
            "RB_KFE": 0.3,
            "LF_KFE": 0.3,
            "LM_KFE": 0.3,
            "LB_KFE": 0.3,
        }

    class sim(ElSpiderLidarConfinedCfg.sim):
        class physx(ElSpiderLidarConfinedCfg.sim.physx):
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5

    class terrain(ElSpiderLidarConfinedCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        measure_heights = False
        goal_navigation = False
        goal_offset_y = 2.0

    class commands(ElSpiderLidarConfinedCfg.commands):
        goal_directed = False
        heading_command = False
        curriculum = False
        max_curriculum = 1.0
        resampling_time = 4.0

        class ranges(ElSpiderLidarConfinedCfg.commands.ranges):
            lin_vel_x = [0.15, 0.9]
            lin_vel_y = [-0.25, 0.25]
            ang_vel_yaw = [-0.7, 0.7]

    class domain_rand(ElSpiderLidarConfinedCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.3, 1.25]
        randomize_base_mass = False
        added_mass_range = [-5., 5.]
        push_robots = False
        push_interval_s = 3
        max_push_vel_xy = 1.

    class control(ElSpiderLidarConfinedCfg.control):
        stiffness = {'HAA': 60., 'HFE': 60., 'KFE': 60.}
        damping = {'HAA': 0.8, 'HFE': 0.8, 'KFE': 0.8}
        action_scale = 0.25
        decimation = 4
        use_actuator_network = False
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"

    class rewards(ElSpiderLidarConfinedCfg.rewards):
        base_height_target = 0.28
        only_positive_rewards = False
        multi_stage_rewards = True
        reward_stage_threshold = -20.0
        reward_min_stage = 1
        reward_max_stage = 1
        goal_reach_threshold = 0.15
        terminate_on_goal_reached = False

        class scales(ElSpiderLidarConfinedCfg.rewards.scales):
            tracking_lin_vel = 1.2
            tracking_ang_vel = 0.5
            lin_vel_z = -3.0
            ang_vel_xy = -0.2
            orientation = -1.8
            torques = -0.0001
            dof_vel = -0.00045
            dof_acc = [-5e-8, -1.5e-7]
            base_height = -0.8
            feet_slip = -0.25
            feet_air_time = 0.35
            gait_2_step = -0.25
            collision = -0.7
            feet_stumble = [-0.0, -0.2]
            action_rate = -0.006
            stand_still = 0.0
            stand_still2 = -0.1
            dof_pos_limits = -1.0
            feet_contact_forces = [-0.2, -0.5]
            shank_perp2ground = -0.12
            obstacle_avoidance = 0.0
            collision_penalty = 0.0
            goal_reaching = 0.0
            goal_progress = 0.0
            goal_bonus = 0.0
            goal_heading = 0.0


class ElSpiderLidarWalkFlatSameDimCfgPPO(ElSpiderLidarConfinedCfgPPO):
    """PPO config for stage-1 pure locomotion."""

    class policy(ElSpiderLidarConfinedCfgPPO.policy):
        actor_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        critic_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        activation = 'elu'
        init_noise_std = 0.20

    class algorithm(ElSpiderLidarConfinedCfgPPO.algorithm):
        entropy_coef = 0.004
        desired_kl = 0.008

    class runner(ElSpiderLidarConfinedCfgPPO.runner):
        experiment_name = 'elspider_lidar_walk_flat_same_dim'
        max_iterations = 3000
        multi_stage_rewards = True


class ElSpiderLidarNavFlatSameDimCfg(ElSpiderLidarWalkFlatSameDimCfg):
    """Stage-2: flat navigation with goal-directed policy (same dimensions)."""

    class terrain(ElSpiderLidarWalkFlatSameDimCfg.terrain):
        goal_navigation = True
        goal_offset_y = 3.5

    class commands(ElSpiderLidarWalkFlatSameDimCfg.commands):
        goal_directed = True
        heading_command = True
        curriculum = False
        max_curriculum = 1.0

        class ranges(ElSpiderLidarWalkFlatSameDimCfg.commands.ranges):
            lin_vel_x = [0.15, 0.9]
            lin_vel_y = [-0.25, 0.25]
            ang_vel_yaw = [-0.7, 0.7]

    class rewards(ElSpiderLidarWalkFlatSameDimCfg.rewards):
        base_height_target = 0.28
        goal_reach_threshold = 0.25
        goal_max_distance = 6.5
        terminate_on_goal_reached = True

        class scales(ElSpiderLidarWalkFlatSameDimCfg.rewards.scales):
            obstacle_avoidance = 0.20
            collision_penalty = -0.15
            corridor_centering = 0.25
            goal_reaching = 5.0
            goal_progress = 3.0
            goal_bonus = 10.0
            goal_heading = 0.5


class ElSpiderLidarNavFlatSameDimCfgPPO(ElSpiderLidarWalkFlatSameDimCfgPPO):
    """PPO config for stage-2 flat navigation."""

    class policy(ElSpiderLidarWalkFlatSameDimCfgPPO.policy):
        actor_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        critic_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        activation = 'elu'
        init_noise_std = 0.20

    class runner(ElSpiderLidarWalkFlatSameDimCfgPPO.runner):
        experiment_name = 'elspider_lidar_nav_flat_same_dim'
        max_iterations = 6000

    class algorithm(ElSpiderLidarWalkFlatSameDimCfgPPO.algorithm):
        entropy_coef = 0.004
        desired_kl = 0.008


class ElSpiderLidarConfinedEasySameDimCfg(ElSpiderLidarConfinedCfg):
    """Easy confined stage with same obs/action dimensions for curriculum transfer.

    Stage-2 after flat pretraining and before full confined training.
    """

    class env(ElSpiderLidarConfinedCfg.env):
        num_envs = 512

    class sim(ElSpiderLidarConfinedCfg.sim):
        class physx(ElSpiderLidarConfinedCfg.sim.physx):
            max_gpu_contact_pairs = 2**24
            default_buffer_size_multiplier = 8

    class terrain(ElSpiderLidarConfinedCfg.terrain):
        mesh_type = 'confined_trimesh'
        use_terrain_obj = False
        curriculum = True
        num_rows = 6
        num_cols = 1
        difficulty_scale = 0.10
        goal_offset_y = 4.5
        spawn_area_size = 1.0
        corridor_only = True
        corridor_uniform_width = True
        corridor_width_override = 1.05
        corridor_obstacle_density_override = 0.0
        confined_terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    class rewards(ElSpiderLidarConfinedCfg.rewards):
        base_height_target = 0.28
        goal_max_distance = 6.0
        safe_obstacle_dist = 0.70
        danger_obstacle_dist = 0.40

        class scales(ElSpiderLidarConfinedCfg.rewards.scales):
            goal_reaching = 7.0
            goal_progress = 3.5
            goal_bonus = 15.0
            goal_heading = 0.6
            corridor_centering = 0.5
            obstacle_avoidance = 0.30
            collision = -1.1
            collision_penalty = -0.30


class ElSpiderLidarConfinedEasySameDimCfgPPO(ElSpiderLidarConfinedCfgPPO):
    """PPO config for easy confined same-dim stage."""

    class policy(ElSpiderLidarConfinedCfgPPO.policy):
        actor_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        critic_hidden_dims = SAME_DIM_POLICY_HIDDEN_DIMS
        activation = 'elu'
        init_noise_std = SAME_DIM_INIT_NOISE_STD

    class runner(ElSpiderLidarConfinedCfgPPO.runner):
        experiment_name = 'elspider_lidar_confined_easy_same_dim'
        max_iterations = 12000
