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

from legged_gym.envs.el_4090.spider_nomal.el4090_spider_config import El4090SpiderCfg, El4090SpiderCfgPPO

SAME_DIM_POLICY_HIDDEN_DIMS = [128, 64, 32]
SAME_DIM_INIT_NOISE_STD = 0.35

class El4090LidarCfg(El4090SpiderCfg):
    class env(El4090SpiderCfg.env):
        num_envs = 512  # Reduced to fit GPU memory for confined terrain initialization
        num_actions = 18  # 6 legs × 3 joints
        
        # Base observations: 3+3+3+3+18+18+18 = 66
        # LiDAR observations: 12×8 = 96
        # Goal observations: 2 (direction_angle, normalized_distance)
        num_lidar_obs = 96  # num_theta_bins × num_phi_bins
        num_goal_obs = 2    # goal direction angle + normalized distance
        num_observations = 66 + 96 + 2  # 164 total
        
        episode_length_s = 24  # Longer episode to reach goal

        # Debug settings
        debug_mode = False  # Enable debug output
        debug_interval = 100  # Print debug info every N steps
        debug_env_id = 0  # Which environment to debug (0-based index)

    class sim(El4090SpiderCfg.sim):
        class physx(El4090SpiderCfg.sim.physx):
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
        sensor_offset = [0.0, 0.0, 0.15]  # [x, y, z] in meters
        sensor_rotation_deg = [0.0, 0.0, 0.0]  # [roll, pitch, yaw] in degrees

    class terrain(El4090SpiderCfg.terrain):
        mesh_type = 'confined_trimesh'  # "heightfield" # none, plane, heightfield or trimesh

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

        # selected = False  # select a unique terrain type and pass all arguments
        # terrain_kwargs = None  # Dict of arguments for selected terrain
        # max_init_terrain_level = 0  # starting curriculum state

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

    class control(El4090SpiderCfg.control):
        control_type = 'P'
        # PD Drive parameters matching Anymal:
        # stiffness = {'HAA': 120., 
        #              'HFE': 120., 
        #              'KFE': 120.}  # [N*m/rad]
        # damping = {'HAA': 2, 
        #            'HFE': 2, 
        #            'KFE': 2}     # [N*m*s/rad]
        stiffness = {'HAA': 60., 'HFE': 60., 'KFE': 60.}  # [N*m/rad]
        damping = {'HAA': 0.8, 'HFE': 0.8, 'KFE': 0.8}    # [N*m*s/rad]

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25  # Enable Network-0.5 | Disable Network-0.3

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = False
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"


    class asset(El4090SpiderCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/el_4090/urdf/el_4090.urdf"
        name = "el_4090"
        foot_name = "FOOT"
        collapse_fixed_joints = False # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        shoulder_name = "shoulder"
        penalize_contacts_on = ["BASE","SHANK","THIGH"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class init_state(El4090SpiderCfg.init_state):
        pos = [0.0, 0.0, 0.45]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "RF_HAA": 0.0,
            "RM_HAA": 0.0,
            "RB_HAA": 0.0,
            "LF_HAA": 0.0,
            "LM_HAA": 0.0,
            "LB_HAA": 0.0,

            "RF_HFE": 0.0,
            "RM_HFE": 0.0,
            "RB_HFE": 0.0,
            "LF_HFE": 0.0,
            "LM_HFE": 0.0,
            "LB_HFE": 0.0,

            "RF_KFE": 0.0,
            "RM_KFE": 0.0,
            "RB_KFE": 0.0,
            "LF_KFE": 0.0,
            "LM_KFE": 0.0,
            "LB_KFE": 0.0,
        }

    ## Rewards V1 (normal dof_acc)
    class rewards(El4090SpiderCfg.rewards):
        max_contact_force = 225.
        base_height_target = 0.45
        only_positive_rewards = False  # Allow negative rewards for collision
        # Multi-stage
        # Stage 0: Learn to walk with tripod gait (with / w\o actuator net)
        # Stage 1: Correct DOF and FootZ positions / Prevent Slip
        multi_stage_rewards = True  # if true, reward scales should be list
        reward_stage_threshold = 1.0
        reward_min_stage = 0  # Start from 0
        reward_max_stage = 1
        
        # Obstacle avoidance parameters
        safe_obstacle_dist = 0.60    # Keep a conservative but feasible clearance in pillar piles
        danger_obstacle_dist = 0.18  # Start danger penalty close to true collision envelope
        collision_threshold = 0.03  # REDUCED from 0.05: only terminate on actual collision (3cm)
        
        # Termination protection - generous grace period
        collision_termination_after_steps = 200  # INCREASED from 50: let robot survive much longer
        allow_initial_contact_steps = 30  # Grace period at episode start
        
        # Goal navigation reward parameters
        goal_reach_threshold = 0.9    # Tighten goal reach criterion for the final task
        goal_max_distance = 6.5       # Max expected distance to goal [meters] (for normalization)

        class scales:
            termination = -2.0         # Penalize episode termination
            tracking_lin_vel = 3
            tracking_ang_vel = 1.5
            lin_vel_z = -3.0
            ang_vel_xy = -0.2
            orientation = -1.8
            torques = -0.0001
            dof_vel = -0.00045
            dof_acc = -5e-8
            base_height = -0.8
            feet_slip = -0.0  # Before feet_air_time
            feet_air_time = 0.35
            collision = -1.0
            feet_stumble = -0.0
            action_rate = -0.006
            stand_still = -0.1
            dof_pos_limits = -1.0
            dof_vel_limits = -1.
            torque_limits = -0.01
            feet_contact_forces = -0.2
            shank_perp2ground = -0.12
            shank_vertical = -1
            stand_on_six_legs = -2
            # swing_leg_y_stability = -5
            feet_async = -1.5
            feet_sync = -1.5

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

    class commands(El4090SpiderCfg.commands):
        curriculum = True  # Enable: start with slow commands, increase over time
        max_curriculum = 1.0
        num_commands = 4
        resampling_time = 10.0  # Longer resampling: goal provides consistent direction
        heading_command = True  # Will be overridden to use goal heading
        goal_directed = True    # Use goal position to generate heading commands
        
        class ranges(El4090SpiderCfg.commands.ranges):
            lin_vel_x = [-3.0, 3.0]  # min max [m/s]
            lin_vel_y = [-1.5, 1.5]   # min max [m/s]
            ang_vel_yaw = [-1.5, 1.5]    # min max [rad/s]
            heading = [-3.14, 3.14]   # Will be overridden by goal heading

    class domain_rand(El4090SpiderCfg.domain_rand):
        # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.
        randomize_friction = True
        friction_range = [0.3, 1.0]
        randomize_base_mass = True
        added_mass_range = [-10., 10.]
        push_robots = True
        push_interval_s = 3
        max_push_vel_xy = 1.

    class noise(El4090SpiderCfg.noise):
        add_noise = True
        noise_level = 1.5  # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 1.5
            lin_vel = 0.8
            ang_vel = 0.8
            gravity = 0.5
            height_measurements = 0.1
            lidar = 0.05  # LiDAR observation noise

class El4090LidarCfgPPO(El4090SpiderCfgPPO):
    class policy(El4090SpiderCfgPPO.policy):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class runner (El4090SpiderCfgPPO.runner):
        run_name = ''
        experiment_name = 'el4090_spider_normal' # el4090_lidar
        load_run = -1
        max_iterations = 3000
        multi_stage_rewards = True

        # Checkpointing
        save_interval = 50
        
        # Logging
        log_interval = 10

    class algorithm(El4090SpiderCfgPPO.algorithm):
        # Symmetry augmentation configuration
        entropy_coef = 0.005         # Slightly more exploration for robust obstacle handling
        learning_rate = 7e-4          # Smoother policy updates for gait stability
        num_learning_epochs = 5
        gamma = 0.99
        lam = 0.95
        num_mini_batches = 4
        desired_kl = 0.008            # Tighter updates for stability
        schedule = 'adaptive'         # Use adaptive LR schedule based on KL divergence

        
        class symmetry_cfg:
            use_data_augmentation = True
            use_mirror_loss = True
            mirror_loss_coeff = 0.6
            data_augmentation_func = "legged_gym.envs.el_4090.spider_lidar.el_4090_lidar:get_elair_xsym_obs_act"
        