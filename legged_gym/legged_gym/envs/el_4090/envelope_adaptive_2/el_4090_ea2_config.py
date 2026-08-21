"""Configuration for ``el4090_ea2`` (envelope_adaptive_2, M1).

Environment cfg: simplified BaseTask, no robot actor, one global fixed map.
PPO cfg: rsl_rl ``ActorCriticRecurrent`` (single GRU + MLP heads).

All numeric values follow README v2.  Implementation agents must NOT edit this
file; integration owner updates it only when the README contract changes.
"""

import math

from legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)


class El4090EA2Cfg(LeggedRobotCfg):
    """M1 envelope-perception environment config (BaseTask, no robot)."""

    class env(LeggedRobotCfg.env):
        num_envs = 1024  # start small; benchmark A*/raycast before scaling
        num_observations = 190          # 187 selected channels + 3 ego-motion
        num_privileged_obs = None       # no asymmetric critic
        num_actions = 5                 # raw envelope params (sigmoid-affine in env)
        env_spacing = 0.                # all envs share the single global map origin
        episode_length_s = 40.          # timeout timer for probabilistic GRU reset
        memory_reset_prob = 0.15        # probability to clear GRU at each timeout
        send_timeouts = True

    class sim(LeggedRobotCfg.sim):
        dt = 0.02                       # 50 Hz policy step
        substeps = 1
        gravity = [0.0, 0.0, -9.81]
        up_axis = 1

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23
            default_buffer_size_multiplier = 5
            contact_collection = 2

    class map:
        size_m = 74.0                   # 4 x 4 tiles (16m) + 5m border each side
        resolution_m = 0.1
        grid_shape = [740, 740]
        # No physical boundary walls.  When True, only the *inflated planning*
        # grid border is blocked so A* keeps the robot inside the map.
        boundary_occupied = True
        ground_margin_m = 2.0            # warp ground plane extends past map edges
        inflation_m = 0.35               # A* lateral safety (3.5 cells -> 4)
        inflation_cells = 4
        n_tiles = 4                      # 4 x 4 pillar-field plots
        tile_size_m = 16.0               # pd_gru terrain_length / terrain_width
        border_size_m = 5.0              # pd_gru border_size
        min_free_component_ratio = 0.95  # inflated free space must be mostly connected
        max_gen_attempts = 20
        n_validation_paths = 12          # 8~16 validation A* runs at map acceptance
        min_solved_ratio = 0.8
        path_near_obstacle_ratio = 0.3
        near_obstacle_range = [0.7, 1.5]
        require_constraint_primitive = False  # pillar field has no corridor type

    class obstacles:
        # pd_gru_lidar pillar_field_terrain parameters (per 16m x 16m tile)
        pillar_count_min = 15
        pillar_count_max = 15
        pillar_size_x_min = 0.5
        pillar_size_x_max = 4.0
        pillar_size_y_min = 0.5
        pillar_size_y_max = 4.0
        pillar_height_min = 1.0
        pillar_height_max = 2.0
        pillar_min_separation = 2.2
        pillar_center_clear_radius = 2.2
        pillar_spawn_radius = 7.5
        pillar_allow_height_variation = True

    class path:
        speed_range = [1.0, 1.0]         # stage 1: fixed forward speed
        resample_time_s = 4.0
        delta_target_deg_range = [0.0, 0.0]  # stage 1: heading follows tangent
        omega_max = 1.5
        k_p = 5.0
        min_turn_radius = 1.0
        resample_dist = 0.2
        goal_min_obstacle_dist = 0.5
        min_path_len = 3.0
        noise_amp_range = [0.0, 0.0]     # stage 1: no lateral path noise
        noise_fc_hz = 1.0
        noise_retries = 8

    class sway:
        pos_amp_range = [0.0, 0.0]       # stage 1: no lateral sway
        heading_amp_range = [0.0, 0.0]   # stage 1: no heading sway
        fc_hz = 1.0

    class height:
        min_m = 0.52                     # stage 1: fixed base height
        max_m = 0.52
        resample_time_s = 4.0
        tau_s = 0.8
        wobble_amp_range = [0.0, 0.0]    # stage 1: no vertical wobble
        wobble_fc_hz = 1.0

    class lidar:
        airy_n_azimuth = 900
        airy_n_elevation = 96
        airy_horizontal_resolution_deg = 0.4
        airy_vertical_fov_deg = [0, 90.0]
        far_plane = 60.0                 # LidarConfig.max_range
        min_range = 0.2                  # kept for compatibility; not used by 187-channel path
        update_frequency_hz = 10.0
        effective_max_range = 3.2        # max slant of selected 187 channels
        use_reduced_raycast = True       # train with only the 187 fixed channels
        offset_pos = [0.7, 0.0, -0.05]   # current EA2 sensor placement
        sensor_offset_rpy = [0.0, math.pi / 2.0 + 0.1, 0.0]
        pointcloud_in_world_frame = False
        randomize_placement = False

        # Debug point-cloud visualization (README 2.7 red/green spheres)
        debug_env_ids = [0]
        debug_point_stride = 1

        # LidarConfig noise fields; applied to ALL rays (camera-original semantics)
        enable_sensor_noise = True
        pixel_std_dev_multiplier = 0.02
        pixel_dropout_prob = 0.02
        random_distance_noise = 0.0
        random_angle_noise = 0.0

    class envelope:
        margin = 0.05                    # exact half-plane offset for collision
        # 5 params + 3 priors are loaded from the frozen contract path
        # (_contracts.ENVELOPE_SPEC_CONFIG_PATH).
        # spider_envelop_2-style bold footprint drawing
        debug_color = (0.0, 0.85, 1.0)
        debug_line_radius = 0.012
        debug_line_samples = 8
        debug_ground_z_offset = 0.02

    class rewards:
        # BaseTask does not auto-scale reward terms; env multiplies directly.
        class scales:
            potential = 1.0
            collision = -2.0
            action_rate = -0.01

    class normalization:
        clip_observations = 100.
        clip_actions = 100.


class El4090EA2CfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.3
        actor_hidden_dims = [256, 128]
        critic_hidden_dims = [256, 128]
        activation = "elu"
        # recurrent fields (num_actor_obs/num_critic_obs come from runner env,
        # never declare them here -- duplicate keyword -> TypeError)
        rnn_type = "gru"
        rnn_hidden_dim = 187
        rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        class symmetry_cfg:
            # M1 default: symmetry off.  Optional T8 symmetry module can enable:
            # data_augmentation_func =
            #   "legged_gym.envs.el_4090.envelope_adaptive_2.symmetry:get_ea2_xsym_obs_act"
            use_data_augmentation = False
            use_mirror_loss = False
            mirror_loss_coeff = 0.0
            data_augmentation_func = None

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        algorithm_class_name = "PPO"
        num_steps_per_env = 50          # 1 s ~= 10 LiDAR frames per segment
        max_iterations = 3000
        save_interval = 50
        experiment_name = "el4090_ea2"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
