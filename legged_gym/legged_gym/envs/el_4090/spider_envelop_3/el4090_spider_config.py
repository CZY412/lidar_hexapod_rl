"""Complete configuration for the API-driven third EL4090 envelope task."""

from legged_gym.envs.el_4090.spider_envelop.el4090_spider_config import (
    El4090EnvelopCfg,
    El4090EnvelopCfgPPO,
)


class El4090Envelop3Cfg(El4090EnvelopCfg):
    class env(El4090EnvelopCfg.env):
        # Proprioception plus six HAA [lower, upper] range pairs.
        num_observations = 78
        num_physical_priors = 0
        num_haa_range_observations = 12
        num_gait_phase_observations = 0

    class commands(El4090EnvelopCfg.commands):
        num_commands = 3
        heading_command = False
        condition_dim = 0
        condition_names = []
        envelope_debug_viz = True
        envelope_debug_env_ids = [0]

        class ranges(El4090EnvelopCfg.commands.ranges):
            lin_vel_x = [-1.5, 1.5]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class envelope:
        # The only external training inputs. Together they define C-mode's
        # symmetric six-vertex allowed polygon.
        condition_names = [
            "front_width", "middle_width", "back_width", "forward_limit", "backward_limit",
        ]

        class ranges:
            front_width = [0.3, 0.6]
            middle_width = [0.3, 0.7]
            back_width = [0.3, 0.6]
            forward_limit = [0.6, 0.9]
            backward_limit = [-0.9, -0.6]

    class envelope_api_range:
        # New el4090_envelope API settings. Change geometry above to alter the
        # allowed envelope; use these only to control numerical computation.
        directions = 32
        margin = 0.05
        tolerance = 1e-6
        sweep_steps = 201
        cache_precision = 3
        sampling_max_attempts = 32

        # HAA is deliberately searched only on the old spider-to-mammal
        # segment, but there is no mammal pose or morphology interpolation.
        # Order is LB, LF, LM, RB, RF, RM, matching Isaac HAA DOF order.
        haa_search_ranges = [
            [0.0, 1.308], [-1.308, 0.0], [0.0, 1.308],
            [0.0, 1.308], [-1.308, 0.0], [0.0, 1.308],
        ]

    class haa_range_network:
        # Five geometry inputs -> six [lower, upper] HAA intervals.
        network_checkpoint = (
            "{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/el_4090/"
            "spider_envelop_3/envelop_network/api_haa_range_network.pt"
        )

    class control(El4090EnvelopCfg.control):
        control_type = "P"
        default_dof_pos_filter_tau = 0.5

    class init_state(El4090EnvelopCfg.init_state):
        pos = [0.0, 0.0, 0.47]
        # The only fixed posture. HFE/KFE are pinned to these values while the
        # API computes HAA running ranges over physical HAA joint limits.
        default_joint_angles = {
            "RF_HAA": 0.0, "RM_HAA": 0.0, "RB_HAA": 0.0,
            "LF_HAA": 0.0, "LM_HAA": 0.0, "LB_HAA": 0.0,
            "RF_HFE": 0., "RM_HFE": 0., "RB_HFE": 0.,
            "LF_HFE": 0., "LM_HFE": 0., "LB_HFE": 0.,
            "RF_KFE": 0., "RM_KFE": 0., "RB_KFE": 0.,
            "LF_KFE": 0., "LM_KFE": 0., "LB_KFE": 0.,
        }

    class rewards(El4090EnvelopCfg.rewards):
        # A height inside this band is equally acceptable.
        base_height_min = 0.47
        base_height_max = 0.60

        # HAA range reward: motion is rewarded only in the useful interior.
        # The final boundary band earns no swing reward; crossing a predicted
        # range boundary is penalized separately.
        # Allow a small tolerance around the predicted range during learning.
        haa_range_margin = 0.12
        haa_range_swing_min_command = 0.10
        haa_range_swing_velocity_clip = 4.0
        haa_range_swing_target_excursion_ratio = 0.65
        haa_range_swing_boundary_buffer_ratio = 0.12

        feet_air_time_target = 0.2
        # base_height_spider_target = 0.53

        class scales:
            termination = -0.0
            tracking_lin_vel = 6
            tracking_ang_vel = 2.5
            lin_vel_z = -2
            ang_vel_xy = -1.5
            orientation = -5
            base_height = -10
            torques = -1e-5
            dof_vel = -1e-5
            dof_acc = -1.5e-6
            action_rate = -0.003
            action_jerk = -0.005
            feet_slip = -0.05
            feet_air_time = 1.0
            collision = -1.0
            stand_still = -3.0
            feet_contact_forces = -0.03
            feet_async = -0.1
            feet_sync = -0.1
            tripod_contact_pattern = -1
            dof_pos_limits = -0.5
            dof_vel_limits = -0.1
            torque_limits = -0.01
            haa_range_violation = -5.0
            haa_range_swing = 0.5


class El4090Envelop3CfgPPO(El4090EnvelopCfgPPO):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy:
        init_noise_std = 0.1
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm(El4090EnvelopCfgPPO.algorithm):
        class symmetry_cfg:
            use_data_augmentation = True
            use_mirror_loss = True
            mirror_loss_coeff = 0.6
            data_augmentation_func = (
                "legged_gym.envs.el_4090.spider_envelop_3.symmetry_obs78:"
                "get_elair_lidar_xsym_obs_act"
            )

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 3000
        save_interval = 50
        experiment_name = "obs78_Envelop_3_api_ranges"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
