"""Complete configuration for the API-driven third EL4090 envelope task."""

from legged_gym.envs.el_4090.spider_envelop.el4090_spider_config import (
    El4090EnvelopCfg,
    El4090EnvelopCfgPPO,
)


class El4090Envelop3Cfg(El4090EnvelopCfg):
    class env(El4090EnvelopCfg.env):
        # Proprioception only: geometry and derived priors stay internal.
        num_observations = 66
        num_physical_priors = 0
        num_haa_range_observations = 0
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
        prior_iterations = 3
        cache_precision = 3

    class control(El4090EnvelopCfg.control):
        control_type = "P"
        default_dof_pos_filter_tau = 0.5

    class init_state(El4090EnvelopCfg.init_state):
        pos = [0.0, 0.0, 0.53]
        # Do not change these modal endpoints independently of the envelope
        # API: HAA centers are normalized between them and HFE/KFE follow the
        # same derived front/middle/back prior.
        default_joint_angles = {
            "RF_HAA": 0.0, "RM_HAA": 0.0, "RB_HAA": 0.0,
            "LF_HAA": 0.0, "LM_HAA": 0.0, "LB_HAA": 0.0,
            "RF_HFE": 0.3020, "RM_HFE": 0.3020, "RB_HFE": 0.3020,
            "LF_HFE": 0.3020, "LM_HFE": 0.3020, "LB_HFE": 0.3020,
            "RF_KFE": -0.2708, "RM_KFE": -0.2708, "RB_KFE": -0.2708,
            "LF_KFE": -0.2708, "LM_KFE": -0.2708, "LB_KFE": -0.2708,
        }
        mammal_default_joint_angles = {
            "RF_HAA": -1.308, "RM_HAA": 1.308, "RB_HAA": 1.308,
            "LF_HAA": -1.308, "LM_HAA": 1.308, "LB_HAA": 1.308,
            "RF_HFE": 0.3022, "RM_HFE": 0.3022, "RB_HFE": 0.3022,
            "LF_HFE": 0.3022, "LM_HFE": 0.3022, "LB_HFE": 0.3022,
            "RF_KFE": -0.3018, "RM_KFE": -0.3018, "RB_KFE": -0.3018,
            "LF_KFE": -0.3018, "LM_KFE": -0.3018, "LB_KFE": -0.3018,
        }

    class rewards(El4090EnvelopCfg.rewards):
        haa_range_margin = 0.0
        haa_phase_period = 0.5
        haa_phase_amplitude_ratio = 0.6
        haa_phase_min_command = 0.1
        haa_phase_full_command = 0.5
        haa_phase_yaw_weight = 0.5
        feet_air_time_target = 0.2
        base_height_spider_target = 0.53
        base_height_mammal_target = 0.53

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
            feet_async = -0.2
            feet_sync = -0.2
            tripod_contact_pattern = -2
            dof_pos_limits = -0.5
            dof_vel_limits = -0.1
            torque_limits = -0.01
            haa_range_violation = -3.0


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
                "legged_gym.envs.el_4090.spider_envelop_3.symmetry:"
                "get_elair_lidar_xsym_obs_act"
            )

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 3000
        save_interval = 50
        experiment_name = "obs66_Envelop_3_api_ranges"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
