"""Configuration for the second envelope-conditioned EL4090 environment."""

from legged_gym.envs.el_4090.spider_envelop.el4090_spider_config import (
    El4090EnvelopCfg,
    El4090EnvelopCfgPPO,
)


class El4090Envelop2Cfg(El4090EnvelopCfg):
    class env(El4090EnvelopCfg.env):
        # Proprioception only. Morphology priors and HAA ranges remain internal
        # environment state and are not policy inputs.
        num_observations = 66
        num_physical_priors = 0
        num_haa_range_observations = 0
        num_gait_phase_observations = 0

    class commands(El4090EnvelopCfg.commands):
        # Policy command is strictly [vx, vy, yaw_rate].
        num_commands = 3
        heading_command = False
        condition_dim = 0
        condition_names: list = []

        class ranges(El4090EnvelopCfg.commands.ranges):
            lin_vel_x = [-1.5, 1.5]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class envelope:
        condition_names: list = list(El4090EnvelopCfg.commands.condition_names)
        morphology_prior_mode = El4090EnvelopCfg.commands.morphology_prior_mode
        morphology_prior_weights = El4090EnvelopCfg.commands.morphology_prior_weights
        morphology_middle_front_follow_weight = (
            El4090EnvelopCfg.commands.morphology_middle_front_follow_weight
        )

        class ranges:
            front_width = list(El4090EnvelopCfg.commands.ranges.front_width)
            middle_width = list(El4090EnvelopCfg.commands.ranges.middle_width)
            back_width = list(El4090EnvelopCfg.commands.ranges.back_width)
            forward_limit = list(El4090EnvelopCfg.commands.ranges.forward_limit)
            backward_limit = list(El4090EnvelopCfg.commands.ranges.backward_limit)
            morphology_front_prior = list(
                El4090EnvelopCfg.commands.ranges.morphology_front_prior
            )
            morphology_middle_prior = list(
                El4090EnvelopCfg.commands.ranges.morphology_middle_prior
            )
            morphology_back_prior = list(
                El4090EnvelopCfg.commands.ranges.morphology_back_prior
            )

    class control(El4090EnvelopCfg.control):
        # Smooth morphology-preset changes before they reach the PD controller.
        # This prevents a sampled spider<->mammal change from producing an
        # instantaneous joint-target jump and contact impulse.
        control_type = "P"
        default_dof_pos_filter_tau = 0.5

    class init_state(El4090EnvelopCfg.init_state):
            pos = [0.0, 0.0, 0.53]  # x,y,z [m]
            default_joint_angles = {  # = target angles [rad] when action = 0.0
                "RF_HAA": 0.0,
                "RM_HAA": 0.0,
                "RB_HAA": 0.0,
                "LF_HAA": 0.0,
                "LM_HAA": 0.0,
                "LB_HAA": 0.0,
    
                "RF_HFE": 0.3020,
                "RM_HFE": 0.3020,
                "RB_HFE": 0.3020,
                "LF_HFE": 0.3020,
                "LM_HFE": 0.3020,
                "LB_HFE": 0.3020,
    
                "RF_KFE": -0.2708,
                "RM_KFE": -0.2708,
                "RB_KFE": -0.2708,
                "LF_KFE": -0.2708,
                "LM_KFE": -0.2708,
                "LB_KFE": -0.2708,
            }
            mammal_default_joint_angles = {
                "RF_HAA": -1.308,
                "RM_HAA": 1.308,
                "RB_HAA": 1.308,
                "LF_HAA": -1.308,
                "LM_HAA": 1.308,
                "LB_HAA": 1.308,
    
                "RF_HFE": 0.3022,
                "RM_HFE": 0.3022,
                "RB_HFE": 0.3022,
                "LF_HFE": 0.3022,
                "LM_HFE": 0.3022,
                "LB_HFE": 0.3022,

                "RF_KFE": -0.3018,
                "RM_KFE": -0.3018,
                "RB_KFE": -0.3018,
                "LF_KFE": -0.3018,
                "LM_KFE": -0.3018,
                "LB_KFE": -0.3018,
            }

    class haa_swing_range:
        # analytic, monte_carlo, or network
        method = "network"
        joint_lower = -3.0
        joint_upper = 3.0
        leg_reach = 0.55
        front_hip_offset = 0.10
        middle_hip_offset = 0.20
        back_hip_offset = 0.10
        spider_swing_limit = 1.05
        mammal_swing_limit = 0.45
        minimum_half_range = 0.05
        monte_carlo_samples = 2048
        monte_carlo_quantile = 0.0
        monte_carlo_seed = None
        network_checkpoint = (
            "{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/el_4090/"
            "spider_envelop_2/envelop_network/haa_range.pt"
        )

    class rewards(El4090EnvelopCfg.rewards):
        haa_range_margin = 0.0
        haa_phase_period = 0.5
        haa_phase_amplitude_ratio = 0.6
        haa_phase_min_command = 0.1
        haa_phase_full_command = 0.5
        haa_phase_yaw_weight = 0.5

        feet_air_time_target = 0.2  # 期望单脚离地时间 [s]，用于 feet_air_time 奖励

        base_height_spider_target = 0.53  # 蜘蛛形态下的机身目标高度 [m]
        base_height_mammal_target = 0.53  # 哺乳形态下的机身目标高度 [m]


        class scales:
            # Basic locomotion tracking and body stabilization.
            termination = -0.0
            tracking_lin_vel = 6
            tracking_ang_vel = 2.5
            lin_vel_z = -2
            ang_vel_xy = -1.5   #-0.1
            orientation = -5
            base_height = -10

            # Smooth and energy-efficient joint motion.
            torques = -1e-5
            dof_vel = -1e-5
            dof_acc = -1.5e-6  #-1e-6
            action_rate = -0.003
            action_jerk = -0.005

            # Basic foot contact and gait quality.
            feet_slip = -0.05
            feet_air_time = 1.0
            collision = -1.0
            stand_still = -3.0  #-1.0
            feet_contact_forces = -0.03  #-0.01
            feet_async = -0.2
            feet_sync = -0.2
            tripod_contact_pattern = -2

            # stand_contact = -5.0

            # Generic actuator safety limits.
            dof_pos_limits = -0.5
            dof_vel_limits = -0.1
            torque_limits = -0.01

            # Keep only the generated per-leg HAA range constraint here. A
            # phase target encourages smooth use of the available HAA range.
            haa_range_violation = -3.0
            # haa_phase_tracking = -3.0


class El4090Envelop2CfgPPO(El4090EnvelopCfgPPO):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy:
        init_noise_std = 0.1
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm(El4090EnvelopCfgPPO.algorithm):
        # Symmetry augmentation configuration
        class symmetry_cfg:
            use_data_augmentation = True
            use_mirror_loss = True
            mirror_loss_coeff = 0.6
            data_augmentation_func = (
                "legged_gym.envs.el_4090.spider_envelop_2.symmetry:"
                "get_elair_lidar_xsym_obs_act"
            )

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 3000

        # logging
        save_interval = 50
        experiment_name = "obs66_Envelop_2"
        run_name = ""

        # load and resume
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
