"""Complete configuration for the API-driven third EL4090 envelope task."""

from legged_gym.envs.el_4090.spider_envelop.el4090_spider_config import (
    El4090EnvelopCfg,
    El4090EnvelopCfgPPO,
)


class El4090Envelop3Cfg(El4090EnvelopCfg):
    class env(El4090EnvelopCfg.env):
        # Proprioception, one target height, and six HAA [lower, upper] pairs.
        num_observations = 79
        num_physical_priors = 0
        num_haa_range_observations = 12
        num_gait_phase_observations = 0

    class commands(El4090EnvelopCfg.commands):
        num_commands = 3
        heading_command = False
        condition_dim = 0
        condition_names = []
        # The parent task's visualization expects the removed five-parameter
        # envelope. Direct HAA-range v3 has no envelope polygon to draw.
        envelope_debug_viz = False

        class ranges(El4090EnvelopCfg.commands.ranges):
            lin_vel_x = [-1.5, 1.5]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class haa_range:
        # Direct distributions for six [lower, upper] HAA intervals. Order is
        # LB, LF, LM, RB, RF, RM; LB/RB, LF/RF, and LM/RM share one interval.
        leg_names = ["LB", "LF", "LM", "RB", "RF", "RM"]
        joint_limits = [-1.3, 1.3]
        min_width = 0.3
        max_sampling_attempts = 16
        # The midpoint of each sampled interval becomes the dynamic HAA bias
        # added to (but never written into) the unique spider default posture.
        center_bias_scale = 1.0

    class asset(El4090EnvelopCfg.asset):
        # Keep PhysX self-collision enabled. The active ``collision`` reward
        # penalizes contacts on these non-foot links, including leg-leg hits.
        self_collisions = 0
        penalize_contacts_on = ["BASE", "THIGH", "SHANK"]

    class control(El4090EnvelopCfg.control):
        control_type = "P"
        default_dof_pos_filter_tau = 0.5

    class init_state(El4090EnvelopCfg.init_state):
        pos = [0.0, 0.0, 0.52]
        # Unique spider standing posture calibrated for a 0.52 m base height.
        default_joint_angles = {
            "RF_HAA": 0.0, "RM_HAA": 0.0, "RB_HAA": 0.0,
            "LF_HAA": 0.0, "LM_HAA": 0.0, "LB_HAA": 0.0,
            "RF_HFE": 0.2475, "RM_HFE": 0.2475, "RB_HFE": 0.2475,
            "LF_HFE": 0.2475, "LM_HFE": 0.2475, "LB_HFE": 0.2475,
            "RF_KFE": -0.2325, "RM_KFE": -0.2325, "RB_KFE": -0.2325,
            "LF_KFE": -0.2325, "LM_KFE": -0.2325, "LB_KFE": -0.2325,
        }

    class rewards(El4090EnvelopCfg.rewards):
        # One shared reset, observation, and reward target height.
        target_height = 0.52

        # HAA range curriculum: keep the policy inside its sampled box without
        # prescribing a centre point or a particular HAA swing trajectory.
        haa_range_violation_softness = 0.25  # rad; Huber transition scale
        haa_range_violation_initial_weight = 0.10
        haa_range_violation_warmup_steps = 100000
        haa_range_compliance_margin = 0.04  # keep a small buffer from edges
        haa_range_compliance_softness = 0.04  # rad; soft in/out transition

        feet_air_time_target = 0.2


        class scales:
            termination = -0.0
            collision = -1.0  # PhysX contact penalty; includes self-collisions
            tracking_lin_vel = 6
            tracking_ang_vel = 2.5

            lin_vel_z = -2
            ang_vel_xy = -1.5
            orientation = -5
            base_height = -10

            torques = -1e-5
            dof_vel = -1e-5
            dof_acc = -1e-7
            action_rate = -0.003
            action_jerk = -0.005

            feet_slip = -0.05
            feet_air_time = 1.0

            stand_still = -3.0
            feet_contact_forces = -0.03

            feet_async = -0.1
            feet_sync = -0.1
            tripod_contact_pattern = -1

            dof_pos_limits = -0.5
            dof_vel_limits = -0.1
            torque_limits = -0.01

            haa_range_violation = -1.5
            haa_range_compliance = 0.30


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
                "legged_gym.envs.el_4090.spider_envelop_3.symmetry_obs79:"
                "get_elair_lidar_xsym_obs_act"
            )

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 3000
        save_interval = 50
        experiment_name = "obs79_Envelop_3_direct_haa_ranges"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
