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
        # The v3 viewer draws this five-parameter footprint as well as the
        # selected HAA interval.  Make the footprint deliberately prominent.
        envelope_debug_viz = True
        envelope_debug_env_ids = [0]
        envelope_debug_ground_z_offset = 0.03
        envelope_debug_color = (0.0, 0.95, 1.0)
        envelope_debug_line_radius = 0.025
        envelope_debug_line_samples = 12
        haa_range_debug_env_ids = [0]
        haa_range_debug_leg = "LF"
        haa_range_debug_radius = 0.22
        haa_range_debug_samples = 33
        haa_range_debug_arc_color = (0.0, 0.42, 0.0)
        haa_range_debug_endpoint_color = (0.0, 0.24, 0.0)
        haa_range_debug_marker_color = (0.90, 0.08, 0.0)
        # Match the thick multi-line style used by envelop_2's visible envelope.
        haa_range_debug_line_radius = 0.014
        haa_range_debug_line_samples = 8

        class ranges(El4090EnvelopCfg.commands.ranges):
            lin_vel_x = [-1.5, 1.5]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

        # Do not inject standing commands during locomotion training.  The
        # standing-contact reward remains available for externally supplied
        # zero or near-zero commands.
        enable_stand_commands = False
        stand_command_probability = 0.05
        stand_command_exact_zero_fraction = 0.5
        stand_command_near_zero_limit = 0.05

    class haa_range:
        # Five envelope values are freely sampled from the network domain.
        # Runtime retries only for a non-finite or < 0.2 rad predicted interval;
        # collision analysis remains an offline diagnostic.
        # The directional rejection-range network and its 20k-entry valid
        # envelope bank.  This path is resolved from ``envs/el_4090``.
        network_path = "../../utils/envelop/data/api_haa_range_network.pt"
        leg_names = ["LB", "LF", "LM", "RB", "RF", "RM"]
        # Match the HAA-network output domain exactly.
        joint_limits = [-1.5, 1.5]
        min_width = 0.2
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
        pos = [0.0, 0.0, 0.53]
        # The HFE/KFE posture used when generating HAA-range network labels.
        default_joint_angles = {
            "RF_HAA": 0.0, "RM_HAA": 0.0, "RB_HAA": 0.0,
            "LF_HAA": 0.0, "LM_HAA": 0.0, "LB_HAA": 0.0,
            "RF_HFE": 0.3022, "RM_HFE": 0.3022, "RB_HFE": 0.3022,
            "LF_HFE": 0.3022, "LM_HFE": 0.3022, "LB_HFE": 0.3022,
            "RF_KFE": -0.3018, "RM_KFE": -0.3018, "RB_KFE": -0.3018,
            "LF_KFE": -0.3018, "LM_KFE": -0.3018, "LB_KFE": -0.3018,
        }

    class rewards(El4090EnvelopCfg.rewards):
        # Keep the v1 locomotion objective.  V3 only changes how the HAA
        # interval is sampled and observed; it does not add a second gait or
        # posture objective on top of v1.
        target_height = 0.53
        max_contact_force = 400.0
        feet_air_time_target = 0.25
        tripod_contact_threshold = 1.0
        tripod_contact_min_command = 0.1
        envelope_constraint_margin = 0.0
        envelope_constraint_min_command = 0.15
        stand_command_threshold = 0.1
        stand_contact_threshold = 10.0
        # Embedded-state target is ``default_dof_pos + haa_range_bias``.
        # HFE/KFE keep the base tolerance; only HAA receives the wider range.
        embedded_state_dof_pos_tolerance = 0.12
        embedded_state_haa_pos_tolerance = 0.35
        structure_transition_error_threshold = 0.10

        class scales:
            termination = -0.0
            collision = -1.0
            tracking_lin_vel = 6
            tracking_ang_vel = 2.5

            lin_vel_z = -2
            ang_vel_xy = -0.1
            orientation = -5
            base_height = -50

            torques = -1e-5
            dof_vel = -1e-5
            dof_acc = -1e-7
            feet_slip = -0.05
            feet_air_time = 3.0

            embedded_state_dof_pos = -5.0
            embedded_state_dof_vel = -0.02
            # Reuse the v1 reward slot to keep actual HAA positions inside
            # the per-leg interval inferred by the range network.
            morphology_haa_range = -3.0

            collision = -1.0
            action_rate = -0.001
            stand_still = -1.5
            stand_contact = -1.0

            dof_pos_limits = -0.5
            dof_vel_limits = -0.1
            torque_limits = -0.01
            feet_contact_forces = -0.05

            feet_async = -0.05
            feet_sync = -0.05
            tripod_contact_pattern = -2.0

            envelope_constraint = -20.0


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
            mirror_loss_coeff = 1.0
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
