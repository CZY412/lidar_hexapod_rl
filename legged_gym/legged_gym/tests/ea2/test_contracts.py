"""T0 sanity checks: frozen contracts and config match the fixed-channel design."""

from legged_gym.envs.el_4090.envelope_adaptive_2 import (
    El4090EA2Cfg,
    El4090EA2CfgPPO,
)
from legged_gym.envs.el_4090.envelope_adaptive_2 import _contracts as c


def test_env_cfg_contract():
    cfg = El4090EA2Cfg()
    assert cfg.env.num_observations == 190
    assert cfg.env.num_actions == 5
    assert cfg.env.num_privileged_obs is None
    assert cfg.env.episode_length_s == 30.0
    assert cfg.sim.dt == 0.02
    assert cfg.height.min_m == 0.52
    assert cfg.height.max_m == 0.52
    assert cfg.path.speed_range == [1.0, 1.0]
    assert cfg.path.delta_target_deg_range == [0.0, 0.0]
    assert cfg.path.noise_amp_range == [0.0, 0.0]
    assert cfg.sway.pos_amp_range == [0.0, 0.0]
    assert cfg.sway.heading_amp_range == [0.0, 0.0]
    assert cfg.height.wobble_amp_range == [0.0, 0.0]
    assert cfg.path.omega_max == 1.5
    assert cfg.map.size_m == 74.0
    assert cfg.map.grid_shape == [740, 740]
    assert cfg.map.n_tiles == 4
    assert cfg.map.tile_size_m == 16.0
    assert cfg.obstacles.pillar_count_min == 18
    assert cfg.obstacles.pillar_count_max == 18
    assert cfg.obstacles.pillar_size_x_min == 0.5
    assert cfg.obstacles.pillar_size_x_max == 4.0
    assert cfg.obstacles.pillar_min_separation == 2.6
    assert cfg.obstacles.pillar_center_clear_radius == 1.0
    assert cfg.lidar.offset_pos == [0.7, 0.0, -0.05]
    assert cfg.lidar.far_plane == 60.0
    assert cfg.lidar.effective_max_range == 3.2
    assert cfg.lidar.airy_horizontal_resolution_deg == 0.4
    assert cfg.lidar.use_reduced_raycast is True
    assert cfg.lidar.update_frequency_hz == 10.0
    assert cfg.lidar.enable_sensor_noise is True
    assert cfg.lidar.pixel_std_dev_multiplier == 0.02
    assert cfg.lidar.pixel_dropout_prob == 0.02
    assert cfg.envelope.margin == 0.10
    assert cfg.envelope.soft_margin == 0.10
    assert cfg.envelope.action_max == 4.0
    assert cfg.envelope.soft_dof_pos_limit == 0.9
    assert cfg.envelope.oracle_margin == 0.10
    # the env reads this via getattr(..., True); pin it so a rename/delete of
    # the key cannot silently flip the reward oracle back to nearest-cell mode
    assert cfg.envelope.oracle_interp_crossing is True
    assert cfg.envelope.oracle_step == 0.05
    assert cfg.envelope.oracle_max_dist == 5.0
    assert cfg.rewards.scales.collision == 0.0
    assert cfg.rewards.scales.envelope_limits == -0.8
    assert cfg.rewards.scales.oracle_mse == -3.0


def test_ppo_cfg_contract():
    cfg = El4090EA2CfgPPO()
    assert cfg.runner.policy_class_name == "ActorCriticRecurrent"
    assert cfg.runner.algorithm_class_name == "PPO"
    assert cfg.runner.num_steps_per_env == 24
    assert cfg.policy.rnn_type == "gru"
    assert cfg.policy.rnn_hidden_dim == 187
    assert cfg.policy.actor_hidden_dims == [256, 128]
    assert cfg.policy.critic_hidden_dims == [256, 128]
    # num_actor_obs / num_critic_obs must NOT be declared in policy cfg
    assert not hasattr(cfg.policy, "num_actor_obs")
    assert not hasattr(cfg.policy, "num_critic_obs")
    assert cfg.algorithm.entropy_coef == 0.01


def test_airy_fixed_channel_constants():
    assert c.EA2_AIRY_N_AZIMUTH_FULL == 900
    assert c.EA2_AIRY_N_ELEVATION == 96
    assert c.EA2_AIRY_HORIZONTAL_RES_DEG == 0.4
    assert c.EA2_FULL_N_RAYS == 86400
    assert c.EA2_GRID_ROWS == 11
    assert c.EA2_GRID_COLS == 17
    assert c.EA2_RANGE_DIM == 187
    assert c.EA2_RANGE_MAX_M == 3.2
    assert c.EA2_SENSOR_OFFSET_POS == (0.7, 0.0, -0.05)
    assert abs(c.EA2_SENSOR_OFFSET_RPY[1] - (3.14159265 / 2.0 + 0.1)) < 1e-6
    assert c.EA2_MAP_SIZE_M == 74.0
    assert c.EA2_GRID_SHAPE == (740, 740)
    assert c.EA2_N_TILES == 4
    assert c.EA2_SELECTED_CHANNELS_FILE.exists()


def test_envelope_spec_source_exists():
    assert c.ENVELOPE_SPEC_CONFIG_PATH.exists()
