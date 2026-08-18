"""T0 sanity checks: frozen contracts and config match README v2 numbers."""

from legged_gym.envs.el_4090.envelope_adaptive_2 import (
    El4090EA2Cfg,
    El4090EA2CfgPPO,
)
from legged_gym.envs.el_4090.envelope_adaptive_2 import _contracts as c


def test_env_cfg_contract():
    cfg = El4090EA2Cfg()
    assert cfg.env.num_observations == 453
    assert cfg.env.num_actions == 5
    assert cfg.env.num_privileged_obs is None
    assert cfg.env.episode_length_s == 20.0
    assert cfg.sim.dt == 0.02
    assert cfg.height.min_m == 0.53
    assert cfg.height.max_m == 0.64
    assert cfg.path.speed_range == [0.5, 1.5]
    assert cfg.path.delta_target_deg_range == [-20.0, 20.0]
    assert cfg.path.omega_max == 1.5
    assert cfg.lidar.far_plane == 60.0
    assert cfg.lidar.effective_max_range == 5.0
    assert cfg.lidar.update_frequency_hz == 10.0
    assert cfg.lidar.enable_sensor_noise is True
    assert cfg.lidar.pixel_std_dev_multiplier == 0.02
    assert cfg.lidar.pixel_dropout_prob == 0.02


def test_ppo_cfg_contract():
    cfg = El4090EA2CfgPPO()
    assert cfg.runner.policy_class_name == "ActorCriticRecurrent"
    assert cfg.runner.algorithm_class_name == "PPO"
    assert cfg.runner.num_steps_per_env == 50
    assert cfg.policy.rnn_type == "gru"
    assert cfg.policy.rnn_hidden_dim == 187
    assert cfg.policy.actor_hidden_dims == [256, 128]
    assert cfg.policy.critic_hidden_dims == [256, 128]
    # num_actor_obs / num_critic_obs must NOT be declared in policy cfg
    assert not hasattr(cfg.policy, "num_actor_obs")
    assert not hasattr(cfg.policy, "num_critic_obs")


def test_airy_bucket_constants():
    assert c.EA2_N_RAYS == 5760
    assert c.EA2_SELECTED_AZ == tuple(range(18, 43))
    assert c.EA2_SELECTED_EL == tuple(range(6, 96))
    assert c.EA2_N_COLS == 25
    assert c.EA2_N_ROWS == 18
    assert c.EA2_RANGE_DIM == 450
    assert c.EA2_SENSOR_OFFSET_POS == (0.62, 0.0, 0.0)
    assert abs(c.EA2_SENSOR_OFFSET_RPY[1] - (3.14159265 / 2.0 + 0.35)) < 1e-6


def test_envelope_spec_source_exists():
    assert c.ENVELOPE_SPEC_CONFIG_PATH.exists()
