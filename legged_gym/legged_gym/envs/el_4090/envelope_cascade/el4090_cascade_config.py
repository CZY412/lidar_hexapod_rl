"""Configuration for the ``el4090_cascade`` demo task.

Inherits the full SE2 locomotion config (83-dim MLP policy, HAA range
network, P control) and adds an ``ea2`` namespace holding everything the
merged EA2 perception/policy needs.  The EA2 action-mapping constants are
referenced from the live EA2 config so a single source of truth governs the
fold scale; ``ea2.fold_scale`` pins the value the checkpoint was exported
with and is asserted at env construction (see checkpoints/README.md).
"""

import math

from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_config import (
    El4090EA2Cfg,
)
from legged_gym.envs.el_4090.spider_envelop_2.el4090_spider_config import (
    El4090Envelop2Cfg,
    El4090Envelop2CfgPPO,
)


class El4090CascadeCfg(El4090Envelop2Cfg):
    class terrain(El4090Envelop2Cfg.terrain):
        mesh_type = "trimesh"  # warp raycast mesh is built from this terrain
        terrain_length = 16.0
        terrain_width = 16.0
        # Exactly 9 elements REQUIRED: utils/terrain.make_terrain indexes
        # proportions[5..8]; with fewer entries the pillar/channel branches
        # are silently unreachable (flat terrain, no obstacles).
        terrain_proportions = [0.0] * 7 + [0.5, 0.5]  # pillar + channel tiles
        # Obstacle distribution matched to EA2 training maps
        # (El4090EA2Cfg.obstacles) so the perception policy sees in-distribution
        # geometry.  pillar_field_terrain reads these via getattr.
        pillar_count_min = 24
        pillar_count_max = 24
        pillar_size_x_min = 0.5
        pillar_size_x_max = 4.0
        pillar_size_y_min = 0.5
        pillar_size_y_max = 4.0
        pillar_height_min = 1.0
        pillar_height_max = 2.0
        pillar_min_separation = 1.0
        # 出生净空：pillar_field 的 center_clear_radius 是【中心距】语义
        # （柱子中心距 tile 中心 ≥ 此值即可放置），不是净空区！4m×4m 柱的
        # 最坏半对角 ≈2.83m，取 4.0 才能保证柱身边缘距出生点 ≥1.17m
        # （取 2.0 时柱身仍会盖住出生点，机器人被顶上柱顶循环摔落，
        # 已在闭环测试实证：出生高度 1.91m = 柱顶）。
        pillar_center_clear_radius = 2.5
        pillar_spawn_radius = 8.0
        pillar_allow_height_variation = True
        difficulty_scale = 1.0

    class lidar(El4090Envelop2Cfg.lidar):
        # The inherited v1 LiDAR stack (11x17 simple_grid) must stay OFF;
        # the EA2 187-channel perception lives under cfg.ea2.
        enable = False

    class haa_swing_range(El4090Envelop2Cfg.haa_swing_range):
        # Override the inherited pointer: the demo pins its own byte-identical
        # copy (checkpoints/haa_range.pt, md5 640627fd…) so the whole policy
        # set is self-contained.  Keep in sync with
        # spider_envelop_2/envelop_network/haa_range.pt on retrain (see
        # checkpoints/README.md).
        network_checkpoint = (
            "{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/el_4090/"
            "envelope_cascade/checkpoints/haa_range.pt"
        )

    class se2_policy:
        """Pinned SE2 locomotion policy (TorchScript, exported via
        ``export_policy_as_jit``) — consumes 83-dim obs, emits 18 actions."""

        checkpoint = (
            "{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/el_4090/"
            "envelope_cascade/checkpoints/policy_1.pt"
        )

    class ea2:
        # ── module switches ─────────────────────────────────────────────
        enable = True  # False restores pure SE2 behaviour (sampled envelopes)
        debug_stats = False  # per-step telemetry consumed by play/tests
        reset_condition = "max"  # birth pose preset: "max" | "midpoint"
        yaw_only = False  # True = EA2's yaw-only mount (tilt OOD ablation)

        # ── assets ──────────────────────────────────────────────────────
        # v3att：attitude-replay 重训权重（真机俯仰/横滚/高度回放语料，
        # val R²=0.738）；旧 v2_multik 权重保留于 ea2_envelope.pt 可秒回滚
        checkpoint = (
            "{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/el_4090/"
            "envelope_cascade/checkpoints/ea2_envelope_v3att.pt"
        )
        channel_file = (
            "{LEGGED_GYM_ROOT_DIR}/legged_gym/envs/el_4090/"
            "envelope_adaptive_2/selected_airy_channels.pt"
        )

        # ── LiDAR (identical to El4090EA2Cfg.lidar training values) ─────
        update_frequency_hz = 10.0
        far_plane = 60.0
        offset_pos = (0.7, 0.0, -0.05)
        sensor_offset_rpy = (0.0, math.pi / 2.0 + 0.1, 0.0)
        enable_sensor_noise = False  # demo default; True matches training
        pixel_std_dev_multiplier = 0.02
        pixel_dropout_prob = 0.02

        # ── observation ─────────────────────────────────────────────────
        ego_scales = (1.5, 1.0, 1.5)  # [vx, vy, wz] normalisation

        # ── action mapping (must match the pinned checkpoint's fold) ────
        soft_dof_pos_limit = float(El4090EA2Cfg.envelope.soft_dof_pos_limit)
        action_max = float(El4090EA2Cfg.envelope.action_max)
        # k used when this checkpoint was exported; env asserts
        # soft/(2*action_max) == fold_scale so a stale checkpoint cannot be
        # paired silently with a drifted EA2 config (README §8 trap).
        fold_scale = 0.11875


class El4090CascadeCfgPPO(El4090Envelop2CfgPPO):
    """Demo task: training config fully inherited from SE2 so the upcoming
    envelop_2 checkpoint in the shared experiment dir loads unchanged."""
