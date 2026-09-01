#!/usr/bin/env python
"""Offline extreme-terrain policy validation for EA2.

This script does NOT modify training.  It builds synthetic 2D distance fields
(open / narrow / front / rear), synthesizes the 187-channel range image from
the fixed selected ray directions, and runs the loaded policy to see whether
the mapped 5-parameter envelope actually changes across those terrains.

Usage (from legged_gym/legged_gym):
    python tests/ea2/validate_ea2_policy_extreme.py \
        --task=el4090_ea2 --headless \
        --load_run <run_dir> --checkpoint <iter>
"""

from __future__ import annotations

import isaacgym  # noqa: F401

import numpy as np
import torch
from isaacgym.torch_utils import quat_apply, quat_from_euler_xyz, quat_mul
from scipy import ndimage

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.envs.el_4090.envelope_adaptive_2.airy_mount import (
    load_selected_channels,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_config import (
    El4090EA2Cfg,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_env import (
    assemble_observation,
    map_actions_to_params,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_oracle import (
    compute_direct_oracle_params_with_stats,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.range_image import (
    build_selected_range_image,
)
from legged_gym.utils import get_args, task_registry

try:  # pytest: package ``ea2``
    from . import _ea2_testlib as tl
except ImportError:  # direct script execution
    import _ea2_testlib as tl

_LOW, _HIGH = tl.LOW, tl.HIGH
_WORLD_MIN, _RES, _SIZE = tl.WORLD_MIN, tl.RES, tl.SIZE
_RAY_MIN_T = 0.05
_RAY_STEP = 0.1


def _corridor_field(half_width: float) -> np.ndarray:
    return tl.corridor_field(half_width)[0]


def _point_field(front=0.0, rear=0.0, left=0.0, right=0.0) -> np.ndarray:
    pillars = []
    if front:
        pillars.append((front, 0.0))
    if rear:
        pillars.append((rear, 0.0))
    if left:
        pillars.append((0.0, left))
    if right:
        pillars.append((0.0, right))
    return tl.point_field(pillars)[0]


def _sensor_setup(cfg):
    offset_pos = list(cfg.lidar.offset_pos)
    rpy = list(cfg.lidar.sensor_offset_rpy)
    offset_quat = quat_from_euler_xyz(
        torch.tensor(float(rpy[0])),
        torch.tensor(float(rpy[1])),
        torch.tensor(float(rpy[2])),
    ).view(1, 4)
    translation = torch.tensor(offset_pos, dtype=torch.float32).view(1, 3)
    return translation, offset_quat


def _synthetic_range_image(
    df: np.ndarray,
    base_xy: tuple[float, float],
    heading: float,
    ray_dirs: torch.Tensor,
    range_max: float,
    sensor_translation: torch.Tensor,
    sensor_offset_quat: torch.Tensor,
) -> torch.Tensor:
    # World sensor pose (same as env._update_lidar, ignoring z for 2D field).
    yaw_quat = torch.tensor(
        [np.cos(heading / 2.0), 0.0, 0.0, np.sin(heading / 2.0)],
        dtype=torch.float32,
    ).view(1, 4)
    sensor_quat = quat_mul(yaw_quat, sensor_offset_quat)
    sensor_pos = torch.tensor(
        [base_xy[0], base_xy[1], 0.0], dtype=torch.float32
    ).view(1, 3) + quat_apply(yaw_quat, sensor_translation)

    world_dirs = quat_apply(
        sensor_quat, ray_dirs.unsqueeze(0)
    ).squeeze(0)  # (187,3)

    # Use horizontal projection for the 2D distance field.
    dir_xy = world_dirs[:, :2]
    norm = torch.norm(dir_xy, dim=-1, keepdim=True).clamp_min(1e-6)
    dir_xy = dir_xy / norm

    dists = torch.full((1, ray_dirs.shape[0]), range_max, dtype=torch.float32)
    sensor_xy = sensor_pos[0, :2].detach().cpu().numpy()
    for t in np.arange(_RAY_MIN_T, range_max + 1e-6, _RAY_STEP):
        pts_xy = (
            sensor_xy[None, :]
            + t * dir_xy.detach().cpu().numpy()
        )
        ix = np.floor((pts_xy[:, 0] - _WORLD_MIN) / _RES).astype(np.int64)
        iy = np.floor((pts_xy[:, 1] - _WORLD_MIN) / _RES).astype(np.int64)
        ix = np.clip(ix, 0, _SIZE - 1)
        iy = np.clip(iy, 0, _SIZE - 1)
        clearance = df[iy, ix]
        hit = clearance < 0.05
        # first hit
        first = hit & (dists[0].numpy() == range_max)
        dists[0, first] = t
    return build_selected_range_image(dists, range_max)


def main() -> None:
    args = get_args()
    sc = load_selected_channels()
    ray_dirs = sc["ray_directions"].float()
    range_max = float(sc["max_range"])

    cfg = El4090EA2Cfg()
    sensor_translation, sensor_offset_quat = _sensor_setup(cfg)
    low = _LOW.to(torch.float32)
    high = _HIGH.to(torch.float32)

    # Create one env + policy loader (no training modification).
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    if args.checkpoint is not None:
        train_cfg.runner.checkpoint = args.checkpoint
    ppo_runner, _ = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    scenarios = {
        "open 2.0m": (_corridor_field(2.0), (0.0, 0.0), 0.0),
        "corridor 0.65m": (_corridor_field(0.65), (0.0, 0.0), 0.0),
        "corridor 0.45m": (_corridor_field(0.45), (0.0, 0.0), 0.0),
        "corridor 0.35m": (_corridor_field(0.35), (0.0, 0.0), 0.0),
        "front 0.85": (_point_field(front=0.85), (0.0, 0.0), 0.0),
        "rear -0.85": (_point_field(rear=-0.85), (0.0, 0.0), 0.0),
    }

    param_names = [
        "front_width",
        "middle_width",
        "back_width",
        "forward_limit",
        "backward_limit",
    ]

    print("\n=== extreme-scenario policy validation ===")
    for name, (df, xy, yaw) in scenarios.items():
        obs = _synthetic_range_image(
            df, xy, yaw, ray_dirs, range_max,
            sensor_translation, sensor_offset_quat,
        )
        ego = torch.zeros(1, 3, dtype=torch.float32)
        obs_full = assemble_observation(obs, ego, max_range=range_max)
        ppo_runner.alg.policy.reset()
        action = policy(obs_full.detach().to(env.device))
        params = map_actions_to_params(
            action.detach().cpu(),
            low,
            high,
            float(cfg.envelope.soft_dof_pos_limit),
            float(cfg.envelope.action_max),
        )[0].numpy()

        oracle, _ = compute_direct_oracle_params_with_stats(
            torch.tensor([yaw], dtype=torch.float32),
            torch.tensor([[xy[0], xy[1]]], dtype=torch.float32),
            df,
            low,
            high,
            margin=float(cfg.envelope.oracle_margin),
            step=float(cfg.envelope.oracle_step),
            max_dist=float(cfg.envelope.oracle_max_dist),
            soft_dof_pos_limit=float(cfg.envelope.soft_dof_pos_limit),
            interp_crossing=bool(
                getattr(cfg.envelope, "oracle_interp_crossing", True)
            ),
        )
        print(
            f"{name:18s} policy="
            + " ".join(f"{n}={params[i]:.4f}" for i, n in enumerate(param_names))
            + "   oracle="
            + " ".join(f"{n}={oracle[0, i]:.4f}" for i, n in enumerate(param_names))
        )


if __name__ == "__main__":
    main()
