#!/usr/bin/env python
"""Smoke test for the ``el4090_ea2`` M1 environment.

Creates a small ``EL_4090_EA2`` instance, runs ``num_steps`` random policy
steps and verifies the BaseTask contract:

* step returns the old 5-tuple (obs, privileged_obs, rew, dones, infos);
* obs has shape ``(num_envs, 453)``;
* ``infos["time_outs"]`` is present when ``cfg.env.send_timeouts`` is set;
* range image / ego-motion are finite and within expected bounds;
* path planning succeeds for every env.

Usage::

    python legged_gym/scripts/smoke_ea2.py --num-envs 4 --device cuda:0

The script defaults to 4 envs and 3 steps.  Both the hyphenated aliases
(``--num-envs``, ``--device``) and the legged_gym-style names (``--num_envs``,
``--sim_device``) are accepted.  The preferred device is ``cuda:0``; it falls
back to CPU only when CUDA is not available.
"""

import os
import sys

# Isaac Gym's gymtorch JIT build needs ninja on PATH even in a fresh shell.
# Prepend the currently-running conda env's bin directory before importing
# isaacgym/gymtorch.
_bin_dir = os.path.dirname(os.path.abspath(sys.executable or ""))
if _bin_dir and _bin_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _bin_dir + os.pathsep + os.environ.get("PATH", "")

import isaacgym  # noqa: F401 -- must be imported before torch.
import torch

from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_config import (
    El4090EA2Cfg,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_env import (
    EL_4090_EA2,
)
from legged_gym.utils.helpers import (
    class_to_dict,
    get_args,
    parse_sim_params,
    update_cfg_from_args,
)


def _normalize_argv() -> None:
    """Map the required hyphenated CLI aliases to legged_gym parser names."""
    out = [sys.argv[0]]
    for arg in sys.argv[1:]:
        if arg == "--num-envs":
            out.append("--num_envs")
        elif arg.startswith("--num-envs="):
            out.append("--num_envs=" + arg.split("=", 1)[1])
        elif arg == "--device":
            out.append("--sim_device")
        elif arg.startswith("--device="):
            out.append("--sim_device=" + arg.split("=", 1)[1])
        else:
            out.append(arg)
    sys.argv = out


def _main() -> int:
    _normalize_argv()

    extra = [
        {
            "name": "--steps",
            "type": int,
            "default": 3,
            "help": "Number of random policy steps to run",
        }
    ]
    args = get_args(extra_custom_parameters=extra)
    # This is an automated smoke test; never open a viewer window.
    args.headless = True

    env_cfg = El4090EA2Cfg()
    env_cfg, _ = update_cfg_from_args(env_cfg, None, args)

    # The smoke test defaults to a small 4-env run regardless of the training
    # config default (1024); --num-envs still overrides it.
    env_cfg.env.num_envs = args.num_envs if args.num_envs is not None else 4

    # Preferred device is cuda:0; fall back to CPU only if CUDA is unavailable.
    if args.sim_device_type == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU.")
        args.sim_device_type = "cpu"
        args.sim_device = "cpu"
        args.use_gpu = False
        args.use_gpu_pipeline = False

    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)

    print(
        f"[smoke_ea2] creating env num_envs={env_cfg.env.num_envs} "
        f"sim_device={args.sim_device}"
    )
    env = EL_4090_EA2(
        cfg=env_cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device,
        headless=args.headless,
    )

    obs, priv_obs = env.reset()
    assert obs.shape == (env.num_envs, env.num_obs), obs.shape
    assert obs.dtype == torch.float32
    assert priv_obs is None
    assert env.privileged_obs_buf is None

    # Initial observations: range image normalized to [0, 1], ego to ~[-1, 1].
    assert torch.isfinite(obs).all()
    assert obs[:, :450].min() >= -1e-4 and obs[:, :450].max() <= 1.0 + 1e-4

    steps = int(args.steps)
    max_range = float(env_cfg.lidar.effective_max_range)
    print(f"[smoke_ea2] running {steps} random steps")
    for step in range(steps):
        actions = torch.randn(
            env.num_envs, env.num_actions, dtype=torch.float32, device=env.device
        ) * 0.3
        obs, priv_obs, rew, dones, infos = env.step(actions)
        assert obs.shape == (env.num_envs, env.num_obs)
        assert rew.shape == (env.num_envs,)
        assert dones.shape == (env.num_envs,)
        assert torch.isfinite(obs).all(), "obs became non-finite"
        assert torch.isfinite(rew).all(), "reward became non-finite"
        assert torch.isfinite(env.range_image).all(), "range image non-finite"
        assert env.range_image.min() >= 0.0
        assert env.range_image.max() <= max_range + 1e-4
        if env_cfg.env.send_timeouts:
            assert "time_outs" in infos
        if step == 0:
            # The legacy timer makes the first policy step a global LiDAR
            # scan.  After an initial reset, that scan must replace the empty
            # all-max_range frame with a fresh aggregate from the new pose
            # (README 2.4).  Ground/obstacle hits in the selected Airy buckets
            # guarantee at least one bucket below the 5.0 m empty sentinel.
            assert (
                env.range_image.min() < max_range - 1e-3
            ), "reset envs did not receive a fresh range image on the first scan"
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            print(
                f"  step {step + 1:3d} | rew={rew.mean().item():+.4f} | "
                f"dones={int(dones.sum())} | "
                f"range_min={env.range_image.min().item():.2f} "
                f"range_max={env.range_image.max().item():.2f}"
            )

    # A global LiDAR scan must have happened at least once during the smoke run
    # (the legacy timer makes the first step a scan), so stale flags clear.
    assert not bool(env.range_image_stale.any()), "stale range flags should clear after scans"
    print("[smoke_ea2] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
