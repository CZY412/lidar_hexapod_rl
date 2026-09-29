"""Interactive playback for direct random HAA ranges in ``el4090_envelop_3``."""

import os
import select
import sys
import termios
import time
import tty

import isaacgym  # noqa: F401 -- must precede torch-backed environment imports.
import torch

from legged_gym.envs import *  # noqa: F401,F403 -- register tasks.
from legged_gym.utils import get_args, task_registry


TASK_NAME = "el4090_envelop_3"
EXPECTED_OBS_DIM = 79

# Three entries map to the coupled LB/RB, LF/RF, LM/RM pairs.  The first four
# make endpoint directions easy to inspect; the final full-width box is a
# deliberate stress case and may self-contact, which is useful for validation.
EXTREME_HAA_PRESETS = (
    ("neutral narrow", ((-0.15, 0.15), (-0.15, 0.15), (-0.15, 0.15))),
    ("all negative edge", ((-1.30, -0.95), (-1.30, -0.95), (-1.30, -0.95))),
    ("all positive edge", ((0.95, 1.30), (0.95, 1.30), (0.95, 1.30))),
    ("alternating edges", ((-1.30, -0.95), (0.95, 1.30), (-0.15, 0.15))),
    ("full mechanical span (stress)", ((-1.30, 1.30), (-1.30, 1.30), (-1.30, 1.30))),
)


class KeyboardInput:
    """Non-blocking single-key terminal input for interactive playback."""

    def __enter__(self):
        self.fd = None
        self.old_settings = None
        if sys.stdin.isatty():
            self.fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.fd is not None and self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSANOW, self.old_settings)

    def read_keys(self):
        if self.fd is None:
            return []
        keys = []
        while select.select([self.fd], [], [], 0.0)[0]:
            key = os.read(self.fd, 1).decode(errors="ignore")
            if key:
                keys.append(key.lower())
        return keys


def _print_ranges(env, label: str) -> None:
    ranges = env.haa_swing_ranges[0].detach().cpu()
    text = "  ".join(
        f"{env.dof_names[dof]}=[{ranges[leg, 0]:.3f}, {ranges[leg, 1]:.3f}]"
        for leg, dof in enumerate(env.haa_indices.tolist())
    )
    leg_name = str(env.cfg.commands.haa_range_debug_leg)
    leg_index = env.haa_range_state.leg_names.index(leg_name)
    dof_index = env.haa_indices.tolist()[leg_index]
    current = env.dof_pos[0, dof_index].item()
    print(
        f"\n[{label}] target_height={env.cfg.rewards.target_height}\n  {text}\n"
        f"  viewer highlight: {leg_name}_HAA=[{ranges[leg_index, 0]:.3f}, {ranges[leg_index, 1]:.3f}], "
        f"current={current:.3f} (dark-green arc, red marker)"
    )


def _set_extreme_preset(env, preset_index: int) -> None:
    """Apply one coupled HAA stress preset to every playback environment."""
    name, pair_ranges = EXTREME_HAA_PRESETS[preset_index]
    ranges = torch.tensor(pair_ranges, dtype=torch.float, device=env.device)
    pair_to_leg = torch.tensor((0, 1, 2, 0, 1, 2), dtype=torch.long, device=env.device)
    ranges = ranges[pair_to_leg].unsqueeze(0).repeat(env.num_envs, 1, 1)
    env.set_haa_ranges(ranges)
    env.compute_observations()
    _print_ranges(env, f"extreme preset {preset_index + 1}/{len(EXTREME_HAA_PRESETS)}: {name}")


def _print_help() -> None:
    print(
        "\nControls: W/S forward/back, A/D lateral, Q/E yaw, X stop, "
        "R resample direct HAA ranges, T cycle extreme HAA presets, "
        "P print ranges, Esc quit."
    )


def main() -> None:
    args = get_args([{
        "name": "--max_steps",
        "type": int,
        "default": 1000000,
        "help": "Maximum playback steps; use Esc to stop interactive playback.",
    }, {
        "name": "--haa_debug_leg",
        "type": str,
        "default": "LF",
        "help": "Leg whose HAA interval is drawn in the viewer (LB/LF/LM/RB/RF/RM).",
    }])
    args.task = TASK_NAME
    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 1e9
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.curriculum = False
    # In v3 this inherited hook draws the selected HAA interval rather than
    # the removed five-parameter envelope polygon.
    env_cfg.commands.envelope_debug_viz = True
    env_cfg.commands.haa_range_debug_leg = args.haa_debug_leg.upper()
    # R controls sampling explicitly, so intervals do not change unattended.
    env_cfg.commands.resampling_time = 1e9
    env, _ = task_registry.make_env(name=TASK_NAME, args=args, env_cfg=env_cfg)
    env.commands[:, :3] = torch.tensor([0.0, 0.0, 0.0], device=env.device)
    env.sample_haa_ranges()
    env.compute_observations()
    obs = env.get_observations()
    if obs.shape[-1] != EXPECTED_OBS_DIM:
        raise RuntimeError(f"Expected {EXPECTED_OBS_DIM} observations, got {obs.shape[-1]}")

    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(env=env, name=TASK_NAME, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)
    command = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    extreme_preset_index = -1
    _print_help()
    _print_ranges(env, "initial direct sample")

    keyboard = KeyboardInput()
    keyboard.__enter__()
    try:
        if keyboard.fd is None:
            print("stdin is not a TTY; running until --max_steps.")
        for step in range(args.max_steps):
            step_start = time.time()
            stop = False
            for key in keyboard.read_keys():
                if key == "\x1b":
                    stop = True
                elif key == "w":
                    command["x"] = 1.2
                elif key == "s":
                    command["x"] = -1.2
                elif key == "a":
                    command["y"] = 0.8
                elif key == "d":
                    command["y"] = -0.8
                elif key == "q":
                    command["yaw"] = 0.8
                elif key == "e":
                    command["yaw"] = -0.8
                elif key == "x":
                    command = {"x": 0.0, "y": 0.0, "yaw": 0.0}
                elif key == "r":
                    env.sample_haa_ranges()
                    env.compute_observations()
                    obs = env.get_observations()
                    _print_ranges(env, "new direct sample")
                elif key == "t":
                    extreme_preset_index = (extreme_preset_index + 1) % len(EXTREME_HAA_PRESETS)
                    _set_extreme_preset(env, extreme_preset_index)
                    obs = env.get_observations()
                elif key == "p":
                    _print_ranges(env, "current sample")
            if stop:
                break

            env.commands[:, 0] = command["x"]
            env.commands[:, 1] = command["y"]
            env.commands[:, 2] = command["yaw"]
            obs, _, _, dones, _ = env.step(policy(obs.detach()).detach())
            if torch.any(dones):
                _print_ranges(env, f"resampled by reset at step {step}")
            if not args.headless:
                sleep_time = env.dt - (time.time() - step_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        keyboard.__exit__(None, None, None)


if __name__ == "__main__":
    main()
