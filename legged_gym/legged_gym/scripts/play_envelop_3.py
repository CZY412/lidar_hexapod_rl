"""Interactive playback for v3's five-parameter envelope/range pipeline."""

import os
import select
import sys
import termios
import time
import tty
from typing import Optional

import isaacgym  # noqa: F401 -- must precede torch-backed environment imports.
from isaacgym import gymtorch
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403 -- register tasks.
from legged_gym.utils import export_policy_as_jit, get_args, task_registry
from legged_gym.utils.envelop.train.train_api_haa_range import (
    ApiRangeLabeler,
    NoFeasibleHaaEnvelopeError,
)


TASK_NAME = "el4090_envelop_3"
EXPECTED_OBS_DIM = 79
# Keep v3 aligned with the v1/v2 playback workflow: loading a checkpoint in
# play also produces the standalone TorchScript actor used by deployment.
EXPORT_POLICY = True
# These are deliberately corners of the range-network input box rather than
# hand-written HAA intervals.  Thus every case exercises the exact same
# envelope -> network -> HAA-range path as v3 training.
EXTREME_ENVELOPE_CASES = (
    ("narrow_long", {
        "front_width": "low", "middle_width": "low", "back_width": "low",
        "forward_limit": "high", "backward_limit": "low",
    }),
    ("wide_short", {
        "front_width": "high", "middle_width": "high", "back_width": "high",
        "forward_limit": "low", "backward_limit": "high",
    }),
    ("narrow_forward", {
        "front_width": "low", "middle_width": "low", "back_width": "low",
        "forward_limit": "high", "backward_limit": "high",
    }),
    ("narrow_backward", {
        "front_width": "low", "middle_width": "low", "back_width": "low",
        "forward_limit": "low", "backward_limit": "low",
    }),
    ("wide_forward", {
        "front_width": "high", "middle_width": "high", "back_width": "high",
        "forward_limit": "high", "backward_limit": "high",
    }),
    ("wide_backward", {
        "front_width": "high", "middle_width": "high", "back_width": "high",
        "forward_limit": "low", "backward_limit": "low",
    }),
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
    range_text = "  ".join(
        f"{env.dof_names[dof]}=[{ranges[leg, 0]:.3f}, {ranges[leg, 1]:.3f}]"
        for leg, dof in enumerate(env.haa_indices.tolist())
    )
    condition = env.envelope_parameters[0].detach().cpu()
    condition_text = ", ".join(
        f"{name}={value:.3f}"
        for name, value in zip(env.haa_range_state.condition_names, condition)
    )
    haa_bias = env.haa_range_bias[0, env.haa_indices].detach().cpu()
    leg_name = str(env.cfg.commands.haa_range_debug_leg)
    leg_index = env.haa_range_state.leg_names.index(leg_name)
    dof_index = env.haa_indices.tolist()[leg_index]
    current = env.dof_pos[0, dof_index].item()
    print(
        f"\n[{label}] target_height={env.cfg.rewards.target_height}\n"
        f"  envelope: {condition_text}\n"
        f"  HAA ranges: {range_text}\n"
        f"  HAA centre bias: "
        f"{', '.join(f'{value:.3f}' for value in haa_bias.tolist())}\n"
        f"  viewer highlight: {leg_name}_HAA=[{ranges[leg_index, 0]:.3f}, {ranges[leg_index, 1]:.3f}], "
        f"current={current:.3f} (dark-green arc, red marker)"
    )


def _print_help() -> None:
    print(
        "\nControls: W/S forward/back, A/D lateral, Q/E yaw, X stop, "
        "R resample five envelope parameters and infer HAA ranges, "
        "T next extreme case, N next extreme case, B previous extreme case, "
        "P print envelope/ranges/bias, Esc quit."
    )


def _extreme_envelope_condition(env, bounds) -> torch.Tensor:
    """Return one reproducible in-domain extreme envelope preset."""
    state = env.haa_range_state
    condition = 0.5 * (state.condition_low + state.condition_high)
    condition = condition.unsqueeze(0).repeat(env.num_envs, 1)
    for name, bound_name in bounds.items():
        condition[:, state.condition_names.index(name)] = getattr(state, f"condition_{bound_name}")[
            state.condition_names.index(name)
        ]
    return condition


def _apply_envelope_condition(env, condition: torch.Tensor) -> torch.Tensor:
    """Apply network-derived HAA ranges and their matching posture bias."""
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    ranges = env.haa_range_state.set_conditions(condition, env_ids)
    env._refresh_haa_range_bias(env_ids)
    env.embedded_state_default_dof_pos[env_ids] = env._get_condition_target_dof_pos()[env_ids]
    return ranges


def _validate_envelope_condition(labeler, condition: torch.Tensor) -> Optional[torch.Tensor]:
    """Run the exact rejection-range gate used to create the training bank."""
    try:
        return labeler.label(condition[0].detach().cpu())
    except NoFeasibleHaaEnvelopeError as error:
        values = ", ".join(f"{value:.3f}" for value in condition[0].detach().cpu().tolist())
        print(f"[ENVELOPE NONE] [{values}] is not runnable: {error}. Selection was not applied.")
        return None


def _valid_extreme_cases(env, labeler):
    """Keep only corners accepted by the exact rejection-range algorithm."""
    state = env.haa_range_state
    valid_cases = []
    for name, bounds in EXTREME_ENVELOPE_CASES:
        condition = _extreme_envelope_condition(env, bounds)
        if _validate_envelope_condition(labeler, condition) is not None:
            valid_cases.append((name, bounds))
        else:
            print(f"Skipping NONE extreme envelope case: {name}")
    if not valid_cases:
        raise RuntimeError("The HAA-range network rejected every extreme envelope case")
    return tuple(valid_cases)


def _reset_to_envelope_condition(env, condition: torch.Tensor, command) -> torch.Tensor:
    """Reset then restore a fixed envelope and its matching biased pose.

    ``EL_4090_ENVELOP_3.reset_idx`` intentionally samples an envelope for
    training.  Playback needs the reset mechanics but must replace that random
    draw *after* reset, otherwise a fall would silently change the case under
    inspection.
    """
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.reset_idx(env_ids)
    _apply_envelope_condition(env, condition)
    target = env._get_condition_target_dof_pos()[env_ids]
    env.dof_pos[env_ids] = target
    env.dof_vel[env_ids] = 0.0
    env.last_actions[env_ids] = 0.0
    env.actions[env_ids] = 0.0
    env_ids_int32 = env_ids.to(dtype=torch.int32)
    env.gym.set_dof_state_tensor_indexed(
        env.sim, gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32),
    )
    # reset_idx resamples commands as it should during training.  The playback
    # policy must instead see the command selected from the keyboard on the
    # very first frame of every extreme case.
    env.commands[:, 0] = command["x"]
    env.commands[:, 1] = command["y"]
    env.commands[:, 2] = command["yaw"]
    env.compute_observations()
    return env.get_observations()


def main() -> None:
    args = get_args([{
        "name": "--max_steps",
        "type": int,
        "default": 1000000,
        "help": "Maximum playback steps; use Esc to stop interactive playback.",
    }, {
        "name": "--envelope",
        "type": float,
        "nargs": 5,
        "default": None,
        "metavar": ("FRONT_W", "MIDDLE_W", "BACK_W", "FORWARD_X", "BACKWARD_X"),
        "help": "Apply one explicit five-parameter envelope after exact rejection-range validation.",
    }, {
        "name": "--haa_debug_leg",
        "type": str,
        "default": "LF",
        "help": "Leg whose HAA interval is drawn in the viewer (LB/LF/LM/RB/RF/RM).",
    }, {
        "name": "--extreme_case_hold_s",
        "type": float,
        "default": 0.0,
        "help": "Optional seconds to reset-and-rotate extreme cases automatically; default 0 keeps switching on T only.",
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
    # R controls the same free five-parameter sampling as training, so ranges
    # do not change unattended during playback.  Collision is intentionally
    # not evaluated here, matching the current v3 training path.
    env_cfg.commands.resampling_time = 1e9
    env, _ = task_registry.make_env(name=TASK_NAME, args=args, env_cfg=env_cfg)
    labeler = ApiRangeLabeler(directions=32, margin=.05, tolerance=1e-6, sweep_steps=201)
    command = {"x": 0.0, "y": 0.0, "yaw": 0.0}
    extreme_cases = _valid_extreme_cases(env, labeler)
    case_index = 0
    auto_cycle = args.extreme_case_hold_s > 0.0
    hold_steps = max(1, round(args.extreme_case_hold_s / env.dt)) if auto_cycle else None
    case_name, case_bounds = extreme_cases[case_index]
    initial_condition = _extreme_envelope_condition(env, case_bounds)
    if args.envelope is not None:
        initial_condition = torch.tensor(args.envelope, dtype=torch.float, device=env.device).view(1, 5)
        initial_condition = initial_condition.repeat(env.num_envs, 1)
        if _validate_envelope_condition(labeler, initial_condition) is None:
            raise ValueError("--envelope is NONE; choose a rejection-range-valid five-vector")
        case_name = "explicit validated envelope"
    obs = _reset_to_envelope_condition(env, initial_condition, command)
    if obs.shape[-1] != EXPECTED_OBS_DIM:
        raise RuntimeError(f"Expected {EXPECTED_OBS_DIM} observations, got {obs.shape[-1]}")

    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(env=env, name=TASK_NAME, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)
    if EXPORT_POLICY:
        export_path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(runner.alg.policy, export_path)
        print("Exported policy as jit script to:", export_path)
    _print_help()
    _print_ranges(env, f"initial extreme case 1/{len(extreme_cases)}: {case_name}")

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
                    _print_ranges(env, "new envelope/network sample")
                elif key in ("t", "n", "b"):
                    direction = -1 if key == "b" else 1
                    case_index = (case_index + direction) % len(extreme_cases)
                    case_name, case_bounds = extreme_cases[case_index]
                    obs = _reset_to_envelope_condition(env, _extreme_envelope_condition(env, case_bounds), command)
                    _print_ranges(env, f"extreme case {case_index + 1}/{len(extreme_cases)}: {case_name}")
                elif key == "p":
                    _print_ranges(env, "current sample")
            if stop:
                break

            if auto_cycle and step > 0 and step % hold_steps == 0:
                case_index = (case_index + 1) % len(extreme_cases)
                case_name, case_bounds = extreme_cases[case_index]
                obs = _reset_to_envelope_condition(env, _extreme_envelope_condition(env, case_bounds), command)
                _print_ranges(env, f"automatic extreme case {case_index + 1}/{len(extreme_cases)}: {case_name}")

            env.commands[:, 0] = command["x"]
            env.commands[:, 1] = command["y"]
            env.commands[:, 2] = command["yaw"]
            obs, _, _, dones, _ = env.step(policy(obs.detach()).detach())
            if torch.any(dones):
                # The environment samples on reset during training.  Restore
                # the currently selected playback case instead.
                obs = _reset_to_envelope_condition(env, _extreme_envelope_condition(env, case_bounds), command)
                _print_ranges(env, f"restored extreme case after reset at step {step}: {case_name}")
            if not args.headless:
                sleep_time = env.dt - (time.time() - step_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
    finally:
        keyboard.__exit__(None, None, None)


if __name__ == "__main__":
    main()
