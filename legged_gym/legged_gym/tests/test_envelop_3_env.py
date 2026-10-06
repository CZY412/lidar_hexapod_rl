"""Headless integration test for direct-HAA-range ``el4090_envelop_3``.

Run from ``legged_gym``:

    python tests/test_envelop_3_env.py --headless

The test deliberately exercises both direct random sampling and a coupled
extreme range.  It is a simulator smoke test, not a policy-quality benchmark.
"""

import isaacgym  # noqa: F401 -- must precede torch-backed environment imports.
from isaacgym import gymapi, gymtorch
import torch

from legged_gym.envs import *  # noqa: F401,F403 -- register tasks.
from legged_gym.utils import get_args, task_registry


TASK_NAME = "el4090_envelop_3"
PAIR_TO_LEG = torch.tensor((0, 1, 2, 0, 1, 2), dtype=torch.long)
PANEL_KEYS = (
    ("front_width", gymapi.KEY_1),
    ("middle_width", gymapi.KEY_2),
    ("back_width", gymapi.KEY_3),
    ("forward_limit", gymapi.KEY_4),
    ("backward_limit", gymapi.KEY_5),
)


def _print_control_panel(env, values: torch.Tensor, active: int) -> None:
    """Print the five-value viewer control panel and the applied V3 posture."""
    names = env.haa_range_state.condition_names
    low, high = env.haa_range_state.condition_low, env.haa_range_state.condition_high
    print("\n=== V3 envelope control panel (focus the Isaac Gym viewer) ===")
    print("1..5 select parameter | Q decrease | E increase | R random valid | 0 midpoint | P print")
    for index, (name, _) in enumerate(PANEL_KEYS):
        marker = ">" if index == active else " "
        print(f"{marker} {index + 1}. {name:15s} = {values[index].item():+.3f} "
              f"[{low[index].item():+.3f}, {high[index].item():+.3f}]")
    target = env._get_condition_target_dof_pos()[0, env.haa_indices]
    default = env.default_dof_pos[0, env.haa_indices]
    bias = env.haa_range_bias[0, env.haa_indices]
    actual = env.dof_pos[0, env.haa_indices]
    print("HAA actual pose in viewer (red marker) = default + bias:")
    for leg, dof_index in enumerate(env.haa_indices.tolist()):
        print(f"  {env.dof_names[dof_index]:6s}: default={default[leg]:+.3f} "
              f"bias={bias[leg]:+.3f} target={target[leg]:+.3f} actual={actual[leg]:+.3f}")


def _apply_panel_values(env, values: torch.Tensor) -> None:
    """Run envelope -> range network and place the simulator at default+bias."""
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    condition = values.to(device=env.device, dtype=torch.float).unsqueeze(0).repeat(env.num_envs, 1)
    env.haa_range_state.set_conditions(condition, env_ids)
    env._refresh_haa_range_bias(env_ids)
    target = env._get_condition_target_dof_pos()
    env.embedded_state_default_dof_pos[:] = target
    env.filtered_embedded_state_default_dof_pos[:] = target
    env.dof_pos[env_ids] = target[env_ids]
    env.dof_vel[env_ids] = 0.0
    env.actions[env_ids] = 0.0
    env.last_actions[env_ids] = 0.0
    env.gym.set_dof_state_tensor_indexed(
        env.sim, gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(env_ids.to(dtype=torch.int32)), len(env_ids),
    )
    env.compute_observations()


def _subscribe_panel_keys(env) -> None:
    for index, (_, key) in enumerate(PANEL_KEYS):
        env.gym.subscribe_viewer_keyboard_event(env.viewer, key, f"panel_select_{index}")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_Q, "panel_decrease")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_E, "panel_increase")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_R, "panel_random")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_0, "panel_midpoint")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_P, "panel_print")


def _assert_valid_ranges(env, ranges: torch.Tensor, label: str) -> None:
    low, high = env.cfg.haa_range.joint_limits
    min_width = env.cfg.haa_range.min_width
    assert ranges.shape == (env.num_envs, 6, 2), f"{label}: unexpected shape {tuple(ranges.shape)}"
    assert bool(torch.all(ranges[..., 0] >= low)), f"{label}: lower endpoint below {low}"
    assert bool(torch.all(ranges[..., 1] <= high)), f"{label}: upper endpoint above {high}"
    assert bool(torch.all(ranges[..., 1] - ranges[..., 0] >= min_width - 1e-6)), (
        f"{label}: interval narrower than {min_width}"
    )
    assert torch.equal(ranges[:, 0], ranges[:, 3]), f"{label}: LB/RB ranges differ"
    assert torch.equal(ranges[:, 1], ranges[:, 4]), f"{label}: LF/RF ranges differ"
    assert torch.equal(ranges[:, 2], ranges[:, 5]), f"{label}: LM/RM ranges differ"


def _safe_ranges(env) -> torch.Tensor:
    """A neutral coupled box known to be suitable for explicit-set testing."""
    pair_ranges = torch.tensor(
        ((-0.15, 0.15), (-0.15, 0.15), (-0.15, 0.15)),
        dtype=torch.float,
        device=env.device,
    )
    return pair_ranges[PAIR_TO_LEG.to(env.device)].unsqueeze(0).repeat(env.num_envs, 1, 1)


def _check_standing_command_sampling(env, rounds: int = 128) -> None:
    """Check optional standing-command sampling without changing HAA ranges."""
    exact_zero_count = 0
    standing_count = 0
    total = 0
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    # v1's standing objective uses the environment threshold directly; v3 no
    # longer carries a separate standing-only reward setting.
    threshold = float(getattr(env.cfg.rewards, "stand_command_threshold", env.speed_min))
    ranges_before = env.haa_swing_ranges.clone()
    for _ in range(rounds):
        env._resample_commands(env_ids)
        commands = env.commands[:, :3]
        exact_zero_count += int(torch.all(commands == 0.0, dim=1).sum().item())
        standing_count += int((torch.linalg.vector_norm(commands, dim=1) < threshold).sum().item())
        total += env.num_envs

    # Expected rates are 2.5% exact-zero and 5% total standing. These broad
    # bounds avoid statistical flakes while detecting removed sampling logic.
    exact_fraction = exact_zero_count / total
    standing_fraction = standing_count / total
    if bool(getattr(env.cfg.commands, "enable_stand_commands", False)):
        assert 0.005 <= exact_fraction <= 0.06, f"exact-zero fraction {exact_fraction:.3f} is unexpected"
        assert 0.02 <= standing_fraction <= 0.10, f"standing fraction {standing_fraction:.3f} is unexpected"
    else:
        assert exact_zero_count == 0, "disabled standing-command sampler produced exact-zero commands"
    assert torch.equal(env.haa_swing_ranges, ranges_before), "command resampling changed an episode-fixed HAA range"


def _standing_diagnostics(env, step: int) -> None:
    """Print static clearance and sagittal-joint deviation for env zero."""
    # On a plane, Isaac Gym stores measured_heights as scalar zero; keep the
    # exact tensor-minus-scalar formulation used by the v3 height rewards.
    height = torch.mean(env.root_states[:, 2].unsqueeze(1) - env.measured_heights, dim=1)
    sagittal_indices = torch.tensor(
        [index for index, name in enumerate(env.dof_names) if name.endswith(("_HFE", "_KFE"))],
        dtype=torch.long,
        device=env.device,
    )
    names = [env.dof_names[index] for index in sagittal_indices.tolist()]
    offset = env.dof_pos[:, sagittal_indices] - env.default_dof_pos[:, sagittal_indices]
    print(
        "  env0 envelope: "
        + ", ".join(f"{name}={value:.3f}" for name, value in zip(env.haa_range_state.condition_names, env.envelope_parameters[0].tolist()))
    )
    print(
        "  env0 HAA centres [rad]: "
        + ", ".join(f"{value:.3f}" for value in (0.5 * (env.haa_swing_ranges[0, :, 0] + env.haa_swing_ranges[0, :, 1])).tolist())
    )
    print(
        f"stand step={step}: height mean/min/max="
        f"{height.mean().item():.3f}/{height.min().item():.3f}/{height.max().item():.3f} m"
    )
    print(
        "  env0 HFE/KFE offsets from default [rad]: "
        + ", ".join(f"{name}={offset[0, i].item():+.3f}" for i, name in enumerate(names))
    )


def test_env(args) -> None:
    """Create v3, validate its direct-range contracts, and run zero-action steps."""
    env_cfg, _ = task_registry.get_cfgs(name=TASK_NAME)
    env_cfg.env.num_envs = int(args.num_envs)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    # In viewer mode draw the five-parameter footprint, HAA range arc, and
    # actual joint marker so default+bias is visible on the simulated robot.
    env_cfg.commands.envelope_debug_viz = not args.headless
    env_cfg.commands.resampling_time = 1e9
    env_cfg.env.episode_length_s = 1e9

    env, _ = task_registry.make_env(name=TASK_NAME, args=args, env_cfg=env_cfg)
    env.reset()
    obs = env.get_observations()
    assert obs.shape == (env.num_envs, 79), f"unexpected observation shape {tuple(obs.shape)}"
    assert bool(torch.isfinite(obs).all()), "initial observation contains NaN or Inf"

    sampled = env.sample_haa_ranges()
    _assert_valid_ranges(env, sampled, "network envelope sample")
    parameters = env.envelope_parameters
    assert parameters.shape == (env.num_envs, 5), "missing five-parameter envelope state"
    assert bool(torch.all(parameters >= env.haa_range_state.condition_low)), "sampled envelope below its configured domain"
    assert bool(torch.all(parameters <= env.haa_range_state.condition_high)), "sampled envelope above its configured domain"
    assert bool(torch.all(env._get_gait_reward_active_mask() == 1.0)), (
        "v1-style rewards must not be masked after an HAA-range sample"
    )
    centers = 0.5 * (sampled[..., 0] + sampled[..., 1])
    target = env._get_condition_target_dof_pos()[:, env.haa_indices]
    assert torch.allclose(target, centers, atol=1e-6), "HAA centre bias does not match sampled interval centres"

    safe_ranges = _safe_ranges(env)
    applied = env.set_haa_ranges(safe_ranges)
    _assert_valid_ranges(env, applied, "neutral explicit sample")
    assert torch.equal(applied, safe_ranges), "set_haa_ranges changed a valid safe range"

    # The actual stand uses the same random envelope/network path as v3
    # training, rather than the neutral interval used only to test the setter.
    standing_ranges = env.sample_haa_ranges()
    _assert_valid_ranges(env, standing_ranges, "policy-free envelope sample")

    _check_standing_command_sampling(env)
    env.commands[:, :3] = 0.0
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    panel_values = env.envelope_parameters[0].detach().clone()
    active_panel_parameter = 0
    panel_step = float(args.panel_step)
    if not args.headless:
        _subscribe_panel_keys(env)
        _apply_panel_values(env, panel_values)
        _print_control_panel(env, panel_values, active_panel_parameter)
    report_every = max(1, int(args.report_every))
    for step in range(int(args.steps)):
        if not args.headless:
            # Poll before env.step(): BaseTask.render() also drains viewer
            # actions, so this must happen before its render call.
            panel_changed = False
            for event in env.gym.query_viewer_action_events(env.viewer):
                if event.value <= 0:
                    continue
                if event.action.startswith("panel_select_"):
                    active_panel_parameter = int(event.action.rsplit("_", 1)[1])
                    _print_control_panel(env, panel_values, active_panel_parameter)
                elif event.action == "panel_decrease":
                    panel_values[active_panel_parameter] -= panel_step
                    panel_changed = True
                elif event.action == "panel_increase":
                    panel_values[active_panel_parameter] += panel_step
                    panel_changed = True
                elif event.action == "panel_random":
                    bank = env.haa_range_state.valid_conditions
                    panel_values = bank[torch.randint(bank.shape[0], (), device=env.device)].detach().clone()
                    panel_changed = True
                elif event.action == "panel_midpoint":
                    panel_values = 0.5 * (
                        env.haa_range_state.condition_low + env.haa_range_state.condition_high
                    )
                    panel_changed = True
                elif event.action == "panel_print":
                    _print_control_panel(env, panel_values, active_panel_parameter)
            if panel_changed:
                panel_values = torch.maximum(
                    torch.minimum(panel_values, env.haa_range_state.condition_high),
                    env.haa_range_state.condition_low,
                )
                _apply_panel_values(env, panel_values)
                _print_control_panel(env, panel_values, active_panel_parameter)
        obs, _, reward, _, _ = env.step(actions)
        assert obs.shape == (env.num_envs, 79), f"unexpected step observation shape {tuple(obs.shape)}"
        assert bool(torch.isfinite(obs).all()), "step observation contains NaN or Inf"
        assert bool(torch.isfinite(reward).all()), "reward contains NaN or Inf"
        if (step + 1) % report_every == 0 or step + 1 == int(args.steps):
            _standing_diagnostics(env, step + 1)

    print(
        f"PASS {TASK_NAME}: envs={env.num_envs}, steps={args.steps}, "
        f"obs={tuple(obs.shape)}, policy-free standing diagnostic completed."
    )


if __name__ == "__main__":
    args = get_args([
        {"name": "--steps", "type": int, "default": 1000000,
         "help": "Zero-action simulator steps after validation; use Ctrl+C or close the viewer to stop."},
        {"name": "--report_every", "type": int, "default": 100, "help": "Print standing diagnostics every N steps."},
        {"name": "--panel_step", "type": float, "default": 0.01,
         "help": "Increment in metres for Q/E envelope-panel adjustments."},
    ])
    args.task = TASK_NAME
    # ``get_args`` already owns --num_envs. Keep this smoke test small unless
    # the caller explicitly supplies its built-in override.
    if args.num_envs is None:
        args.num_envs = 1
    test_env(args)
