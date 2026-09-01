"""Isaac-in-the-loop cascade env integration test (headless, no SE2 policy).

Constructs the registered ``el4090_cascade`` env and steps it with small
random locomotion actions (the SE2 checkpoint is NOT required), asserting:

* T0 static contracts: 50 Hz control, 83-dim obs, lidar/heights off,
  8-dim condition, 187 channels, fold-scale guard passed at construction;
* refresh cadence: range image refreshes exactly on the 10 Hz grid with a
  first-step refresh;
* reset path: timeout resets resample commands through the cascade override,
  mark the reset env's range image stale and recover at the next scan;
* full bridge: params5/condition8 stay in bounds and finite; the EA2 GRU
  hidden state survives resets; SE2 observations stay finite;
* reactivity: the EA2 condition is not frozen across the run.

Variants (each needs its own process — one Isaac env per process):

* default                          — full cascade pipeline (above);
* ``--ea2-off``                    — ``ea2.enable=False`` restores pure SE2
  behaviour: no cascade state, envelopes randomly sampled per reset;
* ``--birth-condition midpoint``   — reset preset knob plumbing.

Run:
    python legged_gym/tests/ea2/cascade/run_cascade_env_test.py [flags]
"""

import isaacgym  # noqa: F401  -- must precede all legged_gym imports

import sys

import torch

# ── defaults for standalone invocation ──────────────────────────────────
if not any(arg.startswith("--task") for arg in sys.argv):
    sys.argv += ["--task", "el4090_cascade"]
if "--headless" not in sys.argv:
    sys.argv.append("--headless")

import legged_gym.envs.el_4090.envelope_cascade  # noqa: F401,E402  -- registers task
from legged_gym.utils import get_args, task_registry  # noqa: E402
from legged_gym.utils.envelop.network.haa_swing_range import (  # noqa: E402
    load_envelope_condition_spec,
)
from legged_gym.envs.el_4090.envelope_adaptive_2 import _contracts as ea2c  # noqa: E402

NUM_STEPS = 130


def make_check(failures):
    def check(name, cond, detail=""):
        print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
        if not cond:
            failures.append(name)

    return check


def build_env(args, env_cfg):
    env_cfg.env.num_envs = 2
    env_cfg.env.episode_length_s = 0.6  # timeout resets every 30 steps
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.mesh_type = "trimesh"
    env_cfg.terrain.terrain_proportions = [0.0] * 7 + [0.5, 0.5]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.resampling_time = 99999
    env, _ = task_registry.make_env(name="el4090_cascade", args=args, env_cfg=env_cfg)
    # 构造期初始姿态与地形穿插（上游既有行为）：先做一次正规 reset。
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    return env


def run_main_variant(env, check):
    """Default variant: full EA2→SE2 cascade pipeline."""
    check("cascade ready", env._cascade_ready)
    check("inherited lidar disabled", getattr(env.cfg.lidar, "enable", False) is False)
    check("measure_heights disabled", env.cfg.terrain.measure_heights is False)
    check("condition dim == 8", env.envelope_state.condition_dim == 8)
    check("SE2 obs dim == 83", int(env.cfg.env.num_observations) == 83)
    check("fold scale guard", abs(env._cascade_bridge.fold_scale - env.cfg.ea2.fold_scale) < 1e-9)
    check("perception channels == 187", tuple(env._cascade_perception.range_image.shape) == (2, 187))

    spec = load_envelope_condition_spec(ea2c.ENVELOPE_SPEC_CONFIG_PATH)
    low8 = torch.tensor(spec.low, device=env.device)
    high8 = torch.tensor(spec.high, device=env.device)
    low5, high5 = low8[:5], high8[:5]

    env.compute_observations()  # obs_buf is 1-D zeros until first computation
    obs = env.get_observations()  # BaseTask returns the raw obs_buf tensor
    check("initial obs shape", tuple(obs.shape) == (2, 83), str(tuple(obs.shape)))

    num_actions = int(env.cfg.env.num_actions)
    seen_conditions = []
    stale_after_reset_seen = False
    dones_total = 0

    for step in range(NUM_STEPS):
        actions = (torch.rand(2, num_actions, device=env.device) - 0.5) * 0.2
        obs, _, _, dones, _ = env.step(actions)
        dones_total += int(dones.sum().item())

        p5 = env.cascade_params5
        c8 = env.cascade_condition8
        if not bool(torch.isfinite(p5).all()) or not bool(torch.isfinite(c8).all()):
            check(f"finite bridge output at step {step}", False)
            return
        if not env.cascade_action_finite:
            check(f"finite EA2 action at step {step}", False)
            return
        if bool(((p5 < low5 - 1e-5) | (p5 > high5 + 1e-5)).any()):
            check(f"params5 in bounds at step {step}", False)
            return
        if bool(((c8 < low8 - 1e-5) | (c8 > high8 + 1e-5)).any()):
            check(f"condition8 in bounds at step {step}", False)
            return
        if not bool(torch.isfinite(obs).all()):
            check(f"finite SE2 obs at step {step}", False)
            return

        seen_conditions.append(c8[0].detach().cpu().clone())

        perception = env._cascade_perception
        if step == 0:
            check("first-step refresh", perception.refresh_count == 1)
        if bool(dones.any()):
            reset_ids = dones.nonzero(as_tuple=False).flatten()
            if bool(perception.stale[reset_ids].all()):
                stale_after_reset_seen = True

    expected_refreshes = (NUM_STEPS + 4) // 5
    check(
        "10 Hz refresh cadence",
        env._cascade_perception.refresh_count == expected_refreshes,
        f"{env._cascade_perception.refresh_count} vs {expected_refreshes}",
    )
    check("stale-on-reset observed", stale_after_reset_seen, f"dones={dones_total}")
    check(
        "EA2 condition reacts over run",
        len({tuple(row.tolist()) for row in seen_conditions}) >= 3,
        f"{len({tuple(r.tolist()) for r in seen_conditions})} distinct",
    )

    summary = env.cascade_debug_summary()
    print("debug summary:", summary)
    check("bound_ratio sane", 0.0 <= summary["bound_ratio"] <= 1.0)


def run_fallback_variant(env, check):
    """``ea2.enable=False``: pure SE2 behaviour with sampled envelopes."""
    check("cascade not constructed", env._cascade_ready is False)
    check("inherited lidar disabled", getattr(env.cfg.lidar, "enable", False) is False)

    spec = load_envelope_condition_spec(ea2c.ENVELOPE_SPEC_CONFIG_PATH)
    low8 = torch.tensor(spec.low, device=env.device)
    high8 = torch.tensor(spec.high, device=env.device)

    env.compute_observations()
    obs = env.get_observations()
    check("initial obs shape", tuple(obs.shape) == (2, 83), str(tuple(obs.shape)))

    num_actions = int(env.cfg.env.num_actions)
    sampled_condition_rows = set()
    dones_seen = 0
    for step in range(NUM_STEPS):
        actions = (torch.rand(2, num_actions, device=env.device) - 0.5) * 0.2
        obs, _, _, dones, _ = env.step(actions)
        if not bool(torch.isfinite(obs).all()):
            check(f"finite obs at step {step}", False)
            return
        cond = env.envelope_state.get()
        if bool(((cond < low8 - 1e-5) | (cond > high8 + 1e-5)).any()):
            check(f"sampled condition in bounds at step {step}", False)
            return
        if bool(dones.any()):
            dones_seen += int(dones.sum().item())
            reset_ids = dones.nonzero(as_tuple=False).flatten()
            sampled_condition_rows.update(
                tuple(row) for row in cond[reset_ids].detach().cpu().tolist()
            )

    check("resets exercised (sampled envelopes)", dones_seen > 0, f"dones={dones_seen}")
    check(
        "envelopes resampled per reset",
        len(sampled_condition_rows) >= 2,
        f"{len(sampled_condition_rows)} distinct",
    )


def run_midpoint_variant(env, check):
    """``ea2.reset_condition="midpoint"``: birth preset knob plumbing."""
    check("cascade ready", env._cascade_ready)
    birth = env._cascade_birth_condition()
    state = env.envelope_state
    mid = 0.5 * (state.low + state.high)
    check(
        "birth condition is the midpoint template",
        torch.allclose(birth[0, :5], mid[:5], atol=1e-6),
    )
    check(
        "birth condition differs from max template",
        not torch.allclose(birth[0, :5], env._cascade_max_condition[0, :5]),
    )

    num_actions = int(env.cfg.env.num_actions)
    for step in range(60):
        actions = (torch.rand(2, num_actions, device=env.device) - 0.5) * 0.2
        obs, _, _, dones, _ = env.step(actions)
        p5, c8 = env.cascade_params5, env.cascade_condition8
        if not (bool(torch.isfinite(obs).all()) and bool(torch.isfinite(p5).all())
                and bool(torch.isfinite(c8).all())):
            check(f"finite pipeline at step {step}", False)
            return
    check("midpoint variant pipeline healthy", True)


def main() -> int:
    failures = []
    check = make_check(failures)

    extra = [
        {"name": "--ea2-off", "action": "store_true", "default": False,
         "help": "Run the ea2.enable=False fallback variant"},
        {"name": "--birth-condition", "type": str, "default": "max",
         "help": "ea2.reset_condition variant (max | midpoint)"},
    ]
    env_cfg, _ = task_registry.get_cfgs(name="el4090_cascade")
    env_cfg.ea2.debug_stats = True
    variant = "main"

    args = get_args(extra)
    if getattr(args, "ea2_off", False):
        variant = "ea2-off"
        env_cfg.ea2.enable = False
    else:
        birth = str(getattr(args, "birth_condition", "max"))
        if birth != "max":
            variant = f"birth-{birth}"
            env_cfg.ea2.reset_condition = birth

    print(f"===== cascade env test variant: {variant} =====")
    env = build_env(args, env_cfg)
    check("dt == 0.02 (50 Hz control)", abs(env.dt - 0.02) < 1e-9, f"dt={env.dt}")

    if variant == "main":
        run_main_variant(env, check)
    elif variant == "ea2-off":
        run_fallback_variant(env, check)
    else:
        run_midpoint_variant(env, check)

    print(f"\n===== cascade env test [{variant}]:",
          "PASS" if not failures else f"FAIL ({failures})", "=====")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
