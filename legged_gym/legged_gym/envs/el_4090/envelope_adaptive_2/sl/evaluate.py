"""Closed-loop evaluation of the EA2 supervised policy (B3).

Offline R2 is a proxy; what actually matters is how the policy behaves when it
is wired back into the environment.  This module measures that, reporting both
the reward the environment hands out and the physical quantities behind it
(envelope area, hard-collision rate, temporal variation).

Action mapping
--------------
The environment maps raw actions to envelope parameters via
``target = default + a * scale`` with ``scale = (high-low) * k`` and
``k = soft_dof_pos_limit / (2*action_max)`` taken from the *live* config
(:func:`sl.sl_config.env_action_scale`).  Normalising gives
``s = 0.5 + k*a`` for the four "larger is wider" dimensions, but
``backward_limit`` has ``low = -0.9 > high = -0.6``, which flips the sign.
Hence::

    a = ACTION_SIGN * (s - 0.5) / k          # ACTION_SIGN = [1,1,1,1,-1]

Deliberately **not** clipped to ``action_max``: ``s`` in {0, 1} maps slightly
past the soft bound, and letting it through places the envelope on its hard
bound, matching what the oracle produced.  The residual
``envelope_limit_violation`` is ~1e-3 * 0.8, i.e. negligible.  The same fold is
baked into the final actor layer by ``export.export``, which is why
``play_ea2`` needs no conversion while this module (which drives the raw
``EnvelopeNet``) does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .sl_config import ACTION_SIGN, SLConfig, env_action_scale


def load_checkpoint(path: str, device: str = "cpu"):
    """Load ``(net, meta)`` from a checkpoint written by ``train.save_checkpoint``."""
    from .model import EnvelopeNet
    from .sl_config import SLModelConfig

    raw = torch.load(path, map_location="cpu")
    cfg = SLModelConfig()
    if "cfg_model" in raw:
        for k, v in raw["cfg_model"].items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    net = EnvelopeNet(cfg)
    net.load_state_dict(raw["state_dict"])
    net.to(device).eval()
    return net, {k: v for k, v in raw.items() if k != "state_dict"}


def s_to_action(s: torch.Tensor, device=None) -> torch.Tensor:
    """Normalised params -> raw actions (inverse of the env mapping)."""
    sign = torch.tensor(ACTION_SIGN, dtype=s.dtype, device=s.device)
    return sign * (s - 0.5) / env_action_scale()


@dataclass
class RolloutResult:
    step_reward: float
    pred_target_mse: float
    collision_rate: float
    temporal_std: float
    #: Mean envelope area in m^2.  NaN when there is no prediction (zero action).
    area: float
    n_steps: int
    #: Area of the oracle envelope over the same states, for comparison.
    oracle_area: float = float("nan")

    def as_dict(self) -> Dict:
        return {
            "step_reward": self.step_reward,
            "pred_target_mse": self.pred_target_mse,
            "collision_rate": self.collision_rate,
            "temporal_std": self.temporal_std,
            "area": self.area,
            "oracle_area": self.oracle_area,
            "n_steps": self.n_steps,
        }


def _make_predictor(net, mode: str, seq_len: int, decimation: int, device: str) -> Callable:
    """Build a per-step action-producing closure.

    ``mode="stateful"``  **recommended**.  Feeds one frame at a time and carries
                         the GRU hidden state forward across the whole episode.
                         Although the net was only ever unrolled for
                         ``seq_len`` steps during training, the learned dynamics
                         remain stable over far longer horizons -- and the extra
                         memory pays off: measured over four maps, stateful
                         reaches MSE ~0.013 versus ~0.049 for the window mode
                         (about 4x better) with a slightly lower collision rate.

    ``mode="window"``    re-runs the last ``seq_len`` frames through the GRU
                         every ``decimation`` steps, exactly reproducing the
                         training condition.  It works, but discards everything
                         older than the window, which measurably costs accuracy.
                         Useful as a conservative fallback and for comparison.

    History note
    ------------
    An earlier version of the stateful branch returned ``pred[0]`` instead of
    ``pred``, collapsing the ``(batch, 5)`` output to a single environment's
    parameters that every environment then shared via broadcasting.  Its poor
    numbers were reported as "long-horizon drift" -- that conclusion was an
    artefact of the indexing bug, not a property of recurrent deployment.
    """
    if mode == "stateful":
        state = {"hidden": None}

        def fn(obs_frame: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                # net.step already returns (batch, 5) -- indexing with [0] here
                # would collapse it to one env's parameters, which every env
                # would then share through broadcasting.
                pred, state["hidden"] = net.step(obs_frame.unsqueeze(0).to(device), state["hidden"])
                return pred.clamp(0.0, 1.0)

        return fn

    if mode == "window":
        history: List[torch.Tensor] = []

        def fn(obs_frame: torch.Tensor) -> torch.Tensor:
            history.append(obs_frame)
            if len(history) < seq_len:
                return torch.full((obs_frame.shape[0], 5), 0.5, device=device)
            window = torch.stack(history[-seq_len:]).to(device)
            with torch.no_grad():
                pred = net(window)[-1]
            return pred.clamp(0.0, 1.0)

        return fn

    raise ValueError(f"unknown mode {mode!r}")


def closed_loop_rollout(
    env,
    net,
    cfg: SLConfig,
    mode: str = "stateful",  # recommended; see _make_predictor
    num_steps: int = 700,
    warmup_steps: int = 30,
    device: str = "cuda",
    zero_action: bool = False,
    constant_s: Optional[torch.Tensor] = None,
) -> RolloutResult:
    """Deploy the policy inside ``env`` and report environment-level metrics.

    Args:
        env: an ``EL_4090_EA2`` instance (already constructed).
        net: trained ``EnvelopeNet``.
        mode: ``"stateful"`` or ``"window"``.
        zero_action: run the all-zeros action instead of the net (baseline).
        constant_s: run a constant normalised envelope instead (baseline).
    """
    dev = env.device
    n_envs = env.num_envs
    decimation = cfg.data.lidar_decimation

    predict = None if (zero_action or constant_s is not None) else _make_predictor(
        net, mode, cfg.train.seq_len, decimation, device
    )

    last_s = None
    last_a = torch.zeros(n_envs, 5, device=dev)
    rewards: List[float] = []
    mses: List[float] = []
    collisions: List[float] = []
    s_hist: List[torch.Tensor] = []

    # The first observation must come from reset(); afterwards we consume the
    # observation returned by step(), which already corresponds to the state
    # produced by the action we just applied.
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    for t in range(num_steps):
        if zero_action:
            last_a = torch.zeros(n_envs, 5, device=dev)
            last_s = None
        elif constant_s is not None:
            last_s = constant_s
            last_a = s_to_action(constant_s).to(dev)
        else:
            # obs is the observation for the *current* state -> act on it.
            if t % decimation == 0:
                last_s = predict(obs).to(dev)
                last_a = s_to_action(last_s).to(dev)

        out = env.step(last_a)
        obs, _, reward = out[0], out[1], out[2]

        if t < warmup_steps:
            continue
        # reward / collision are always meaningful, including for the zero-action
        # baseline which has no prediction of its own
        rewards.append(float(reward.mean()))
        if hasattr(env, "_collision_hard"):
            collisions.append(float((env._collision_hard > 0).float().mean()))
        if last_s is not None:
            mses.append(float(((last_s - env._oracle_smoother.prev_s) ** 2).mean()))
            s_hist.append(last_s.detach().cpu())

    if not rewards:
        raise RuntimeError("no steps recorded -- increase num_steps beyond warmup")

    preds = torch.stack(s_hist) if s_hist else None  # (T, N, 5)

    low = torch.tensor([0.3, 0.3, 0.3, 0.6, -0.9], device=dev)
    high = torch.tensor([0.6, 0.7, 0.6, 0.9, -0.6], device=dev)
    min_v = torch.stack([low[0], low[1], low[2], low[3], high[4]]).view(1, 5)
    max_v = torch.stack([high[0], high[1], high[2], high[3], low[4]]).view(1, 5)

    def _area_of(s: torch.Tensor) -> float:
        """Mean hexagon area in m^2 for normalised params ``s``.

        The hexagon is two triangles sharing the lateral axis: the front one
        spans ``front_width * forward_limit``, the rear one
        ``back_width * |backward_limit|`` (``backward_limit`` is negative).
        """
        p = min_v + s.clamp(0, 1) * (max_v - min_v)
        return float((p[..., 0] * p[..., 3] - p[..., 4] * p[..., 2]).mean())

    result = RolloutResult(
        step_reward=float(np.mean(rewards)),
        # NaN for the zero-action baseline, which produces no prediction
        pred_target_mse=float(np.mean(mses)) if mses else float("nan"),
        collision_rate=float(np.mean(collisions)) if collisions else float("nan"),
        temporal_std=float(preds.float().std(dim=0).mean()) if preds is not None else float("nan"),
        area=_area_of(preds.to(dev)) if preds is not None else float("nan"),
        n_steps=len(rewards),
    )
    result.oracle_area = _area_of(env._oracle_smoother.prev_s) if preds is not None else float("nan")
    return result


def evaluate_closed_loop(
    cfg: SLConfig,
    net,
    seed: int = 1,
    num_envs: int = 64,
    num_steps: int = 700,
    device: str = "cuda",
    env=None,
    modes=("stateful", "window"),
    oracle_mean: Optional[torch.Tensor] = None,
) -> Dict[str, Dict]:
    """Run the baselines plus the policy and return a comparison dictionary.

    Args:
        modes: deployment modes to evaluate.  Defaults to both, stateful first
            because it is the recommended one (see ``_make_predictor``).
        oracle_mean: constant envelope used for the "best constant" baseline.
            When ``None`` it is estimated live by running a zero-action rollout
            and averaging the oracle targets, which is the *strongest* possible
            constant policy.  Comparing against 0.5 instead would understate the
            baseline by roughly 2x on measured data (MSE 0.167 vs 0.083).
    """
    from .dataset import build_env

    owned = env is None
    if owned:
        env = build_env(seed, num_envs)

    try:
        results: Dict[str, Dict] = {}

        # --- baselines -----------------------------------------------------
        zero = closed_loop_rollout(
            env, net, cfg, num_steps=num_steps, device=device, zero_action=True
        )
        results["zero_action"] = zero.as_dict()

        if oracle_mean is None:
            oracle_mean = estimate_oracle_mean(env, cfg, device=device)
        mean_s = oracle_mean.to(env.device).view(1, 5).expand(env.num_envs, 5)
        const = closed_loop_rollout(
            env, net, cfg, num_steps=num_steps, device=device, constant_s=mean_s
        )
        results["constant_oracle_mean"] = const.as_dict()

        # --- policy --------------------------------------------------------
        for mode in modes:
            res = closed_loop_rollout(env, net, cfg, mode=mode, num_steps=num_steps, device=device)
            results[f"sl_{mode}"] = res.as_dict()
        return results
    finally:
        if owned:
            del env
            torch.cuda.empty_cache()


def estimate_oracle_mean(env, cfg: SLConfig, num_steps: int = 300, device: str = "cuda") -> torch.Tensor:
    """Estimate the oracle target mean with a short zero-action rollout.

    This is the best constant policy achievable: the envelope that minimises
    expected MSE against the oracle.  It is a much stronger baseline than the
    midpoint 0.5 of the normalised range.
    """
    dev = env.device
    zero = torch.zeros(env.num_envs, 5, device=dev)
    acc = []
    env.reset()
    for t in range(num_steps):
        env.step(zero)
        if t >= cfg.data.warmup_steps:
            acc.append(env._oracle_smoother.prev_s.detach().mean(dim=0))
    if not acc:
        return torch.full((5,), 0.5, device=device)
    return torch.stack(acc).mean(dim=0).to(device)
