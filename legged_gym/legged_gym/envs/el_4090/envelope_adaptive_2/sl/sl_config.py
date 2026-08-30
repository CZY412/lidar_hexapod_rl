"""Configuration for the offline supervised-learning pipeline.

Deliberately *separate* from :class:`El4090EA2Cfg` / :class:`El4090EA2CfgPPO`.

The most important consequence of this separation: the SL sequence length is
NOT bounded by ``runner.num_steps_per_env``.  PPO's BPTT horizon equals its
rollout length (100 steps == 2 s), whereas SL can train on any window it likes
(40 LiDAR frames == 4 s), which is where a large part of its advantage comes
from.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class SLDataConfig:
    """Data-collection options."""

    #: Map seeds to collect, each paired with an obstacle-count override in
    #: ``pillar_counts``.  ``env.py`` reads ``cfg.seed`` as ``map_seed``, so each
    #: seed yields a different obstacle layout.  ``1`` is the production value
    #: (``task_registry`` copies ``train_cfg.seed`` into ``env_cfg.seed``).
    #:
    #: Include at least one dense map -- dense scenes are the measured weak spot
    #: (R2 0.72 vs 0.83, collision 25% vs 18%).
    seeds: List[int] = field(default_factory=lambda: [1, 7, 13, 21])

    #: Obstacle-count overrides, positionally paired with ``seeds``.  ``None``
    #: keeps the config default (18).  Use :meth:`seed_specs` to iterate the two
    #: lists together safely.
    pillar_counts: List[Optional[int]] = field(
        default_factory=lambda: [None, None, None, 28]
    )

    def seed_specs(self) -> List["tuple[int, Optional[int]]"]:
        """Zip ``seeds`` with ``pillar_counts``, padding with ``None``."""
        counts = list(self.pillar_counts) + [None] * (
            len(self.seeds) - len(self.pillar_counts)
        )
        return list(zip(self.seeds, counts[: len(self.seeds)]))

    #: Parallel environments used during collection.
    num_envs: int = 96

    #: Control steps per collection episode.
    num_steps: int = 1400

    #: LiDAR refresh decimation.  The sensor runs at 10 Hz while control runs at
    #: 50 Hz, so with ``1`` every control step is stored and the SL cadence
    #: matches PPO/play exactly (both act on every 50 Hz observation).  The
    #: historical ``5`` (10 Hz frames) trained the GRU on a 5x sparser cadence
    #: than deployment and cost ~6x oracle-MSE when the same weights acted at
    #: 50 Hz (measured: 0.011 -> 0.067 on the baseline weights).
    lidar_decimation: int = 1

    #: Steps discarded at the start of every episode.  On reset the rate limiter
    #: seeds ``prev_s = 1`` (fully open) and shrinks at ``shrink_rate = 2.0 m/s``;
    #: with a max span of 0.4 m that settles in ~10 steps.  30 keeps a 3x margin.
    warmup_steps: int = 30


@dataclass
class SLModelConfig:
    """Model architecture.  Must stay isomorphic to rsl_rl's actor."""

    obs_dim: int = 190  # 187 range channels + 3 ego motion
    action_dim: int = 5  # normalised envelope params [0, 1]
    rnn_hidden_dim: int = 187
    rnn_num_layers: int = 1
    rnn_type: str = "gru"
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    activation: str = "elu"

    #: Training-only auxiliary memory heads (a single Linear(hidden, action)
    #: probe per k, attached to the GRU output).  Empty list = disabled; the
    #: heads exist only during SL training and are never exported.
    #: ``aux_mode="recall"`` asks h_t to reconstruct s_{t-k} (pure memory
    #: demand, cannot be satisfied by spatial extrapolation of the current
    #: view); ``"forward"`` asks h_t to predict s_{t+k} (mixed extrapolation
    #: + retention, gradient lands at the write moment).  k is in frames at
    #: the control rate (75 = 1.5 s at 50 Hz ~= the pass-by invisible tail).
    aux_ks: List[int] = field(default_factory=list)
    aux_mode: str = "recall"


@dataclass
class SLTrainConfig:
    """Optimisation options."""

    #: Frames per training sequence.  200 frames @ 50 Hz == 4 s of memory,
    #: the horizon where the offline observability scan saturated (R2 = 1.0)
    #: when it was expressed as 40 frames @ 10 Hz.
    seq_len: int = 200

    #: Stride (in frames) between consecutive window start positions.
    #: 10 frames @ 50 Hz == 0.2 s, the overlap semantics of the historical
    #: stride 2 @ 10 Hz.
    window_stride: int = 10

    #: Store windows as ``uint8``-quantised observations to cut memory ~4x.
    #:
    #: Windowing expands the source data by roughly ``seq_len / stride`` (20x
    #: at seq_len=200, stride=10): four maps worth ~0.4 GiB become ~8 GiB of
    #: float32 tensors.  Quantising the range image to 8 bits trades a
    #: quantisation error of ~3.2/255 m for a 4x memory reduction, which keeps
    #: larger corpora feasible.
    quantise_obs: bool = False

    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    grad_clip: float = 1.0

    #: Fraction of environments held out for validation.  Splitting is done by
    #: environment (never by timestep) so no trajectory straddles the split.
    val_fraction: float = 0.2

    split_seed: int = 1

    #: ``std`` written into the exported ActorCriticRecurrent.
    export_std: float = 0.5

    #: Weight of the auxiliary memory loss (see ``SLModelConfig.aux_ks``).
    #: 0 disables the loss even when heads exist.  Monitor the main val MSE:
    #: if it degrades, lower this.
    aux_beta: float = 0.5

    #: Weight of the differentiable safety (collision) loss on the policy's own
    #: prediction: mean bilinear-sampled violation over the 24 boundary hex
    #: samples, with floor-pinned frames masked out.  0 disables it.  The loss
    #: is zero-gradient on safe frames, so it only acts where the prediction
    #: actually violates geometry.  Requires >=1 for effect; watch the main val
    #: MSE and the policy clearance (collapse guard).
    safe_lambda: float = 0.0


@dataclass
class SLConfig:
    data: SLDataConfig = field(default_factory=SLDataConfig)
    model: SLModelConfig = field(default_factory=SLModelConfig)
    train: SLTrainConfig = field(default_factory=SLTrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)


#: Per-dimension sign of ``s -> a``.  Index 4 is ``backward_limit``.
#: (The scalar scale of the mapping is derived from the live env config by
#: :func:`env_action_scale`; the historical constant 0.1125 for
#: ``soft_dof_pos_limit=0.9`` was removed as dead code once every consumer
#: switched to the derived value.)
ACTION_SIGN: List[float] = [1.0, 1.0, 1.0, 1.0, -1.0]


def env_action_scale() -> float:
    """Span-normalised raw-action scale of the *live* env mapping.

    The environment realises ``s = 0.5 + env_action_scale() * ACTION_SIGN * a``
    through ``target = default + a * scale`` with
    ``scale = span * soft_dof_pos_limit / (2 * action_max)``.  It is derived
    lazily from :class:`El4090EA2Cfg` so the export-time action fold and
    ``evaluate.s_to_action`` can never drift from the mapping the running env
    actually applies (``soft_dof_pos_limit`` 0.9 -> 0.95 moved the scale from
    0.1125 to 0.11875; a stale fold silently pins the deployed envelope near
    its midpoint).
    """
    from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_config import (
        El4090EA2Cfg,
    )

    env_cls = El4090EA2Cfg.envelope
    return float(env_cls.soft_dof_pos_limit) / (2.0 * float(env_cls.action_max))

#: Envelope parameter names, ordered as the environment packs them.
PARAM_NAMES: List[str] = [
    "front_width",
    "middle_width",
    "back_width",
    "forward_limit",
    "backward_limit",
]


# ---------------------------------------------------------------------------
# Default artifact locations
#
# Everything lives under ``sl/logs/`` so that experiments stay with the project
# instead of in /tmp (which, while not auto-cleaned on this machine, is
# semantically disposable and lives on the root partition).
# ---------------------------------------------------------------------------

import os as _os

#: ``.../envelope_adaptive_2/sl``
SL_ROOT: str = _os.path.dirname(_os.path.abspath(__file__))

#: ``.../envelope_adaptive_2/sl/logs``
SL_LOGS: str = _os.path.join(SL_ROOT, "logs")

#: Collected rollouts: ``logs/data/map_seed<N>.pt``
SL_DATA_DIR: str = _os.path.join(SL_LOGS, "data")

#: Per-experiment outputs: ``logs/runs/<run_name>/``
SL_RUNS_DIR: str = _os.path.join(SL_LOGS, "runs")


def data_path(seed: int) -> str:
    return _os.path.join(SL_DATA_DIR, f"map_seed{int(seed)}.pt")


def run_dir(run_name: str) -> str:
    return _os.path.join(SL_RUNS_DIR, run_name)


def available_seeds(data_dir: str = None) -> List[int]:
    """Map seeds that have already been collected, sorted ascending."""
    directory = data_dir or SL_DATA_DIR
    if not _os.path.isdir(directory):
        return []
    out = []
    for name in _os.listdir(directory):
        if name.startswith("map_seed") and name.endswith(".pt"):
            try:
                out.append(int(name[len("map_seed") : -len(".pt")]))
            except ValueError:
                continue
    return sorted(out)
