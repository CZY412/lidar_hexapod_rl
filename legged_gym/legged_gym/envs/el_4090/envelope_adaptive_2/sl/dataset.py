"""Dataset construction for the EA2 supervised-learning pipeline.

Collection
----------
Data is gathered with a *zero-action* rollout.  This is legitimate because the
EA2 state transition does not depend on the action: ``_step_kinematics`` and
``_update_lidar`` read only ``heading`` / ``base_pos`` / the static mesh, and
every consumer of ``self.actions`` lives outside those two functions.  The
consequence is that the collected data has **no distribution shift** relative
to any policy we might later deploy.

Stored per frame (after LiDAR decimation)
-----------------------------------------
``obs``    (T, N, 190)  [0..187) range image / 3.2  ->  [0, 1]
                       [187..190) ego / (1.5, 1.0, 1.5)  ->  [-1, 1]
                       The ego channels are signed: longitudinal speed is
                       non-negative, lateral speed is near zero, and the yaw
                       rate saturates the normaliser at +/-1.  The overall
                       observation range is therefore [-1, 1], not [0, 1].
``target`` (T, N, 5)    ``_oracle_smoother.prev_s``, the rate-limited oracle
``done``   (T, N)       episode boundary, used to avoid straddling windows
``heading`` (T, N)      body heading, needed for physics evaluation
``pos``    (T, N, 2)    body position, needed for physics evaluation

The distance field is stored once per map so that evaluation can recompute
collision/area metrics without rebuilding the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .sl_config import SLConfig


@dataclass
class MapData:
    """One collected map: per-frame tensors plus the static distance field."""

    seed: int
    obs: torch.Tensor  # (T, N, 190) float32
    target: torch.Tensor  # (T, N, 5)   float32
    done: torch.Tensor  # (T, N)      bool
    heading: torch.Tensor  # (T, N)      float32
    pos: torch.Tensor  # (T, N, 2)   float32
    distance_field: torch.Tensor  # (H, W)      float32
    meta: Dict

    def __post_init__(self) -> None:
        assert self.obs.ndim == 3 and self.obs.shape[-1] == 190
        assert self.target.ndim == 3 and self.target.shape[-1] == 5
        assert self.obs.shape[:2] == self.target.shape[:2] == self.done.shape
        assert self.target.dtype == torch.float32


def build_env(
    cfg_seed: int,
    num_envs: int,
    pillar_count: Optional[int] = None,
    sim_device: str = "cuda:0",
    headless: bool = True,
):
    """Construct an ``EL_4090_EA2`` without going through ``task_registry``.

    ``task_registry.get_cfgs`` overwrites ``env_cfg.seed`` with
    ``train_cfg.seed``; constructing directly keeps the seed under our control,
    which is exactly what multi-map collection needs.

    Only one environment may exist per process -- constructing a second one
    segfaults inside Isaac Gym.
    """
    import isaacgym  # noqa: F401  (must be imported before torch-dependent modules)
    from isaacgym import gymapi
    from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_config import El4090EA2Cfg
    from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_env import EL_4090_EA2

    cfg = El4090EA2Cfg()
    cfg.env.num_envs = int(num_envs)
    cfg.lidar.debug_env_ids = []
    cfg.seed = int(cfg_seed)  # env.py reads this as map_seed
    if pillar_count is not None:
        cfg.obstacles.pillar_count_min = int(pillar_count)
        cfg.obstacles.pillar_count_max = int(pillar_count)

    sim_params = gymapi.SimParams()
    sim_params.dt = cfg.sim.dt
    sim_params.substeps = cfg.sim.substeps
    sim_params.gravity = gymapi.Vec3(*cfg.sim.gravity)
    sim_params.up_axis = gymapi.UP_AXIS_Y
    px, cpx = sim_params.physx, cfg.sim.physx
    px.num_threads = cpx.num_threads
    px.solver_type = cpx.solver_type
    px.num_position_iterations = cpx.num_position_iterations
    px.num_velocity_iterations = cpx.num_velocity_iterations
    px.contact_offset = cpx.contact_offset
    px.rest_offset = cpx.rest_offset
    px.bounce_threshold_velocity = cpx.bounce_threshold_velocity
    px.max_depenetration_velocity = cpx.max_depenetration_velocity
    px.max_gpu_contact_pairs = cpx.max_gpu_contact_pairs
    px.default_buffer_size_multiplier = cpx.default_buffer_size_multiplier
    # cfg.sim.physx.contact_collection is stored as an int (2); gymapi expects
    # the ContactCollection enum, which int() converts correctly.
    px.contact_collection = gymapi.ContactCollection(int(cpx.contact_collection))
    sim_params.use_gpu_pipeline = True

    return EL_4090_EA2(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=gymapi.SIM_PHYSX,
        sim_device=sim_device,
        headless=headless,
    )


def collect_map(
    seed: int,
    num_envs: int = 96,
    num_steps: int = 1400,
    lidar_decimation: int = 5,
    pillar_count: Optional[int] = None,
    device: str = "cuda:0",
) -> MapData:
    """Run one zero-action rollout and return the collected tensors."""
    import torch as _torch

    env = build_env(seed, num_envs, pillar_count)
    try:
        env.reset()
        dev = env.device
        zero = _torch.zeros(int(num_envs), 5, device=dev)

        n_frames = int(num_steps) // int(lidar_decimation)
        obs = _torch.zeros(n_frames, num_envs, 190)
        target = _torch.zeros(n_frames, num_envs, 5)
        done = _torch.zeros(n_frames, num_envs, dtype=_torch.bool)
        heading = _torch.zeros(n_frames, num_envs)
        pos = _torch.zeros(n_frames, num_envs, 2)

        k = 0
        for t in range(int(num_steps)):
            out = env.step(zero)
            obs_t = out[0]
            done_t = out[3]
            if t % lidar_decimation == 0 and k < n_frames:
                obs[k] = obs_t.cpu()
                # prev_s is updated inside _compute_rewards, i.e. it already
                # reflects the state that produced obs_t -> the two are in sync.
                target[k] = env._oracle_smoother.prev_s.cpu()
                done[k] = done_t.bool().cpu()
                heading[k] = env.heading.cpu()
                pos[k] = env.base_pos[:, :2].cpu()
                k += 1

        meta = {
            "seed": int(seed),
            "num_envs": int(num_envs),
            "num_steps": int(num_steps),
            "lidar_decimation": int(lidar_decimation),
            "pillar_count": pillar_count,
            "oracle_margin": float(env.cfg.envelope.oracle_margin),
            "oracle_group_mode": str(env.cfg.envelope.oracle_group_mode),
            "reward_scales": dict(env.reward_scales),
            "n_frames": int(k),
        }
        return MapData(
            seed=int(seed),
            obs=obs[:k],
            target=target[:k],
            done=done[:k],
            heading=heading[:k],
            pos=pos[:k],
            distance_field=env.distance_field.detach().cpu().clone(),
            meta=meta,
        )
    finally:
        del env
        _torch.cuda.empty_cache()


def save_map(data: MapData, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(
        {
            "seed": data.seed,
            "obs": data.obs,
            "target": data.target,
            "done": data.done,
            "heading": data.heading,
            "pos": data.pos,
            "distance_field": data.distance_field,
            "meta": data.meta,
        },
        path,
    )


def load_map(path: str) -> MapData:
    raw = torch.load(path, map_location="cpu")
    return MapData(
        seed=int(raw["seed"]),
        obs=raw["obs"].float(),
        target=raw["target"].float(),
        done=raw["done"].bool(),
        heading=raw["heading"].float(),
        pos=raw["pos"].float(),
        distance_field=raw["distance_field"].float(),
        meta=raw.get("meta", {}),
    )


def build_windows(
    obs: torch.Tensor,
    target: torch.Tensor,
    done: torch.Tensor,
    heading: torch.Tensor,
    pos: torch.Tensor,
    seq_len: int,
    warmup_frames: int,
    stride: int = 2,
) -> Dict[str, torch.Tensor]:
    """Slice per-env time series into fixed-length windows.

    Windows that contain a ``done`` are rejected so that no sequence spans an
    episode boundary.  Returns CPU tensors shaped ``(S, seq_len, ...)``.
    """
    n_frames, n_envs = obs.shape[0], obs.shape[1]
    starts: List[Tuple[int, int]] = []
    for n in range(n_envs):
        for st in range(warmup_frames, n_frames - seq_len + 1, stride):
            sl = slice(st, st + seq_len)
            if bool(done[sl, n].any()):
                continue
            starts.append((st, n))

    S = len(starts)
    out = {
        "obs": torch.zeros(S, seq_len, obs.shape[-1]),
        "target": torch.zeros(S, seq_len, target.shape[-1]),
        "heading": torch.zeros(S, seq_len),
        "pos": torch.zeros(S, seq_len, 2),
        "env_id": torch.zeros(S, dtype=torch.long),
    }
    for i, (st, n) in enumerate(starts):
        sl = slice(st, st + seq_len)
        out["obs"][i] = obs[sl, n]
        out["target"][i] = target[sl, n]
        out["heading"][i] = heading[sl, n]
        out["pos"][i] = pos[sl, n]
        out["env_id"][i] = n
    return out


def env_split(
    env_ids: torch.Tensor, val_fraction: float = 0.2, seed: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split window indices into train/val **by environment**.

    Splitting by timestep would leak: consecutive windows from the same env
    overlap heavily and share the same obstacle context.
    """
    uniq = torch.unique(env_ids)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uniq))
    n_val = max(1, int(round(len(uniq) * val_fraction)))
    val_envs = set(uniq[perm[:n_val]].tolist())
    is_val = torch.tensor([int(e) in val_envs for e in env_ids.tolist()])
    idx = torch.arange(len(env_ids))
    return idx[~is_val], idx[is_val]


class SLDataset:
    """Concatenated windows from one or more collected maps.

    Memory note
    -----------
    Windowing expands the source data by roughly ``seq_len / window_stride``
    (about 15x at the default seq_len=40, stride=2): four maps of ~90 MiB become
    ~1.4 GiB of float32 tensors.  ``cfg.train.quantise_obs`` stores the
    observations as ``uint8`` instead, cutting that ~4x at the cost of a
    ~1.3 cm quantisation step on the range channels (which span 3.2 m).
    """

    def __init__(self, cfg: SLConfig, maps: Sequence[MapData]):
        self.cfg = cfg
        self.quantised = bool(cfg.train.quantise_obs)
        warmup_frames = max(1, cfg.data.warmup_steps // cfg.data.lidar_decimation)
        obs_all, tgt_all, head_all, pos_all, env_all, src_all = [], [], [], [], [], []
        offset = 0
        self.maps = list(maps)
        for m in maps:
            w = build_windows(
                m.obs,
                m.target,
                m.done,
                m.heading,
                m.pos,
                seq_len=cfg.train.seq_len,
                warmup_frames=warmup_frames,
                stride=cfg.train.window_stride,
            )
            obs_all.append(self._encode(w["obs"]))
            tgt_all.append(w["target"])
            head_all.append(w["heading"])
            pos_all.append(w["pos"])
            env_all.append(w["env_id"] + offset)  # keep env ids globally unique
            src_all.append(torch.full((w["obs"].shape[0],), len(src_all), dtype=torch.long))
            offset += m.obs.shape[1]

        self.obs = torch.cat(obs_all)
        self.target = torch.cat(tgt_all)
        self.heading = torch.cat(head_all)
        self.pos = torch.cat(pos_all)
        self.env_id = torch.cat(env_all)
        self.map_id = torch.cat(src_all)
        self.train_idx, self.val_idx = env_split(
            self.env_id, cfg.train.val_fraction, cfg.train.split_seed
        )

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Quantise observations to uint8 when enabled (range is within [-1, 1]).

        The range channels occupy [0, 1] and the ego channels [-1, 1], so a
        single affine map ``(x + 1) / 2 -> [0, 255]`` covers both.
        """
        if not self.quantised:
            return obs
        return torch.clamp(((obs + 1.0) * 127.5).round(), 0, 255).to(torch.uint8)

    def _decode(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.to(torch.float32) / 127.5 - 1.0 if self.quantised else obs

    def batch(self, idx: torch.Tensor) -> torch.Tensor:
        """Materialise a batch of windows as float32, ready for the model."""
        return self._decode(self.obs[idx])

    def __len__(self) -> int:
        return self.obs.shape[0]

    def summary(self) -> Dict:
        tgt = self.target.reshape(-1, 5)
        corr = np.corrcoef(tgt.numpy().T)
        return {
            "n_windows": int(self.obs.shape[0]),
            "n_train": int(self.train_idx.numel()),
            "n_val": int(self.val_idx.numel()),
            "n_envs": int(self.env_id.unique().numel()),
            "obs_min": float(self.obs.min()),
            "obs_max": float(self.obs.max()),
            "target_mean": [round(float(x), 4) for x in tgt.mean(0)],
            "target_std": [round(float(x), 4) for x in tgt.std(0)],
            "target_var_mean": float(tgt.var(0).mean()),
            "corr_fw_fl": float(corr[0, 3]),
            "corr_bw_bl": float(corr[2, 4]),
            "frac_saturated_1": float((tgt > 0.995).float().mean()),
            "frac_saturated_0": float((tgt < 0.005).float().mean()),
        }
