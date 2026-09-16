#!/usr/bin/env python
"""Offline audit of the A* path-noise module for EA2 (CPU only, no training).

This script does NOT modify training.  It replicates the real training
pipeline exactly:

  1. map generation with the current ``El4090EA2Cfg`` map/obstacle parameters
     (seed = PPO seed), including spawn-cell computation;
  2. start/goal sampling and ``plan_path`` (A* -> LOS -> 0.2 m resampling ->
     bounded lateral noise with rejection sampling -> corner metadata);
  3. the batched stop-and-turn kinematics of
     ``EL_4090_EA2._step_kinematics_batched`` at 50 Hz;
  4. the collision-reward geometry ``hex_collision_terms`` (34 body samples,
     margin=0.10, soft_margin=0.10) along the executed trajectory.

It then compares noise OFF (current stage-1 config ``noise_amp_range=[0,0]``)
against the early noise implementation (``noise_amp_range=[0.15,0.25]`` from
``_contracts.PathCfg``) and reports:

  * motion continuity: stop-and-turn frequency, moving-time fraction,
    effective advance speed, heading-vs-tangent error;
  * clearance: does the noisy path keep gaps larger than the minimum
    envelope, both for the ideal centerline and for the *executed*
    trajectory (heading error + in-place rotation sweep);
  * noise randomness/smoothness: RMS offset, fallback-to-reference ratio,
    per-step offset jumps, lag-1 autocorrelation, corners induced by noise;
  * rigor of the noise acceptance conditions: position curvature vs R_min,
    white-noise retry spikes, episode-to-episode noise diversity.

Usage (from legged_gym/legged_gym):
    python tests/ea2/validate_ea2_path_noise.py [--paths 48] [--episodes 24]
"""

from __future__ import annotations

import argparse
import time

import isaacgym  # noqa: F401  (must precede torch)

import numpy as np
import torch
from scipy.ndimage import label as nd_label
from scipy.ndimage import sum as nd_sum

from legged_gym.envs.el_4090.envelope_adaptive_2 import _contracts as ea2c
from legged_gym.envs.el_4090.envelope_adaptive_2._contracts import (
    MapGenCfg,
    PathCfg,
    PillarFieldCfg,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_config import (
    El4090EA2Cfg,
    El4090EA2CfgPPO,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_geometry import (
    hex_body_sample_points,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.map_generator import (
    generate_map,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.path_planner import (
    _apply_path_noise,
    _astar,
    _compute_segment_dirs,
    _cumulative_arc_lengths,
    _detect_corners,
    _low_pass_noise_offsets,
    _resample_polyline,
    _simplify_path,
    plan_path,
    wrap_to_pi,
)

WORLD_MIN = ea2c.EA2_WORLD_MIN_XY
RES = ea2c.EA2_RESOLUTION_M

DT = 0.02                       # 50 Hz control
CORNER_THRESH = 0.05            # rad, _detect_corners / env align threshold
ENVELOPE_MIN = torch.tensor([0.3, 0.3, 0.3, 0.6, -0.6])
ENVELOPE_MID = torch.tensor([0.45, 0.5, 0.45, 0.75, -0.75])
ENVELOPE_MAX = torch.tensor([0.6, 0.7, 0.6, 0.9, -0.6])
MARGIN = 0.10                   # cfg.envelope.margin
SOFT_MARGIN = 0.10              # cfg.envelope.soft_margin


# ---------------------------------------------------------------------------
# Setup (mirrors EL_4090_EA2._create_sim/_init_buffers)
# ---------------------------------------------------------------------------

def build_world(seed: int):
    map_cfg = MapGenCfg(
        size_m=float(El4090EA2Cfg.map.size_m),
        resolution_m=float(El4090EA2Cfg.map.resolution_m),
        grid_shape=tuple(El4090EA2Cfg.map.grid_shape),
        boundary_occupied=bool(El4090EA2Cfg.map.boundary_occupied),
        ground_margin_m=float(El4090EA2Cfg.map.ground_margin_m),
        inflation_m=float(El4090EA2Cfg.map.inflation_m),
        inflation_cells=int(El4090EA2Cfg.map.inflation_cells),
        n_tiles=int(El4090EA2Cfg.map.n_tiles),
        tile_size_m=float(El4090EA2Cfg.map.tile_size_m),
        border_size_m=float(El4090EA2Cfg.map.border_size_m),
        min_free_component_ratio=float(El4090EA2Cfg.map.min_free_component_ratio),
        max_gen_attempts=int(El4090EA2Cfg.map.max_gen_attempts),
        n_validation_paths=int(El4090EA2Cfg.map.n_validation_paths),
        min_solved_ratio=float(El4090EA2Cfg.map.min_solved_ratio),
        path_near_obstacle_ratio=float(El4090EA2Cfg.map.path_near_obstacle_ratio),
        near_obstacle_range=tuple(El4090EA2Cfg.map.near_obstacle_range),
        require_constraint_primitive=bool(El4090EA2Cfg.map.require_constraint_primitive),
    )
    pillar_cfg = PillarFieldCfg(
        count_min=int(El4090EA2Cfg.obstacles.pillar_count_min),
        count_max=int(El4090EA2Cfg.obstacles.pillar_count_max),
        size_x_min=float(El4090EA2Cfg.obstacles.pillar_size_x_min),
        size_x_max=float(El4090EA2Cfg.obstacles.pillar_size_x_max),
        size_y_min=float(El4090EA2Cfg.obstacles.pillar_size_y_min),
        size_y_max=float(El4090EA2Cfg.obstacles.pillar_size_y_max),
        height_min=float(El4090EA2Cfg.obstacles.pillar_height_min),
        height_max=float(El4090EA2Cfg.obstacles.pillar_height_max),
        min_separation=float(El4090EA2Cfg.obstacles.pillar_min_separation),
        center_clear_radius=float(El4090EA2Cfg.obstacles.pillar_center_clear_radius),
        spawn_radius=float(El4090EA2Cfg.obstacles.pillar_spawn_radius),
        allow_height_variation=bool(El4090EA2Cfg.obstacles.pillar_allow_height_variation),
    )
    t0 = time.perf_counter()
    map_data = generate_map(map_cfg, pillar_cfg, seed=seed)
    t_gen = time.perf_counter() - t0

    free = map_data.inflated == 0
    labels, n_comp = nd_label(free)
    sizes = nd_sum(free, labels, index=range(1, n_comp + 1))
    largest = int(np.argmax(sizes)) + 1
    free_cells = np.argwhere(labels == largest)

    half_tile = float(El4090EA2Cfg.map.size_m) / 2.0 - float(El4090EA2Cfg.map.border_size_m)
    xs = WORLD_MIN + (free_cells[:, 1] + 0.5) * RES
    ys = WORLD_MIN + (free_cells[:, 0] + 0.5) * RES
    mask = (xs >= -half_tile) & (xs <= half_tile) & (ys >= -half_tile) & (ys <= half_tile)
    spawn_cells = free_cells[mask]

    return map_data, spawn_cells, t_gen


def cell_to_world(cell) -> np.ndarray:
    return np.asarray(
        (WORLD_MIN + (cell[1] + 0.5) * RES, WORLD_MIN + (cell[0] + 0.5) * RES),
        dtype=np.float64,
    )


class Sampler:
    """Mirrors _sample_start_xy / _sample_goal_xy / _min_obstacle_distance_world."""

    def __init__(self, map_data, spawn_cells, rng: np.random.Generator):
        self.occ = map_data.occupancy
        self.infl = map_data.inflated
        self.dist = map_data.distance_field
        self.spawn = spawn_cells
        self.rng = rng

    def clearance(self, xy) -> float:
        ix = int(np.floor((float(xy[0]) - WORLD_MIN) / RES))
        iy = int(np.floor((float(xy[1]) - WORLD_MIN) / RES))
        if ix < 0 or iy < 0 or ix >= self.dist.shape[1] or iy >= self.dist.shape[0]:
            return float("inf")
        return float(self.dist[iy, ix])

    def start(self) -> np.ndarray:
        idx = self.spawn[self.rng.integers(0, self.spawn.shape[0])]
        return cell_to_world(idx)

    def goal(self) -> np.ndarray:
        need = float(El4090EA2Cfg.path.goal_min_obstacle_dist)
        for _ in range(200):
            idx = self.spawn[self.rng.integers(0, self.spawn.shape[0])]
            xy = cell_to_world(idx)
            if self.clearance(xy) >= need:
                return xy
        raise RuntimeError("no valid goal in 200 attempts")

    def start_goal(self):
        return self.start(), self.goal()


def make_path_cfg(noise_amp, noise_fc) -> PathCfg:
    p = El4090EA2Cfg.path
    return PathCfg(
        speed_range=tuple(p.speed_range),
        resample_time_s=float(p.resample_time_s),
        delta_target_deg_range=tuple(p.delta_target_deg_range),
        omega_max=float(p.omega_max),
        k_p=float(p.k_p),
        min_turn_radius=float(p.min_turn_radius),
        resample_dist=float(p.resample_dist),
        goal_min_obstacle_dist=float(p.goal_min_obstacle_dist),
        min_path_len=float(p.min_path_len),
        noise_amp_range=tuple(noise_amp),
        noise_fc_hz=float(noise_fc),
        noise_retries=int(p.noise_retries),
    )


# ---------------------------------------------------------------------------
# Path batch planning (mirrors _plan_paths_parallel serial path)
# ---------------------------------------------------------------------------

def plan_batch_collect(sampler: Sampler, cfg: PathCfg, n: int, seed0: int):
    paths, t_plan, n_fail, n_resample = [], 0.0, 0, 0
    for k in range(n):
        attempts = 0
        while True:
            attempts += 1
            if attempts > 1:
                n_resample += 1
            start, goal = sampler.start_goal()
            rng = np.random.default_rng(seed0 + 1_000_003 * k + attempts)
            t0 = time.perf_counter()
            try:
                data = plan_path(sampler.occ, sampler.infl, start, goal, cfg, rng)
                t_plan += time.perf_counter() - t0
                paths.append(data)
                break
            except (ValueError, RuntimeError):
                n_fail += 1
                t_plan += time.perf_counter() - t0
                if attempts >= 40:
                    raise
    return paths, t_plan, n_fail, n_resample


# ---------------------------------------------------------------------------
# Batched kinematics replica of _step_kinematics_batched (numpy, scalar per env)
# ---------------------------------------------------------------------------

class KinematicsSim:
    """Scalar replica of the env's batched stop-and-turn kinematics."""

    def __init__(self, path, v: float, k_p: float, omega_max: float):
        self.pts = path.points
        self.arc = path.arc
        self.seg_dirs = path.segment_dirs
        self.corner_arcs = (
            np.asarray(path.corner_arcs, dtype=np.float64)
            if path.corner_arcs is not None and len(path.corner_arcs) > 0
            else np.zeros((0,))
        )
        self.corner_targets = (
            np.asarray(path.corner_targets, dtype=np.float64)
            if path.corner_targets is not None and len(path.corner_targets) > 0
            else np.zeros((0,))
        )
        self.v = v
        self.k_p = k_p
        self.omega_max = omega_max
        self.s = 0.0
        self.heading = float(self.seg_dirs[0])   # _reset_one_env initial heading
        self.turn_in_place = False
        self.turn_target = 0.0
        self.last_arc = float(self.arc[-1])

    def query(self, s: float):
        """Mirror PathBatch.query for one env."""
        idx = int(np.searchsorted(self.arc, s, side="right")) - 1
        idx = max(0, min(idx, len(self.arc) - 2))
        a0, a1 = self.arc[idx], self.arc[idx + 1]
        p0, p1 = self.pts[idx], self.pts[idx + 1]
        t = (s - a0) / max(a1 - a0, 1e-12)
        xy = p0 * (1.0 - t) + p1 * t
        seg_idx = min(idx, len(self.seg_dirs) - 1)
        tangent = float(self.seg_dirs[seg_idx])
        cc = int(np.searchsorted(self.corner_arcs, s, side="right"))
        if cc < len(self.corner_arcs):
            next_corner = float(self.corner_arcs[cc])
            next_target = float(self.corner_targets[cc])
            has_next = True
        else:
            next_corner = float("inf")
            next_target = 0.0
            has_next = False
        return xy, tangent, next_corner, next_target, has_next

    def step(self, dt: float):
        """Advance one control step; returns a dict of per-step telemetry."""
        delta_target = 0.0
        if self.turn_in_place:
            delta = wrap_to_pi(self.heading - self.turn_target)
            omega_cmd = self.k_p * wrap_to_pi(delta_target - float(delta))
            omega = float(np.clip(omega_cmd, -self.omega_max, self.omega_max))
            self.heading = float(wrap_to_pi(self.heading + omega * dt))
            delta_new = float(wrap_to_pi(self.heading - self.turn_target))
            aligned = abs(delta_new) < CORNER_THRESH
            self.tangent = self.turn_target
            self.turn_in_place = not aligned     # must not advance in same step
            return {
                "mode": "turn",
                "xy": None,
                "delta": abs(delta_new),
                "omega": omega,
                "moving": False,
                "corner_step": False,
            }

        xy_old, _, next_corner, next_target, has_next = self.query(self.s)
        near_corner = has_next and (self.s + self.v * dt >= next_corner)
        if near_corner:
            s_corner = min(next_corner + 1e-4, self.last_arc)
            self.s = s_corner
            xy, _, _, _, _ = self.query(s_corner)
            self.tangent = next_target
            not_aligned = abs(float(wrap_to_pi(self.heading - next_target))) >= CORNER_THRESH
            if not_aligned:
                self.turn_in_place = True
                self.turn_target = next_target
            return {
                "mode": "corner",
                "xy": xy,
                "delta": float(wrap_to_pi(self.heading - next_target)),
                "omega": 0.0,
                "moving": False,
                "corner_step": True,
            }

        s_after = self.s + self.v * dt
        reached = s_after >= self.last_arc
        if reached:
            xy, _, _, _, _ = self.query(self.last_arc)
            self.s = self.last_arc
            return {
                "mode": "reached",
                "xy": xy,
                "delta": 0.0,
                "omega": 0.0,
                "moving": False,
                "corner_step": False,
            }

        self.s = s_after
        xy, tangent, _, _, _ = self.query(s_after)
        delta = float(wrap_to_pi(self.heading - tangent))
        omega_cmd = self.k_p * wrap_to_pi(delta_target - delta)
        omega = float(np.clip(omega_cmd, -self.omega_max, self.omega_max))
        self.heading = float(wrap_to_pi(self.heading + omega * dt))
        delta_new = float(wrap_to_pi(self.heading - tangent))
        self.tangent = tangent
        return {
            "mode": "advance",
            "xy": xy,
            "delta": abs(delta_new),
            "omega": omega,
            "moving": True,
            "corner_step": False,
        }


def run_episode(sampler: Sampler, cfg: PathCfg, seed: int, episode_s: float):
    """Simulate one training episode: initial path + soft replans on arrival.

    Mirrors reset (aligned heading) + _batch_replan turn-in-place chaining.
    Returns per-step telemetry plus path-level counters.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(episode_s / DT)
    tele = {
        "mode": [],
        "delta": np.zeros(n_steps),
        "omega": np.zeros(n_steps),
        "pos": np.zeros((n_steps, 2)),
        "heading": np.zeros(n_steps),
    }
    corners_hit = 0
    stops = 0
    distance = 0.0
    n_paths = 0

    # initial path (heading already aligned; _reset_one_env)
    while True:
        start, goal = sampler.start_goal()
        try:
            path = plan_path(sampler.occ, sampler.infl, start, goal, cfg, rng)
            break
        except (ValueError, RuntimeError):
            pass
    sim = KinematicsSim(path, v=1.0, k_p=cfg.k_p, omega_max=cfg.omega_max)
    n_paths += 1
    prev_pos = path.points[0].copy()

    for i in range(n_steps):
        out = sim.step(DT)
        tele["mode"].append(out["mode"])
        tele["delta"][i] = out["delta"]
        tele["omega"][i] = out["omega"]
        if out["xy"] is not None:
            tele["pos"][i] = out["xy"]
            distance += float(np.linalg.norm(out["xy"] - prev_pos))
            prev_pos = out["xy"]
        else:
            tele["pos"][i] = prev_pos
        tele["heading"][i] = sim.heading
        if out["mode"] == "corner":
            corners_hit += 1
            if sim.turn_in_place:
                stops += 1
        elif out["mode"] == "turn":
            if i == 0 or tele["mode"][i - 1] != "turn":
                stops += 1
        elif out["mode"] == "reached":
            # _batch_replan: new path from the reached goal, turn-in-place.
            while True:
                start = np.asarray(sim.pts[-1], dtype=np.float64)
                goal = sampler.goal()
                try:
                    path = plan_path(sampler.occ, sampler.infl, start, goal, cfg, rng)
                    break
                except (ValueError, RuntimeError):
                    pass
            sim = KinematicsSim(path, v=1.0, k_p=cfg.k_p, omega_max=cfg.omega_max)
            sim.turn_in_place = True
            sim.turn_target = float(sim.seg_dirs[0])
            n_paths += 1

    modes = np.asarray(tele["mode"])
    return {
        "tele": tele,
        "modes": modes,
        "corners_hit": corners_hit,
        "stops": stops,
        "distance": distance,
        "n_paths": n_paths,
        "moving_frac": float(np.mean(modes == "advance")),
        "turn_frac": float(np.mean(modes == "turn")),
        "effective_speed": distance / (len(modes) * DT),
    }


# ---------------------------------------------------------------------------
# Envelope clearance along executed trajectory (hex_collision_terms replica)
# ---------------------------------------------------------------------------

_SAMPLE_CACHE = {}


def body_samples(params: torch.Tensor) -> torch.Tensor:
    key = tuple(params.tolist())
    if key not in _SAMPLE_CACHE:
        _SAMPLE_CACHE[key] = hex_body_sample_points(params.unsqueeze(0))[0]
    return _SAMPLE_CACHE[key]


def clearance_stats(map_data, pos: np.ndarray, heading: np.ndarray, params: torch.Tensor, stride: int = 2):
    """Mirror hex_collision_terms over recorded trajectory poses."""
    samples = body_samples(params).numpy()      # (34, 2)
    dist = map_data.distance_field
    h, w = dist.shape
    cos_h = np.cos(heading[::stride])
    sin_h = np.sin(heading[::stride])
    px, py = samples[:, 0], samples[:, 1]
    wx = pos[::stride, 0, None] + cos_h[:, None] * px[None, :] - sin_h[:, None] * py[None, :]
    wy = pos[::stride, 1, None] + sin_h[:, None] * px[None, :] + cos_h[:, None] * py[None, :]
    ix = np.clip(np.floor((wx - WORLD_MIN) / RES).astype(np.int64), 0, w - 1)
    iy = np.clip(np.floor((wy - WORLD_MIN) / RES).astype(np.int64), 0, h - 1)
    clearance = dist[iy, ix]                    # (M, 34)
    min_clear = clearance.min(axis=1)
    violation = np.clip((MARGIN - clearance) / SOFT_MARGIN, 0.0, 1.0)
    hard = violation.max(axis=1)                # worst-sample violation
    dense = violation[:, :24].sum(axis=1)       # edge sum (dense reward)
    return {
        "min_clearance": min_clear,
        "reward_active_frac": float(np.mean(min_clear < MARGIN - 1e-9)),
        "hard_contact_frac": float(np.mean(min_clear <= 0.0)),
        "mean_dense": float(dense.mean()),
        "mean_hard": float(hard.mean()),
    }


# ---------------------------------------------------------------------------
# Noise mechanism audit (replicates the _build_path_data first attempt)
# ---------------------------------------------------------------------------

def noise_offsets_for_path(sampler: Sampler, cfg: PathCfg, rng: np.random.Generator):
    """Plan one path and return (reference, noisy, offsets) of the first
    noise attempt, i.e. the low-pass offsets before any fallback machinery."""
    start, goal = sampler.start_goal()
    raw = _astar(sampler.infl, start, goal)
    simple = _simplify_path(raw, sampler.infl)
    ref, _ = _resample_polyline(simple, cfg.resample_dist)
    amp = rng.uniform(cfg.noise_amp_range[0], cfg.noise_amp_range[1])
    offsets = _low_pass_noise_offsets(ref.shape[0], amp, rng, cfg.noise_fc_hz, cfg.resample_dist)
    noisy = _apply_path_noise(ref, sampler.infl, cfg, rng, smoothing_passes=0)
    # realised offsets projected on the reference normals
    n = min(ref.shape[0], noisy.shape[0])
    dx = np.diff(ref, axis=0)
    tang = np.zeros_like(ref)
    tang[1:-1] = dx[:-1] + dx[1:]
    tang[0] = dx[0]
    tang[-1] = dx[-1]
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-12)
    normals = np.stack((-tang[:, 1], tang[:, 0]), axis=1)
    realised = np.einsum("ij,ij->i", noisy[:n] - ref[:n], normals[:n])
    return ref, noisy, offsets, realised


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def path_stats(paths):
    lens = np.asarray([p.arc[-1] for p in paths])
    ncorners = np.asarray([len(p.corner_arcs) if p.corner_arcs is not None else 0 for p in paths])
    npts = np.asarray([p.points.shape[0] for p in paths])
    per_m = ncorners / np.maximum(lens, 1e-9)
    # segment direction changes of the returned polyline
    frac_turn_seg = []
    d90 = []
    for p in paths:
        d = _compute_segment_dirs(p.points)
        if len(d) < 2:
            frac_turn_seg.append(0.0)
            continue
        turn = np.abs(wrap_to_pi(np.diff(d)))
        frac_turn_seg.append(float(np.mean(turn > CORNER_THRESH)))
        d90.append(float(np.quantile(turn, 0.9)))
    # curvature of the returned position polyline vs R_min
    curv_excess = []
    for p in paths:
        ds = np.linalg.norm(np.diff(p.points, axis=0), axis=1)
        dirs = p.segment_dirs
        if len(dirs) < 2:
            curv_excess.append(0.0)
            continue
        dyaw = np.abs(wrap_to_pi(np.diff(dirs)))          # turn at interior vertices
        ds_mid = 0.5 * (ds[:-1] + ds[1:])                 # arc step across each vertex
        ok = ds_mid > 1e-9
        curv_excess.append(float(np.max(dyaw[ok] / ds_mid[ok])))  # vs 1/R_min=1
    return {
        "len_mean": float(lens.mean()),
        "len_p90": float(np.quantile(lens, 0.9)),
        "pts_mean": float(npts.mean()),
        "corners_mean": float(ncorners.mean()),
        "corners_per_m": float(per_m.mean()),
        "corners_per_m_p90": float(np.quantile(per_m, 0.9)),
        "frac_seg_turning": float(np.mean(frac_turn_seg)),
        "turn_p90_deg": float(np.degrees(np.mean(d90))),
        "max_curv_vs_rmin": float(np.max(curv_excess)),
        "median_curv_vs_rmin": float(np.median(curv_excess)),
    }


def corner_sweep_stats(map_data, paths):
    """In-place-rotation sweep risk at stop-and-turn corners.

    Rotating in place sweeps the hexagon through the full disk of radius
    r_sweep = max sample radius.  A corner whose clearance is below r_sweep
    cannot host a full min-envelope rotation without overlap.
    """
    dist = map_data.distance_field
    h, w = dist.shape
    r_sweep = float(np.linalg.norm(body_samples(ENVELOPE_MIN).numpy(), axis=1).max())
    clear_at_corner = []
    risky = 0
    total = 0
    turn_angles = []
    for p in paths:
        if p.corner_arcs is None or len(p.corner_arcs) == 0:
            continue
        dirs = p.segment_dirs
        turns = np.abs(wrap_to_pi(np.diff(dirs)))
        for k, ca in enumerate(np.asarray(p.corner_arcs, dtype=np.float64)):
            idx = int(np.searchsorted(p.arc, ca, side="right")) - 1
            idx = max(0, min(idx, len(p.points) - 1))
            x, y = p.points[idx]
            ix = int(np.floor((x - WORLD_MIN) / RES))
            iy = int(np.floor((y - WORLD_MIN) / RES))
            c = float(dist[iy, ix])
            clear_at_corner.append(c)
            total += 1
            if c < r_sweep:
                risky += 1
            if k < len(turns):
                turn_angles.append(float(turns[k]))
    return {
        "r_sweep_min_envelope": r_sweep,
        "corners_total": total,
        "corners_clearance_below_sweep_frac": risky / max(total, 1),
        "corner_clearance_mean": float(np.mean(clear_at_corner)) if clear_at_corner else 0.0,
        "corner_turn_deg_mean": float(np.degrees(np.mean(turn_angles))) if turn_angles else 0.0,
        "corner_turn_gt10deg_frac": float(np.mean(np.asarray(turn_angles) > np.radians(10.0))) if turn_angles else 0.0,
    }


def print_header(txt: str):
    print("\n" + "=" * 78)
    print(txt)
    print("=" * 78)


def fmt_stats(d: dict, indent: str = "  ") -> str:
    lines = []
    for k, v in d.items():
        lines.append(f"{indent}{k:.<44} {v:.3f}" if isinstance(v, float) else f"{indent}{k:.<44} {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=48, help="paths per config")
    parser.add_argument("--episodes", type=int, default=16, help="episodes per config")
    parser.add_argument("--episode-s", type=float, default=30.0)
    args = parser.parse_args()

    seed = int(El4090EA2CfgPPO.seed)
    print(f"Generating training map (seed={seed}, cfg=El4090EA2Cfg) ...")
    map_data, spawn_cells, t_gen = build_world(seed)
    print(f"  map generated in {t_gen:.1f}s; acceptance={ {k: round(v, 3) for k, v in map_data.acceptance.items() if k in ('n_rects','largest_free_component_ratio','near_obstacle_ratio')} }")
    print(f"  spawn cells: {spawn_cells.shape[0]}")

    # analytic check of the low-pass filter strength at the configured point
    alpha = 1.0 - np.exp(-2.0 * np.pi * 1.0 * 0.2)
    print(f"\n_low_pass_noise_offsets at cfg (fc=1.0 Hz, ds=0.2 m): alpha={alpha:.3f}")
    print("  (AR(1) coefficient a=1-alpha={:.3f}; noise correlation length ~{:.1f} samples = {:.2f} m)".format(
        1 - alpha, 1.0 / max(alpha, 1e-9), 0.2 / max(alpha, 1e-9)))

    configs = {
        "noise OFF (current stage-1 cfg)": (El4090EA2Cfg.path.noise_amp_range, El4090EA2Cfg.path.noise_fc_hz),
        "noise ON  (early impl [0.15,0.25], fc=1.0)": ([0.15, 0.25], 1.0),
        "noise ON  ([0.15,0.25], fc=0.2)": ([0.15, 0.25], 0.2),
        "noise ON  ([0.05,0.15], fc=1.0)": ([0.05, 0.15], 1.0),
    }

    results = {}
    for name, (amp, fc) in configs.items():
        print_header(f"CONFIG: {name}")
        cfg = make_path_cfg(amp, fc)
        rng = np.random.default_rng(seed + 17)
        sampler = Sampler(map_data, spawn_cells, rng)

        t0 = time.perf_counter()
        paths, t_plan, n_fail, n_resample = plan_batch_collect(sampler, cfg, args.paths, seed + 100)
        t_all = time.perf_counter() - t0
        ps = path_stats(paths)
        print(f"  planned {len(paths)} paths in {t_all:.1f}s "
              f"({1000.0 * t_plan / max(len(paths), 1):.1f} ms/path planning, "
              f"{1000.0 * (t_all - t_plan) / max(len(paths), 1):.1f} ms/path sampling)")
        print(f"  plan failures: {n_fail} (extra start/goal resamples: {n_resample})")
        print(fmt_stats(ps))

        # centerline clearance of the returned paths (noise-gap guarantee)
        clear_sampler = Sampler(map_data, spawn_cells, np.random.default_rng(0))
        cl = []
        for p in paths:
            vals = [clear_sampler.clearance(pt) for pt in p.points]
            cl.append(min(vals))
        cl = np.asarray(cl)
        print(f"  centerline min clearance: mean {cl.mean():.3f} m, min {cl.min():.3f} m "
              f"(min-envelope half-width 0.3 + margin 0.05 = 0.35)")

        cs = corner_sweep_stats(map_data, paths)
        print("  stop-and-turn corner sweep risk (in-place rotation):")
        print(fmt_stats(cs, indent="    "))

        # ---- episode simulation ----
        ep_stats = []
        for e in range(args.episodes):
            ep = run_episode(sampler, cfg, seed + 5000 + 7919 * e, args.episode_s)
            t = ep["tele"]
            adv = ep["modes"] == "advance"
            delta_adv = t["delta"][adv] if adv.any() else np.zeros(1)
            clear = clearance_stats(map_data, t["pos"], t["heading"], ENVELOPE_MIN)
            clear_max = clearance_stats(map_data, t["pos"], t["heading"], ENVELOPE_MAX)
            # 10 Hz LiDAR frames: fresh-frame content
            n_frames = int(args.episode_s * 10.0)
            frame_idx = (np.arange(n_frames) * 5).clip(0, len(ep["modes"]) - 1)
            f_adv = float(np.mean(ep["modes"][frame_idx] == "advance"))
            f_turn = float(np.mean(ep["modes"][frame_idx] == "turn"))
            f_corner = float(np.mean(ep["modes"][frame_idx] == "corner"))
            ep_stats.append({
                "moving_frac": ep["moving_frac"],
                "turn_frac": ep["turn_frac"],
                "stops": ep["stops"],
                "corners_hit": ep["corners_hit"],
                "distance": ep["distance"],
                "effective_speed": ep["effective_speed"],
                "delta_adv_mean": float(np.mean(delta_adv)),
                "delta_adv_p95": float(np.quantile(delta_adv, 0.95)),
                "clear_active": clear["reward_active_frac"],
                "clear_hard": clear["hard_contact_frac"],
                "clear_min": float(clear["min_clearance"].min()),
                "maxenv_active": clear_max["reward_active_frac"],
                "frames_adv": f_adv,
                "frames_turn": f_turn + f_corner,
            })
        keys = ep_stats[0].keys()
        agg = {k: float(np.mean([e[k] for e in ep_stats])) for k in keys}
        print("  episode simulation (30 s, soft replan chaining, mean over "
              f"{args.episodes} episodes):")
        print(fmt_stats(agg, indent="    "))

        # ---- noise mechanism audit ----
        if amp[1] > 0.0:
            offs_all, real_all, fallback, spikes = [], [], [], []
            for k in range(min(24, args.paths)):
                r2 = np.random.default_rng(seed + 913 + 31 * k)
                s2 = Sampler(map_data, spawn_cells, r2)
                ref, noisy, offsets, realised = noise_offsets_for_path(s2, cfg, r2)
                interior = slice(1, len(offsets) - 1)
                offs_all.append(offsets[interior])
                real_all.append(realised[interior])
                fallback.append(float(np.mean(np.abs(realised[interior]) < 1e-6)))
                d = np.diff(realised)
                if len(d):
                    spikes.append(float(np.mean(np.abs(d) > 0.5 * (amp[0] + amp[1]) / 2)))
            offs = np.concatenate(offs_all)
            real = np.concatenate(real_all)
            ac = float(np.corrcoef(offs[:-1], offs[1:])[0, 1])
            print("  noise mechanism audit (first-attempt offsets, interior points):")
            print(fmt_stats({
                "amp range (drawn once per path)": f"[{amp[0]}, {amp[1]}]",
                "intended offset RMS (pre-rejection)": float(np.sqrt(np.mean(offs ** 2))),
                "realised offset RMS (post-rejection)": float(np.sqrt(np.mean(real ** 2))),
                "fallback-to-reference ratio": float(np.mean(fallback)),
                "lag-1 autocorr of intended offsets": ac,
                "steps with |doffset| > 0.5*amp": float(np.mean(spikes)),
                "offset-change |d| per 0.2m step (mean)": float(np.mean(np.abs(np.concatenate([np.diff(o) for o in offs_all])))),
            }, indent="    "))
        results[name] = (ps, agg)

    # ---- episode-to-episode noise diversity (same start/goal, many seeds) ----
    print_header("NOISE DIVERSITY: same start/goal, 12 different seeds")
    cfg_on = make_path_cfg([0.15, 0.25], 1.0)
    rng = np.random.default_rng(seed + 31337)
    sampler = Sampler(map_data, spawn_cells, rng)
    start, goal = sampler.start_goal()
    pts = []
    for k in range(12):
        r2 = np.random.default_rng(seed + 1000 + k)
        raw = _astar(sampler.infl, start, goal)
        simple = _simplify_path(raw, sampler.infl)
        ref, _ = _resample_polyline(simple, cfg_on.resample_dist)
        noisy = _apply_path_noise(ref, sampler.infl, cfg_on, r2, smoothing_passes=0)
        pts.append(noisy)
    n = min(p.shape[0] for p in pts)
    base = pts[0][:n]
    diffs = [float(np.mean(np.linalg.norm(p[:n] - base, axis=1))) for p in pts[1:]]
    print(fmt_stats({
        "path point count (shortest)": int(n),
        "mean |P_k - P_0| vs seed 0 (m)": float(np.mean(diffs)),
        "identical-point ratio vs seed 0": float(np.mean([np.mean(np.all(np.isclose(p[:n], base, atol=1e-9), axis=1)) for p in pts[1:]])),
    }))

    print_header("SUMMARY (mean over episodes)")
    for name, (ps, agg) in results.items():
        print(f"\n{name}")
        print(fmt_stats({
            "corners per meter (path)": ps["corners_per_m"],
            "moving fraction": agg["moving_frac"],
            "turn-in-place fraction": agg["turn_frac"],
            "stops per 30 s episode": agg["stops"],
            "effective speed (m/s)": agg["effective_speed"],
            "mean |heading-tangent| while advancing (deg)": np.degrees(agg["delta_adv_mean"]),
            "min-envelope: collision-reward-active step frac": agg["clear_active"],
            "min-envelope: hard contact step frac": agg["clear_hard"],
            "max-envelope: collision-reward-active step frac": agg["maxenv_active"],
            "10Hz frames while advancing": agg["frames_adv"],
        }))


if __name__ == "__main__":
    main()
