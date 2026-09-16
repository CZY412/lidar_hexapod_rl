"""Cascade perception tests (warp raycast + torch; no Isaac sim).

* T1a: observation assembly reuses EA2's own helpers (identity by
  construction, guarded here against signature/divisor drift).
* T1b: flat-ground raycast at the training mount pose must reproduce the
  analytic ``slant_ranges`` stored in the channel table.
* Timing: 10 Hz cadence with first-step refresh; stale/empty contract.

Run: ``python legged_gym/tests/ea2/cascade/test_cascade_perception.py``.
"""

import isaacgym  # noqa: F401

from types import SimpleNamespace

import numpy as np
import torch

from legged_gym.envs.el_4090.envelope_adaptive_2 import _contracts as ea2c
from legged_gym.envs.el_4090.envelope_adaptive_2.airy_mount import (
    load_selected_channels,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_env import (
    assemble_observation,
)
from legged_gym.envs.el_4090.envelope_cascade_83.ea2_perception import (
    Ea2Perception,
    yaw_quat,
)

_DEVICE = "cuda:0"


def _make_cfg(**overrides):
    cfg = SimpleNamespace(
        update_frequency_hz=10.0,
        far_plane=60.0,
        channel_file=str(ea2c.EA2_SELECTED_CHANNELS_FILE),
        enable_sensor_noise=False,
        pixel_std_dev_multiplier=0.02,
        pixel_dropout_prob=0.02,
        offset_pos=(0.7, 0.0, -0.05),
        sensor_offset_rpy=(0.0, np.pi / 2.0 + 0.1, 0.0),
        yaw_only=False,
        ego_scales=(1.5, 1.0, 1.5),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _make_perception(num_envs: int = 2, **overrides) -> Ea2Perception:
    plane = 100.0
    vertices = np.array(
        [
            [-plane, -plane, 0.0],
            [plane, -plane, 0.0],
            [plane, plane, 0.0],
            [-plane, plane, 0.0],
        ],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return Ea2Perception(
        num_envs=num_envs,
        device=_DEVICE,
        cfg=_make_cfg(**overrides),
        dt=0.02,
        terrain_vertices=vertices,
        terrain_triangles=triangles,
    )


def test_perception_requires_mesh():
    import pytest

    with pytest.raises(ValueError):
        Ea2Perception(num_envs=1, device=_DEVICE, cfg=_make_cfg(), dt=0.02)


def test_refresh_cadence_first_step_then_every_fifth():
    perception = _make_perception(num_envs=1)
    flags = [perception.refresh() for _ in range(11)]
    assert flags[0] is True  # timer initialised to dec-1 -> first step refreshes
    assert flags[1:5] == [False, False, False, False]
    assert flags[5] is True
    assert flags[10] is True
    assert perception.refresh_count == 3


def test_flat_ground_reproduces_analytic_slant_table():
    channels = load_selected_channels()
    slant = channels["slant_ranges"].to(_DEVICE)
    perception = _make_perception(num_envs=1)
    base_pos = torch.tensor([[0.0, 0.0, ea2c.EA2_BASE_HEIGHT_M]], device=_DEVICE)
    base_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=_DEVICE)
    perception.update_pose(base_pos, base_quat)
    assert perception.refresh() is True
    assert not bool(perception.stale.any())
    # image stores raw slant distances; empty rows would carry range_max
    error = (perception.range_image[0] - slant).abs()
    assert float(error.max()) < 1e-3, f"max slant error {float(error.max())}"
    cloud = perception.debug_cloud()
    assert cloud is not None and cloud[0].shape == (1, 187, 3)
    # body-frame landing zone: the mount looks at the ground ahead
    # (ground plane sits at z=-0.52 in body frame; sensor at (0.7, 0, -0.05))
    body = cloud[0][0]
    assert bool(((body[:, 0] >= 0.6) & (body[:, 0] <= 3.7)).all()), "hits outside ground region"
    assert bool((body[:, 1].abs() <= 1.1).all()), "hits outside lateral region"
    assert float((body[:, 2] + 0.52).abs().max()) < 0.05, (
        f"points not on the body-frame ground plane: z∈[{float(body[:,2].min()):.2f},"
        f"{float(body[:,2].max()):.2f}] (pre-fix they point at the sky)"
    )


def test_mark_stale_until_next_scan():
    perception = _make_perception(num_envs=2)
    base_pos = torch.zeros(2, 3, device=_DEVICE)
    base_pos[:, 2] = ea2c.EA2_BASE_HEIGHT_M
    base_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=_DEVICE).repeat(2, 1)
    perception.update_pose(base_pos, base_quat)
    while not perception.refresh():
        pass
    perception.mark_stale(torch.tensor([0], device=_DEVICE))
    assert abs(float(perception.range_image[0, 0]) - perception.range_max) < 1e-6
    assert bool(perception.stale[0]) and not bool(perception.stale[1])
    while not perception.refresh():
        pass
    assert not bool(perception.stale.any())


def test_yaw_only_projection_matches_full_quat_mechanics():
    from isaacgym.torch_utils import quat_mul

    perception = _make_perception(num_envs=1, yaw_only=True)
    tilt = torch.tensor(
        [[0.1, -0.2, 0.0, float(np.sqrt(1.0 - 0.01 - 0.04))]],
        dtype=torch.float32,
        device=_DEVICE,
    )
    tilt = tilt / tilt.norm(dim=-1, keepdim=True)
    perception.update_pose(torch.zeros(1, 3, device=_DEVICE), tilt)
    expected_quat = quat_mul(yaw_quat(tilt), perception._offset_quat)
    assert torch.allclose(perception.sensor_quat, expected_quat, atol=1e-6)

    perception_full = _make_perception(num_envs=1, yaw_only=False)
    perception_full.update_pose(torch.zeros(1, 3, device=_DEVICE), tilt)
    assert torch.allclose(perception_full.sensor_quat, quat_mul(tilt, perception_full._offset_quat), atol=1e-6)


def test_obs_assembly_matches_ea2_helper():
    perception = _make_perception(num_envs=2)
    ego = torch.tensor([[1.0, 0.2, -0.3], [0.0, 0.0, 0.0]], device=_DEVICE)
    obs = perception.observe(ego)
    assert obs.shape == (2, 190)
    expected = assemble_observation(
        perception.range_image,
        ego,
        max_range=perception.range_max,
        ego_scales=(1.5, 1.0, 1.5),
    )
    assert torch.equal(obs, expected)
    # ego columns normalised by the EA2 scales
    scales = torch.tensor([1.5, 1.0, 1.5], device=_DEVICE)
    assert torch.allclose(obs[:, 187:190], ego / scales)


def test_noise_disabled_by_default_keeps_dists_deterministic():
    perception = _make_perception(num_envs=1)
    base_pos = torch.tensor([[0.0, 0.0, ea2c.EA2_BASE_HEIGHT_M]], device=_DEVICE)
    perception.update_pose(base_pos, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=_DEVICE))
    first = None
    while first is None:
        if perception.refresh():
            first = perception.range_image.clone()
    perception._timer = perception._decimation - 1
    assert perception.refresh()
    assert torch.equal(first, perception.range_image)


def _make_pillar_scene_arrays():
    """Flat ground plus a 1x1x2 m box at x∈[1.5,2.5], y∈[-0.5,0.5] (m)."""
    plane = 60.0
    vertices = [
        [-plane, -plane, 0.0], [plane, -plane, 0.0],
        [plane, plane, 0.0], [-plane, plane, 0.0],
    ]
    triangles = [[0, 1, 2], [0, 2, 3]]

    x0, x1, y0, y1, h = 1.5, 2.5, -0.5, 0.5, 2.0
    base = len(vertices)
    vertices += [
        [x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0],  # bottom ring
        [x0, y0, h], [x1, y0, h], [x1, y1, h], [x0, y1, h],          # top ring
    ]

    def quad(a, b, c, d):
        triangles.append([base + a, base + b, base + c])
        triangles.append([base + a, base + c, base + d])

    quad(1, 2, 6, 5)  # +x front face
    quad(0, 4, 7, 3)  # -x back face
    quad(2, 3, 7, 6)  # +y
    quad(0, 1, 5, 4)  # -y
    quad(4, 5, 6, 7)  # +z top (bottom skipped: flush with ground)
    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(triangles, dtype=np.int32),
    )


def test_single_pillar_shortens_facing_channels():
    """T1b directional sanity: only channels facing the pillar shorten."""
    vertices, triangles = _make_pillar_scene_arrays()
    perception = Ea2Perception(
        num_envs=1, device=_DEVICE, cfg=_make_cfg(), dt=0.02,
        terrain_vertices=vertices, terrain_triangles=triangles,
    )
    slant = load_selected_channels()["slant_ranges"].to(_DEVICE)
    perception.update_pose(
        torch.tensor([[0.0, 0.0, ea2c.EA2_BASE_HEIGHT_M]], device=_DEVICE),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=_DEVICE),
    )
    assert perception.refresh() is True
    dist = perception.range_image[0]

    shortened = slant - dist
    facing = shortened > 0.05
    n_facing = int(facing.sum())
    assert n_facing >= 5, f"only {n_facing} channels saw the pillar"
    # pillar front face at x=1.5, sensor at x=0.7: pillar hits stay >= ~0.8 m
    # (ground channels can legitimately be shorter: straight-down ray = 0.47)
    assert float(dist[facing].min()) >= 0.7, (
        f"suspicious pillar dist {float(dist[facing].min())}"
    )
    # ground channels are untouched by the pillar.  The 1 m pillar 0.8 m
    # ahead shadows most far rows (shadow half-width reaches the ±1 m grid
    # edge by x≈2.3), but the 4 front rows (x<=1.4 < pillar front 1.5)
    # must survive intact: 4 rows x 17 cols = 68 channels.
    unchanged = int((shortened.abs() <= 1e-3).sum())
    assert unchanged >= 40, f"only {unchanged} ground channels kept"


def test_sensor_noise_path_changes_dists_and_stays_in_range():
    perception = _make_perception(num_envs=1, enable_sensor_noise=True)
    base_pos = torch.tensor([[0.0, 0.0, ea2c.EA2_BASE_HEIGHT_M]], device=_DEVICE)
    perception.update_pose(base_pos, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=_DEVICE))

    slant = load_selected_channels()["slant_ranges"].to(_DEVICE)
    frames = []
    while len(frames) < 6:
        if perception.refresh():
            frames.append(perception.range_image[0].clone())

    for frame in frames:
        assert bool(torch.isfinite(frame).all())
        assert bool(((frame >= 0.0) & (frame <= perception.range_max + 1e-6)).all())
        assert not torch.equal(frame, slant)  # noise actually applied
    # 2% dropout on 187 rays: at least one saturated channel within 6 frames
    saturated = any(bool((frame == perception.range_max).any()) for frame in frames)
    assert saturated, "dropout never saturated a channel in 6 frames"


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc}")
    print(f"===== failures: {failures} =====")
    sys.exit(1 if failures else 0)
