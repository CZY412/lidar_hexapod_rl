"""Airy sensor mounting, selected-channel mapping table and body-frame checks.

This module implements the v2 Airy mounting/bucketing contract:

* ``EA2_SENSOR_OFFSET_RPY`` maps the sensor frame into the body frame using the
  same quaternion convention as ``LidarSensor`` (``quat_from_euler_xyz`` then
  ``quat_apply``).
* The mapping table is azimuth-major (``i = az * 96 + el``) and selects
  physical azimuth channels 18..42 and elevation lines 6..95, producing 450
  row-major bucket indices (``flat = row * 25 + col``).
* ``self_check_mapping_table`` verifies the table and the selected body-frame
  ray coverage so mounting mistakes are caught at import/test time.

The quaternion helpers are local reproductions of ``isaacgym.torch_utils``
formulas, so this module remains importable even when the GPU/Isaac runtime is
not available.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Union

import torch

from ._contracts import (
    EA2_AIRY_N_AZIMUTH,
    EA2_AIRY_N_ELEVATION,
    EA2_MAPPING_TABLE_FILE,
    EA2_N_COLS,
    EA2_N_RAYS,
    EA2_N_ROWS,
    EA2_RANGE_DIM,
    EA2_SELECTED_AZ,
    EA2_SELECTED_EL,
    EA2_SENSOR_OFFSET_RPY,
)

PathLike = Union[str, Path]

_DEG2RAD = math.pi / 180.0


def _quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    """Build an xyzw quaternion from XYZ Euler angles (same as torch_utils).

    Args:
        roll: Rotation about x-axis, radians.
        pitch: Rotation about y-axis, radians.
        yaw: Rotation about z-axis, radians.

    Returns:
        Quaternion tensor of shape ``(4,)`` in ``[x, y, z, w]`` order.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return torch.tensor([qx, qy, qz, qw], dtype=torch.float64)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply quaternion rotation(s) to vector(s), matching torch_utils.

    Args:
        q: Quaternion(s) of shape ``(..., 4)`` in ``[x, y, z, w]`` order.
        v: Vector(s) of shape ``(..., 3)``.

    Returns:
        Rotated vector(s) of shape ``(..., 3)``.
    """
    xyz = q[..., :3]
    t = torch.cross(xyz, v, dim=-1) * 2.0
    return v + q[..., 3:] * t + torch.cross(xyz, t, dim=-1)


def build_airy_mapping_table() -> torch.Tensor:
    """Build the 5760 -> 450 Airy bucket mapping table.

    Ray index ``i = az * 96 + el`` follows the Airy C-order flattening.  A ray
    is mapped only when ``az in 18..42`` and ``el in 6..95``.  The mapped value
    is the row-major bucket index ``(el - 6) // 5 * 25 + (az - 18)``; all other
    entries are ``-1``.

    Returns:
        Integer tensor of shape ``(5760,)`` with dtype ``int64``.
    """
    table = torch.full((EA2_N_RAYS,), -1, dtype=torch.int64)

    az = torch.tensor(EA2_SELECTED_AZ, dtype=torch.int64)
    el = torch.tensor(EA2_SELECTED_EL, dtype=torch.int64)

    az_grid = az.view(-1, 1).expand(len(EA2_SELECTED_AZ), len(EA2_SELECTED_EL)).reshape(-1)
    el_grid = el.view(1, -1).expand(len(EA2_SELECTED_AZ), len(EA2_SELECTED_EL)).reshape(-1)

    ray_indices = az_grid * EA2_AIRY_N_ELEVATION + el_grid
    rows = (el_grid - int(EA2_SELECTED_EL[0])) // 5
    cols = az_grid - int(EA2_SELECTED_AZ[0])
    bucket_indices = rows * EA2_N_COLS + cols
    table[ray_indices] = bucket_indices
    return table


def body_frame_ray_directions() -> torch.Tensor:
    """Return all 5760 Airy ray directions expressed in the body frame.

    Uses the same spherical formula and mounting convention as ``LidarSensor``:
    sensor frame ``x = cos(phi)cos(theta)``, ``y = cos(phi)sin(theta)``,
    ``z = sin(phi)`` with theta = 0..360 deg / 60 channels and phi = 0..90 deg /
    96 lines, then applies the body->sensor offset rotation to express each
    direction in the body frame.

    Returns:
        Float tensor of shape ``(5760, 3)`` with dtype ``float32``.
    """
    phi = (
        torch.arange(EA2_AIRY_N_ELEVATION, dtype=torch.float64)
        * (math.pi / 2.0)
        / (EA2_AIRY_N_ELEVATION - 1)
    )
    theta = (
        torch.arange(EA2_AIRY_N_AZIMUTH, dtype=torch.float64)
        * (2.0 * math.pi)
        / EA2_AIRY_N_AZIMUTH
    )[:, None]

    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    sensor_dirs = torch.stack(
        [
            cos_phi[None, :] * cos_theta,
            cos_phi[None, :] * sin_theta,
            sin_phi[None, :].expand(EA2_AIRY_N_AZIMUTH, EA2_AIRY_N_ELEVATION),
        ],
        dim=-1,
    ).reshape(-1, 3)

    offset_q = _quat_from_euler_xyz(*EA2_SENSOR_OFFSET_RPY)
    body_dirs = _quat_apply(
        offset_q.view(1, 4).expand(EA2_N_RAYS, 4), sensor_dirs
    )
    return body_dirs.to(dtype=torch.float32)


def self_check_mapping_table(table: torch.Tensor) -> Dict[str, object]:
    """Run the mandatory Airy mounting/mapping self-checks.

    Asserts the README/contract invariants:

    * table shape is ``(5760,)`` and uses ``int64``;
    * all 25 x 90 selected rays (2250) map to exactly 450 valid row-major
      bucket indices;
    * every selected ray has body-frame ``x > 0``, body azimuth in
      ``[-80, +80]`` deg and body elevation in ``[-25, +70]`` deg;
    * the table is exactly equal to a freshly built table (flatten-index
      formula ``i = az * 96 + el`` and row-major bucket formula agree).

    Args:
        table: Mapping table produced by :func:`build_airy_mapping_table`.

    Returns:
        Dict with summary values for logging/debugging.
    """
    if not isinstance(table, torch.Tensor):
        raise AssertionError("mapping table must be a torch.Tensor")
    if table.shape != (EA2_N_RAYS,):
        raise AssertionError(
            f"mapping table shape must be {(EA2_N_RAYS,)}, got {tuple(table.shape)}"
        )
    if table.dtype != torch.int64:
        raise AssertionError(f"mapping table dtype must be int64, got {table.dtype}")

    expected = build_airy_mapping_table()
    if not torch.equal(table, expected):
        raise AssertionError("mapping table does not match the fresh built table")

    mapped_mask = table >= 0
    n_mapped = int(mapped_mask.sum().item())
    n_expected_rays = len(EA2_SELECTED_AZ) * len(EA2_SELECTED_EL)
    if n_mapped != n_expected_rays:
        raise AssertionError(
            f"expected {n_expected_rays} mapped rays, got {n_mapped}"
        )

    mapped_indices = torch.nonzero(mapped_mask, as_tuple=False).squeeze(1)
    az = mapped_indices // EA2_AIRY_N_ELEVATION
    el = mapped_indices % EA2_AIRY_N_ELEVATION

    if az.min().item() != EA2_SELECTED_AZ[0] or az.max().item() != EA2_SELECTED_AZ[-1]:
        raise AssertionError("selected azimuth channels differ from 18..42")
    if el.min().item() != EA2_SELECTED_EL[0] or el.max().item() != EA2_SELECTED_EL[-1]:
        raise AssertionError("selected elevation lines differ from 6..95")

    rows = (el - int(EA2_SELECTED_EL[0])) // 5
    cols = az - int(EA2_SELECTED_AZ[0])
    expected_buckets = rows * EA2_N_COLS + cols
    if not torch.equal(table[mapped_indices], expected_buckets):
        raise AssertionError("row-major bucket formula mismatch")

    if (
        mapped_indices.numel() != n_expected_rays
        or len(torch.unique(mapped_indices)) != n_expected_rays
    ):
        raise AssertionError(
            "mapped ray index set is not the expected 2250 unique rays"
        )

    n_unique_buckets = int(table[mapped_mask].unique().numel())
    if n_unique_buckets != EA2_RANGE_DIM:
        raise AssertionError(
            f"expected {EA2_RANGE_DIM} unique bucket ids, got {n_unique_buckets}"
        )

    dirs = body_frame_ray_directions()[mapped_indices]
    body_x = dirs[:, 0]
    if not bool((body_x > 0.0).all()):
        raise AssertionError("not all selected rays have body-frame x > 0")

    azimuth_deg = torch.atan2(dirs[:, 1], dirs[:, 0]) / _DEG2RAD
    elevation_deg = torch.asin(
        dirs[:, 2] / torch.norm(dirs, dim=1).clamp_min(1e-6)
    ) / _DEG2RAD

    if azimuth_deg.min().item() < -80.0 or azimuth_deg.max().item() > 80.0:
        raise AssertionError(
            f"selected body azimuth outside [-80, 80] deg: "
            f"[{azimuth_deg.min().item():.3f}, {azimuth_deg.max().item():.3f}]"
        )
    if elevation_deg.min().item() < -25.0 or elevation_deg.max().item() > 70.0:
        raise AssertionError(
            f"selected body elevation outside [-25, 70] deg: "
            f"[{elevation_deg.min().item():.3f}, {elevation_deg.max().item():.3f}]"
        )

    return {
        "table_shape": tuple(table.shape),
        "dtype": str(table.dtype),
        "n_mapped": n_mapped,
        "n_unique_buckets": n_unique_buckets,
        "range_dim": EA2_RANGE_DIM,
        "azimuth_deg_range": (
            float(azimuth_deg.min().item()),
            float(azimuth_deg.max().item()),
        ),
        "elevation_deg_range": (
            float(elevation_deg.min().item()),
            float(elevation_deg.max().item()),
        ),
        "all_body_x_positive": True,
        "mapping_formula_consistent": True,
    }


def save_airy_mapping_table(
    table: torch.Tensor, path: Optional[PathLike] = None
) -> Path:
    """Save the mapping table to ``.pt``.

    Args:
        table: Mapping table tensor to persist.
        path: Destination path; defaults to ``EA2_MAPPING_TABLE_FILE``.

    Returns:
        The resolved path that was written.
    """
    if path is None:
        path = EA2_MAPPING_TABLE_FILE
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(table, path)
    return path


def load_airy_mapping_table(path: Optional[PathLike] = None) -> torch.Tensor:
    """Load a mapping table from ``.pt``.

    Args:
        path: Source path; defaults to ``EA2_MAPPING_TABLE_FILE``.

    Returns:
        The loaded mapping table tensor.
    """
    if path is None:
        path = EA2_MAPPING_TABLE_FILE
    return torch.load(Path(path), map_location="cpu")


def save_body_ray_projection(path: PathLike) -> Optional[Path]:
    """Save a 2D azimuth/elevation projection of selected body-frame rays.

    This is an optional debug aid (README §2.3.5 item 3).  It writes a PNG only
    when an explicit ``path`` is supplied and matplotlib is installed.

    Args:
        path: Destination PNG path.

    Returns:
        The written path, or ``None`` if matplotlib is unavailable.
    """
    path = Path(path)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - environment-specific
        return None

    table = build_airy_mapping_table()
    dirs = body_frame_ray_directions()[table >= 0]
    azimuth_deg = torch.atan2(dirs[:, 1], dirs[:, 0]) / _DEG2RAD
    elevation_deg = torch.asin(
        dirs[:, 2] / torch.norm(dirs, dim=1).clamp_min(1e-6)
    ) / _DEG2RAD

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(azimuth_deg.numpy(), elevation_deg.numpy(), s=6)
    ax.set_xlabel("body azimuth (deg)")
    ax.set_ylabel("body elevation (deg)")
    ax.set_title("Airy selected rays in body frame")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
