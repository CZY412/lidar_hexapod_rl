"""Hexagon envelope geometry helpers for ``envelope_adaptive_2``.

This module implements the geometric core described in README sections
2.2.3 and 2.8:

* six-vertex body-frame hexagon construction (legacy-compatible order),
* exact half-plane margin offset (not the legacy radial approximation),
* convex point-in-hex tests,
* grid collision ratio against a 2D occupancy map,
* 5-parameter -> 8-dimensional condition conversion reusing the shared
  ``apply_env_morphology_priors`` implementation.

The legacy rule-based implementation in ``envelope_adaptive/envelope_computer.py``
is intentionally not modified; this module only mirrors its vertex order and
uses the shared spider_envelop condition spec for prior derivation.
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch import Tensor

from legged_gym.envs.el_4090.envelope_adaptive_2 import _contracts as ea2c
from legged_gym.utils.envelop.network.haa_swing_range import (
    EnvelopeConditionSpec,
    apply_env_morphology_priors,
)

# Contract vertex order: B, D, F, E, C, A.
_HEX_VERTEX_NAMES: Tuple[str, ...] = (
    "B",
    "D",
    "F",
    "E",
    "C",
    "A",
)


def compute_hex_vertices(
    front_width: Tensor,
    middle_width: Tensor,
    back_width: Tensor,
    forward_limit: Tensor,
    backward_limit: Tensor,
) -> Tensor:
    """Build the 6-vertex body-frame hexagon in legacy-compatible order.

    Vertex order is ``B, D, F, E, C, A`` (counter-clockwise for valid
    envelope parameters):

    .. code-block:: text

        B=( forward_limit,  front_width)
        D=(0,              middle_width)
        F=( backward_limit, back_width)
        E=( backward_limit, -back_width)
        C=(0,              -middle_width)
        A=( forward_limit,  -front_width)

    Args:
        front_width: Half-width at the forward edge, shape ``(...,)``.
        middle_width: Half-width at the middle edge, shape ``(...,)``.
        back_width: Half-width at the backward edge, shape ``(...,)``.
        forward_limit: Forward x limit, shape ``(...,)``.
        backward_limit: Backward x limit (negative), shape ``(...,)``.

    Returns:
        Vertices with shape ``(..., 6, 2)``. The last dimension is ``(x, y)``.
    """
    z = torch.zeros_like(forward_limit)
    B = torch.stack([forward_limit, front_width], dim=-1)
    D = torch.stack([z, middle_width], dim=-1)
    F = torch.stack([backward_limit, back_width], dim=-1)
    E_v = torch.stack([backward_limit, -back_width], dim=-1)
    C = torch.stack([z, -middle_width], dim=-1)
    A = torch.stack([forward_limit, -front_width], dim=-1)
    return torch.stack([B, D, F, E_v, C, A], dim=-2)


def offset_hexagon(vertices: Tensor, margin: float) -> Tensor:
    """Return the exact half-plane offset of a convex hexagon.

    Each edge is translated outward by ``margin`` along its unit outward
    normal; the offset vertices are the intersections of adjacent translated
    edge lines.  This is the precise offset used by the M1 collision
    geometry (README 2.8), unlike the legacy radial vertex scaling.
    Vertices with collinear adjacent edges (valid degenerate convex forms
    such as the rectangle-like minimum envelope) are supported: their offset
    positions are the original vertex translated by ``margin`` along the
    shared edge normal, which lies on the exact offset boundary.

    Args:
        vertices: Convex CCW polygon vertices, shape ``(..., V, 2)``.
        margin: Non-negative offset distance in the same units as vertices.

    Returns:
        Offset polygon vertices with the same shape as ``vertices``.
    """
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    if vertices.shape[-1] != 2:
        raise ValueError("vertices must have last dimension 2 (x, y)")

    v = vertices
    v_next = torch.roll(v, shifts=-1, dims=-2)  # next vertex for each edge
    edges = v_next - v  # (..., V, 2)
    lengths = torch.norm(edges, dim=-1, keepdim=True).clamp_min(1e-12)
    # Outward normal for CCW polygon: rotate edge clockwise by 90 degrees.
    normals = torch.stack([edges[..., 1], -edges[..., 0]], dim=-1) / lengths
    constants = (normals * v).sum(dim=-1) + margin  # (..., V)

    # Offset vertex ``i`` is the intersection of offset edge ``i-1`` and
    # offset edge ``i``.  Rolling the previous line into position i keeps
    # the returned vertex order identical to the input order.
    prev_normals = torch.roll(normals, shifts=1, dims=-2)
    prev_constants = torch.roll(constants, shifts=1, dims=-1)

    a1 = prev_normals[..., 0]
    b1 = prev_normals[..., 1]
    c1 = prev_constants
    a2 = normals[..., 0]
    b2 = normals[..., 1]
    c2 = constants

    det = a1 * b2 - a2 * b1  # (..., V)
    parallel = det.abs() < 1e-12

    # A vertex whose two adjacent edges are collinear (e.g. a rectangle
    # represented in the six-vertex B,D,F,E,C,A order) has no unique line
    # intersection.  Its offset position is the original vertex translated by
    # ``margin`` along the shared outward edge normal, which lies exactly on
    # the offset boundary.
    safe_det = torch.where(parallel, torch.ones_like(det), det)
    x = (c1 * b2 - c2 * b1) / safe_det
    y = (a1 * c2 - a2 * c1) / safe_det
    line_intersections = torch.stack([x, y], dim=-1)
    collinear_offset = v + margin * normals
    return torch.where(parallel.unsqueeze(-1), collinear_offset, line_intersections)


def point_in_hex(pts_xy: Tensor, vertices: Tensor) -> Tensor:
    """Test whether points lie inside (or on the boundary of) a convex hexagon.

    The hexagon is assumed to be convex and counter-clockwise.  Boundary
    points are considered inside (``>= 0`` on all edge cross products).

    Args:
        pts_xy: Points with shape ``(..., N, 2)``.
        vertices: Hexagon vertices with shape ``(..., 6, 2)`` or ``(6, 2)``.
            Leading batch dimensions broadcast with ``pts_xy``.

    Returns:
        Boolean mask with shape ``batch_shape + (N,)``, where ``batch_shape``
        is the broadcast of the leading batch dimensions of ``pts_xy`` and
        ``vertices``.
    """
    if pts_xy.shape[-1] < 2:
        raise ValueError("pts_xy must have at least 2 columns (x, y)")
    if vertices.shape[-1] != 2:
        raise ValueError("vertices must have last dimension 2 (x, y)")

    pts = pts_xy[..., :2]
    verts = vertices[..., :2]

    batch_shape = torch.broadcast_shapes(pts.shape[:-2], verts.shape[:-2])
    pts_b = pts.expand(batch_shape + pts.shape[-2:])
    verts_b = verts.expand(batch_shape + verts.shape[-2:])

    num_vertices = verts_b.shape[-2]
    next_idx = (torch.arange(num_vertices, device=verts_b.device) + 1) % num_vertices
    edges = verts_b[..., next_idx, :] - verts_b  # (batch, V, 2)

    rel = pts_b.unsqueeze(-2) - verts_b.unsqueeze(-3)  # (batch, N, V, 2)
    # cross2d(edge, rel) = edge.x * rel.y - edge.y * rel.x
    cross = (
        edges.unsqueeze(-3)[..., 0] * rel[..., 1]
        - edges.unsqueeze(-3)[..., 1] * rel[..., 0]
    )
    return (cross >= 0).all(dim=-1)


def _default_world_to_grid_fn(
    world_min_xy: float = ea2c.EA2_WORLD_MIN_XY,
    resolution_m: float = ea2c.EA2_RESOLUTION_M,
) -> Callable[[Tensor], Tensor]:
    """Return the canonical world->grid index function for the fixed EA2 map."""

    def world_to_grid_fn(world_xy: Tensor) -> Tensor:
        return torch.floor((world_xy - world_min_xy) / resolution_m).long()

    return world_to_grid_fn


def collision_cell_ratio(
    hex_vertices_world_xy: Tensor,
    occupancy: Tensor | object,
    world_to_grid_fn: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Compute ``covered occupied cells / covered cells`` for each hexagon.

    A cell is considered covered when its center lies inside the (possibly
    offset) hexagon.  The denominator is protected with ``eps`` so empty
    coverage yields zero instead of division by zero.

    Args:
        hex_vertices_world_xy: Hexagon vertices in world coordinates, shape
            ``(..., 6, 2)``.
        occupancy: 2D occupancy array with shape ``(H, W)``.  Accepts
            ``numpy.ndarray`` or ``torch.Tensor``; nonzero entries are
            occupied.
        world_to_grid_fn: Callable mapping world ``(..., 2)`` coordinates to
            integer grid indices ``(..., 2)``.  Defaults to the canonical EA2
            mapping ``ix = floor((x + 6.0) / 0.1)``, ``iy = floor((y + 6.0) /
            0.1)``.

    Returns:
        Collision ratio tensor with the same leading batch shape as
        ``hex_vertices_world_xy`` (or a scalar tensor for a single hexagon).
    """
    if occupancy.ndim != 2:
        raise ValueError("occupancy must be a 2D array (H, W)")
    occ = torch.as_tensor(occupancy, dtype=torch.bool)
    height, width = occ.shape

    if world_to_grid_fn is None:
        world_to_grid_fn = _default_world_to_grid_fn()

    # Build all cell-center world coordinates once, vectorized over the grid.
    # The canonical EA2 grid is used: cell (iy, ix) has world center
    # (world_min + (ix + 0.5) * res, world_min + (iy + 0.5) * res).
    # ``world_to_grid_fn`` is then applied to the flattened centers so the
    # caller-supplied mapping is the authority for which occupancy cells are
    # addressed (this supports custom maps that share the EA2 grid layout).
    ix = torch.arange(width, dtype=torch.float32)
    iy = torch.arange(height, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(ix, iy, indexing="xy")  # (H, W)
    centers = torch.stack(
        [
            ea2c.EA2_WORLD_MIN_XY + (grid_x + 0.5) * ea2c.EA2_RESOLUTION_M,
            ea2c.EA2_WORLD_MIN_XY + (grid_y + 0.5) * ea2c.EA2_RESOLUTION_M,
        ],
        dim=-1,
    ).reshape(-1, 2)

    grid_indices = world_to_grid_fn(centers)
    if isinstance(grid_indices, (tuple, list)):
        # tuple/list form follows the path_planner convention (iy, ix).
        if len(grid_indices) != 2:
            raise ValueError(
                "world_to_grid_fn must return a 2-tuple (iy, ix) or a tensor"
            )
        grid_iy, grid_ix = grid_indices
    else:
        # tensor form follows the README convention: columns are (ix, iy).
        if grid_indices.shape != centers.shape:
            raise ValueError(
                "world_to_grid_fn must return grid indices with shape (..., 2)"
            )
        grid_ix = grid_indices[..., 0]
        grid_iy = grid_indices[..., 1]
    grid_ix = grid_ix.long().clamp(0, width - 1)
    grid_iy = grid_iy.long().clamp(0, height - 1)
    flat_indices = grid_iy * width + grid_ix
    occ_flat = occ.reshape(-1)[flat_indices]

    inside = point_in_hex(centers, hex_vertices_world_xy)

    eps = 1e-6
    if inside.dim() == 1:
        covered = inside.sum(dtype=torch.float32)
        occupied_covered = (inside & occ_flat).sum(dtype=torch.float32)
        return occupied_covered / (covered + eps)

    covered = inside.sum(dim=-1, dtype=torch.float32)  # (...,)
    occupied_covered = (inside & occ_flat.unsqueeze(0)).sum(
        dim=-1, dtype=torch.float32
    )
    return occupied_covered / (covered + eps)


def envelope_params_to_condition(
    params5: Tensor,
    spec: EnvelopeConditionSpec,
) -> Tensor:
    """Convert 5 symmetric envelope parameters to the 8-dim condition tensor.

    The conversion follows README 2.6: append three zero placeholder prior
    values and let the shared ``apply_env_morphology_priors`` derive the real
    directional-ratio priors.  This guarantees M1 uses exactly the same
    downstream prior logic as the locomotion environment.

    Args:
        params5: Envelope parameters with shape ``(..., 5)`` in order
            ``[front_width, middle_width, back_width, forward_limit,
            backward_limit]``.
        spec: ``EnvelopeConditionSpec`` loaded from the spider_envelop config
            (see ``load_envelope_condition_spec``).

    Returns:
        Condition tensor with shape ``(..., 8)``.  The first five columns are
        unchanged and the last three columns are the derived morphology
        priors.
    """
    if params5.shape[-1] != 5:
        raise ValueError(f"Expected params5 width 5, got {params5.shape[-1]}")
    placeholders = torch.zeros_like(params5[..., :3])
    condition8 = torch.cat([params5, placeholders], dim=-1)
    return apply_env_morphology_priors(condition8, spec)
