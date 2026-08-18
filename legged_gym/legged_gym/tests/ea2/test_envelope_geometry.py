"""Tests for ``envelope_geometry`` (hexagon construction/offset/collision/condition).

Covers the task-specified cases:

* vertex values match the legacy ``_build_hex_edges`` formula,
* point inside/outside/boundary classification,
* exact half-plane margin offset boundary behavior,
* collision ratio on a small synthetic occupancy grid,
* 5-parameter -> 8-condition conversion reuses ``apply_env_morphology_priors``.
"""

from __future__ import annotations

# The repo's envs/__init__.py imports isaacgym, which requires torch NOT to
# have been imported yet.  Importing the envelope_geometry package first lets
# isaacgym initialize torch itself; all later torch imports are then safe.
from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_geometry import (
    collision_cell_ratio,
    compute_hex_vertices,
    envelope_params_to_condition,
    offset_hexagon,
    point_in_hex,
)
from legged_gym.envs.el_4090.envelope_adaptive_2 import _contracts as c
from legged_gym.envs.el_4090.envelope_adaptive.envelope_computer import (
    _build_hex_edges,
)
from legged_gym.utils.envelop.network.haa_swing_range import (
    apply_env_morphology_priors,
    load_envelope_condition_spec,
)

import numpy as np
import torch


def test_vertex_values_match_legacy_formula() -> None:
    """compute_hex_vertices must equal legacy _build_hex_edges output."""
    front_width = torch.tensor([0.35, 0.55, 0.30])
    middle_width = torch.tensor([0.45, 0.65, 0.30])
    back_width = torch.tensor([0.38, 0.42, 0.30])
    forward_limit = torch.tensor([0.65, 0.80, 0.60])
    backward_limit = torch.tensor([-0.70, -0.65, -0.60])

    ours = compute_hex_vertices(
        front_width, middle_width, back_width, forward_limit, backward_limit
    )
    legacy = _build_hex_edges(
        forward_limit, backward_limit, front_width, middle_width, back_width
    )
    assert ours.shape == (3, 6, 2)
    torch.testing.assert_close(ours, legacy)

    # Single-scalar input also works and matches the same formula.
    single = compute_hex_vertices(
        torch.tensor(0.4),
        torch.tensor(0.5),
        torch.tensor(0.4),
        torch.tensor(0.7),
        torch.tensor(-0.7),
    )
    legacy_single = _build_hex_edges(
        torch.tensor(0.7),
        torch.tensor(-0.7),
        torch.tensor(0.4),
        torch.tensor(0.5),
        torch.tensor(0.4),
    )
    # Legacy helper stacks scalar 1-D vertices along dim=1 and returns
    # (2, 6); our contract fixes the canonical shape as (6, 2).
    torch.testing.assert_close(single, legacy_single.T)
    assert single.shape == (6, 2)


def test_point_inside_outside_and_boundary() -> None:
    """Point-in-hex classifies interior, exterior and boundary correctly."""
    vertices = compute_hex_vertices(
        torch.tensor(0.40),
        torch.tensor(0.50),
        torch.tensor(0.40),
        torch.tensor(0.70),
        torch.tensor(-0.70),
    )
    points = torch.tensor(
        [
            [0.00, 0.00],   # interior center
            [0.30, 0.00],   # interior front half
            [-0.30, 0.20],  # interior rear half
            [0.70, 0.40],   # vertex B, boundary
            [0.70, 0.00],   # front edge, boundary
            [0.00, 0.50],   # vertex D, boundary
            [0.80, 0.00],   # outside forward
            [-0.80, 0.00],  # outside backward
            [0.00, 0.60],   # outside above middle
            [0.00, -0.60],  # outside below middle
        ],
        dtype=torch.float32,
    )
    mask = point_in_hex(points, vertices)
    expected = torch.tensor([True, True, True, True, True, True, False, False, False, False])
    torch.testing.assert_close(mask, expected)


def _edge_unit_normal(v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """Outward unit normal for a CCW polygon edge (v0 -> v1)."""
    edge = v1 - v0
    normal = torch.tensor([edge[1].item(), -edge[0].item()], dtype=torch.float32)
    return normal / torch.norm(normal)


def test_margin_offset_exact_boundary() -> None:
    """A point exactly margin away along an edge normal lies on the offset boundary."""
    margin = 0.05
    vertices = compute_hex_vertices(
        torch.tensor(0.40),
        torch.tensor(0.50),
        torch.tensor(0.40),
        torch.tensor(0.70),
        torch.tensor(-0.70),
    )
    offset = offset_hexagon(vertices, margin)
    assert offset.shape == vertices.shape

    # The original polygon should be strictly inside the offset polygon.
    assert bool(point_in_hex(vertices, offset).all())

    for i in range(6):
        v0 = vertices[i]
        v1 = vertices[(i + 1) % 6]
        normal = _edge_unit_normal(v0, v1)
        midpoint = (v0 + v1) / 2.0
        on_boundary = midpoint + margin * normal
        just_outside = on_boundary + 0.001 * normal

        assert bool(point_in_hex(on_boundary.unsqueeze(0), offset).all()), (
            f"edge {i} boundary point should be inside/on the offset hexagon"
        )
        assert not bool(point_in_hex(just_outside.unsqueeze(0), offset).all()), (
            f"edge {i} point just outside should be rejected"
        )


def test_offset_hexagon_handles_collinear_minimum_envelope() -> None:
    """The README minimum envelope (rectangle-like 6-vertex hex) must offset."""
    vertices = compute_hex_vertices(
        torch.tensor(0.30),
        torch.tensor(0.30),
        torch.tensor(0.30),
        torch.tensor(0.60),
        torch.tensor(-0.60),
    )
    offset = offset_hexagon(vertices, 0.05)
    assert offset.shape == vertices.shape

    # The exact half-plane offset of the minimum envelope is a 0.05 m
    # expanded rectangle.  Collinear B-D-F / E-C-A vertices stay on the
    # offset boundary in the same vertex order.
    expected = torch.tensor(
        [
            [0.65, 0.35],
            [0.00, 0.35],
            [-0.65, 0.35],
            [-0.65, -0.35],
            [0.00, -0.35],
            [0.65, -0.35],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(offset, expected)

    # The original minimum envelope is strictly inside the offset polygon.
    assert bool(point_in_hex(vertices, offset).all())

    # A point exactly margin away from the top edge normal lies on the
    # offset boundary; moving slightly further rejects it.
    boundary_top = torch.tensor([[0.00, 0.35]], dtype=torch.float32)
    assert bool(point_in_hex(boundary_top, offset).all())
    outside_top = torch.tensor([[0.00, 0.36]], dtype=torch.float32)
    assert not bool(point_in_hex(outside_top, offset).all())

    boundary_bottom = torch.tensor([[0.00, -0.35]], dtype=torch.float32)
    assert bool(point_in_hex(boundary_bottom, offset).all())
    outside_bottom = torch.tensor([[0.00, -0.36]], dtype=torch.float32)
    assert not bool(point_in_hex(outside_bottom, offset).all())


def test_collision_cell_ratio_synthetic_occupancy() -> None:
    """A hex covering 4 cells with 2 occupied has collision ratio 0.5."""
    occupancy = np.zeros(c.EA2_GRID_SHAPE, dtype=np.uint8)
    # Cell centers for ix=60,61 / iy=60,61 are (0.05,0.05),(0.15,0.05),
    # (0.05,0.15),(0.15,0.15). Mark the two lower cells occupied.
    occupancy[60, 60] = 1
    occupancy[60, 61] = 1

    # Degenerate CCW hexagon equal to the world rectangle [0, 0.2] x [0, 0.2].
    # The six vertices are collinear in pairs, matching the legacy B,D,F,E,C,A
    # order while still being a valid convex polygon for coverage tests.
    hex_vertices = torch.tensor(
        [
            [0.20, 0.20],  # B
            [0.10, 0.20],  # D
            [0.00, 0.20],  # F
            [0.00, 0.00],  # E
            [0.10, 0.00],  # C
            [0.20, 0.00],  # A
        ],
        dtype=torch.float32,
    )
    ratio = collision_cell_ratio(hex_vertices, occupancy)
    assert ratio.dim() == 0
    assert abs(ratio.item() - 0.5) < 1e-5

    # No occupied cell inside -> ratio 0.
    empty_occ = np.zeros(c.EA2_GRID_SHAPE, dtype=np.uint8)
    ratio_empty = collision_cell_ratio(hex_vertices, empty_occ)
    assert abs(ratio_empty.item() - 0.0) < 1e-5

    # All covered cells occupied -> ratio 1.
    full_occ = np.zeros(c.EA2_GRID_SHAPE, dtype=np.uint8)
    full_occ[60, 60] = 1
    full_occ[60, 61] = 1
    full_occ[61, 60] = 1
    full_occ[61, 61] = 1
    ratio_full = collision_cell_ratio(hex_vertices, full_occ)
    assert abs(ratio_full.item() - 1.0) < 1e-5


def test_envelope_params_to_condition_matches_apply_priors() -> None:
    """5->8 conversion must equal apply_env_morphology_priors on zero-padded input."""
    spec = load_envelope_condition_spec(c.ENVELOPE_SPEC_CONFIG_PATH)
    params5 = torch.tensor(
        [
            [0.40, 0.55, 0.35, 0.75, -0.70],
            [0.30, 0.30, 0.30, 0.60, -0.90],
            [0.60, 0.70, 0.60, 0.90, -0.60],
        ],
        dtype=torch.float32,
    )
    actual = envelope_params_to_condition(params5, spec)
    expected = apply_env_morphology_priors(
        torch.cat([params5, torch.zeros_like(params5[..., :3])], dim=-1),
        spec,
    )
    assert actual.shape == (3, 8)
    torch.testing.assert_close(actual, expected)

    # First five columns are the input parameters themselves.
    torch.testing.assert_close(actual[..., :5], params5)

    # Priors are in [0, 1].
    assert bool((actual[..., 5:] >= 0.0).all())
    assert bool((actual[..., 5:] <= 1.0).all())
