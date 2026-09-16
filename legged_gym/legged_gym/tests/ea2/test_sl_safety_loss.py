"""Unit tests for the differentiable safety (collision) loss.

Pins the three properties the plan review flagged as failure-prone:

1. *bilinear gradient exists, nearest gradient is exactly zero* -- the env
   reward's nearest-cell lookup is piecewise constant in the sample position
   and therefore unusable as an SL loss; this test is the regression guard
   that keeps the loss on the bilinear path.
2. *floor-pinned masking* -- frames where even the minimum envelope violates
   must be excluded (their gradient would be a constant collapse pressure).
3. *per-map grouping* -- a batch spanning two maps must index each frame
   against its own distance field.
"""

from __future__ import annotations

import pytest
import torch

try:
    from . import _ea2_testlib as tl
except ImportError:
    import _ea2_testlib as tl

from legged_gym.envs.el_4090.envelope_adaptive_2.sl.train import (
    batch_safety_loss,
    s_to_params,
)


def _pose_near_pillar(device="cpu"):
    """Heading/pos with the wide envelope's front tip inside the 2x2 pillar.

    Pillar spans [-1, 1]^2; stand at x=1.8 facing -x -- the front vertex
    reaches x=0.9, deep inside the obstacle (clearance 0 -> violation 1).
    """
    head = torch.tensor([torch.pi])          # facing -x
    pos = torch.tensor([[1.8, 0.0]])
    return head.to(device), pos.to(device)


def _wide_s(device="cpu"):
    """Fully open normalised extents (max envelope violates near the pillar)."""
    return torch.ones(1, 5, device=device)


def test_bilinear_gradient_exists_nearest_is_zero():
    """The documented trap: nearest lookup has exactly zero grad; bilinear
    must have a nonzero one pointing along the shrinking direction."""
    from legged_gym.envs.el_4090.envelope_adaptive_2.envelope_geometry import (
        _hex_sample_violations,
    )

    df, _ = tl.pillar_field_2x2()
    dft = torch.as_tensor(df)
    head, pos = _pose_near_pillar()
    s = _wide_s().requires_grad_(True)
    params = s_to_params(s)

    v_near = _hex_sample_violations(
        params, head, pos, dft, margin=0.10, soft_margin=0.10, sampling="nearest"
    )
    # the gather detaches entirely: no grad_fn, hence no gradient path at all
    assert v_near.requires_grad is False, "nearest lookup must carry no gradient"

    s2 = _wide_s().requires_grad_(True)
    v_bil = _hex_sample_violations(
        s_to_params(s2), head, pos, dft, margin=0.10, soft_margin=0.10,
        sampling="bilinear",
    )
    assert float(v_bil.max()) > 0.0, "wide envelope must violate near the pillar"
    assert v_bil.requires_grad is True
    v_bil.sum().backward()
    assert float(s2.grad.abs().sum()) > 0.0, "bilinear lookup must carry gradient"


def test_floor_pinned_frames_are_masked_out():
    """When even the minimum envelope violates, the loss must be exactly zero
    (no collapse gradient), while a feasible frame keeps its loss."""
    df, _ = tl.pillar_field_2x2()
    dft = torch.as_tensor(df)
    # robot centre INSIDE the pillar: even the min envelope violates
    head = torch.tensor([0.0, 0.0])
    pos = torch.zeros(2, 2)
    s = torch.stack([_wide_s()[0], torch.zeros(5)])
    map_ids = torch.zeros(2, dtype=torch.long)
    dfs = dft.unsqueeze(0)
    heading = torch.zeros(2)
    loss, _ = batch_safety_loss(s, heading, pos, map_ids, dfs)
    assert float(loss) == 0.0

    # a genuinely feasible pose far from everything: also zero (safe), but via
    # the "no violation" path -- probe with a frame that does violate to make
    # sure the mask is the reason, not a broken loss
    head2, pos2 = _pose_near_pillar()
    s2 = torch.cat([torch.zeros(1, 5), _wide_s()])  # frame0=min@center(pinned), frame1=wide@pillar(violating)
    map_ids2 = torch.zeros(2, dtype=torch.long)
    heading2 = torch.cat([torch.zeros(1), head2])
    pos2 = torch.cat([torch.zeros(1, 2), pos2])
    loss2, trig = batch_safety_loss(s2, heading2, pos2, map_ids2, dfs)
    assert float(loss2) > 0.0 and trig > 0.0


def test_batch_grouping_uses_each_frames_own_field():
    """Two maps: frame 0 violates on map A, frame 1 is safe there -- but the
    two frames' clearances must come from different fields."""
    df_violating, _ = tl.pillar_field_2x2()
    df_free = torch.full_like(torch.as_tensor(df_violating), 50.0)  # obstacle-free
    head = torch.tensor([torch.pi, torch.pi])
    pos = torch.tensor([[1.8, 0.0], [1.8, 0.0]])
    s = torch.ones(2, 5)
    map_ids = torch.tensor([0, 1])
    dfs = torch.stack([torch.as_tensor(df_violating), df_free])

    loss, trig = batch_safety_loss(s, head, pos, map_ids, dfs)
    # only the map-A frame contributes; the map-B frame reads a 50 m field
    assert float(loss) > 0.0
    assert abs(trig - 0.5) < 1e-6


def test_s_to_params_matches_deployment_mapping():
    """Endpoints: s=0 -> physical min envelope, s=1 -> physical max envelope."""
    p0 = s_to_params(torch.zeros(1, 5))[0]
    p1 = s_to_params(torch.ones(1, 5))[0]
    assert torch.allclose(p0, torch.tensor([0.3, 0.3, 0.3, 0.6, -0.6]), atol=1e-6)
    assert torch.allclose(p1, torch.tensor([0.6, 0.7, 0.6, 0.9, -0.9]), atol=1e-6)
