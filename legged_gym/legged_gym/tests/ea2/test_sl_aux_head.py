"""Unit tests for the auxiliary memory heads (k-step recall / forward probes).

The riskiest part is the loss slicing: an indexing mistake silently turns the
recall probe into the forward probe (or into a trivial identity), which would
still train -- just without the intended memory pressure.  These tests pin the
direction semantics with constructed targets.
"""

from __future__ import annotations

import pytest
import torch

from legged_gym.envs.el_4090.envelope_adaptive_2.sl.model import EnvelopeNet
from legged_gym.envs.el_4090.envelope_adaptive_2.sl.sl_config import SLConfig, SLModelConfig
from legged_gym.envs.el_4090.envelope_adaptive_2.sl.train import aux_memory_loss


def _seq_targets(L: int, batch: int = 2) -> torch.Tensor:
    """(L, B, 5) targets whose value encodes the frame index (per dim)."""
    t = torch.arange(L, dtype=torch.float32).view(L, 1, 1)
    return t.expand(L, batch, 5).contiguous()


def _shifted(y: torch.Tensor, k: int, mode: str) -> torch.Tensor:
    """A perfect predictor for the given mode (the dropped tail is arbitrary)."""
    L = y.shape[0]
    pred = torch.zeros_like(y)
    if mode == "recall":          # pred[t] = y[t-k]  for t >= k
        pred[k:] = y[:-k]
    else:                         # pred[t] = y[t+k]  for t <= L-1-k
        pred[: L - k] = y[k:]
    return pred


def test_aux_recall_loss_is_zero_for_shifted_prediction():
    """A probe that emits y[t-k] at frame t must score exactly zero."""
    L, k = 40, 7
    y = _seq_targets(L)
    assert float(aux_memory_loss(_shifted(y, k, "recall"), y, k, "recall")) == 0.0


def test_aux_forward_loss_is_zero_for_lead_prediction():
    """A probe that emits y[t+k] at frame t must score exactly zero."""
    L, k = 40, 7
    y = _seq_targets(L)
    assert float(aux_memory_loss(_shifted(y, k, "forward"), y, k, "forward")) == 0.0


def test_aux_losses_are_direction_sensitive():
    """A perfect predictor of one mode must NOT score zero in the other:
    the recall-perfect sequence is off by 2k frames under the forward pairing
    and vice versa, so any indexing swap is caught."""
    L, k = 40, 7
    y = _seq_targets(L)
    rec_pred = _shifted(y, k, "recall")
    fwd_pred = _shifted(y, k, "forward")
    assert float(aux_memory_loss(rec_pred, y, k, "recall")) == 0.0
    assert float(aux_memory_loss(fwd_pred, y, k, "forward")) == 0.0
    assert float(aux_memory_loss(rec_pred, y, k, "forward")) > 0
    assert float(aux_memory_loss(fwd_pred, y, k, "recall")) > 0
    # the unshifted identity must score positive in both modes
    assert float(aux_memory_loss(y, y, k, "recall")) > 0
    assert float(aux_memory_loss(y, y, k, "forward")) > 0


def test_aux_loss_rejects_bad_horizon_and_mode():
    y = _seq_targets(10)
    with pytest.raises(ValueError):
        aux_memory_loss(y, y, 10, "recall")
    with pytest.raises(ValueError):
        aux_memory_loss(y, y, 0, "recall")
    with pytest.raises(ValueError):
        aux_memory_loss(y, y, 3, "sideways")


def test_envelope_net_aux_heads_shapes_and_default_off():
    cfg = SLConfig()
    net = EnvelopeNet(cfg.model)
    assert net.aux_heads is None

    mcfg = SLModelConfig(aux_ks=[75], aux_mode="recall")
    net = EnvelopeNet(mcfg)
    assert set(net.aux_heads.keys()) == {"75"}
    L, B = 12, 3
    obs = torch.randn(L, B, mcfg.obs_dim)
    main, h = net.forward_with_aux(obs)
    assert main.shape == (L, B, mcfg.action_dim)
    assert h.shape == (L, B, mcfg.rnn_hidden_dim)
    pred = net.aux_heads["75"](h)
    assert pred.shape == (L, B, mcfg.action_dim)
    # default forward() unchanged: main output only
    assert net(obs).shape == (L, B, mcfg.action_dim)


def test_aux_head_state_dict_keys_do_not_touch_export_map():
    """The exported key map must stay the fixed 10 GRU/MLP tensors."""
    mcfg = SLModelConfig(aux_ks=[75])
    net = EnvelopeNet(mcfg)
    mapping = net.actor_state_dict_keys()
    assert len(mapping) == 10
    assert all(not k.startswith("aux") for k in mapping)
