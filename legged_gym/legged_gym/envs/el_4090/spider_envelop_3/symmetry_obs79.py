"""79-D Envelop-3 left/right symmetry with direct HAA range observations."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


_LEFT_RIGHT_LEG_MAP = (3, 4, 5, 0, 1, 2)  # LB/LF/LM <-> RB/RF/RM
_DOF_SPLIT = 9
_OBS_DIM = 79
_HEIGHT_TARGET_INDEX = 66
_RANGE_START = 67


def _mirror_observation(obs: torch.Tensor) -> torch.Tensor:
    if obs.ndim != 2 or obs.shape[1] != _OBS_DIM:
        raise ValueError(f"Expected Envelop-3 observation [batch, {_OBS_DIM}], got {tuple(obs.shape)}")
    mirrored = obs.clone()
    mirrored[:, 1] = -obs[:, 1]
    mirrored[:, 3] = -obs[:, 3]
    mirrored[:, 5] = -obs[:, 5]
    mirrored[:, 7] = -obs[:, 7]
    mirrored[:, 10] = -obs[:, 10]
    mirrored[:, 11] = -obs[:, 11]
    for start in (12, 30, 48):
        mirrored[:, start:start + _DOF_SPLIT] = obs[:, start + _DOF_SPLIT:start + 18]
        mirrored[:, start + _DOF_SPLIT:start + 18] = obs[:, start:start + _DOF_SPLIT]
    # The single target height is left/right invariant.
    mirrored[:, _HEIGHT_TARGET_INDEX] = obs[:, _HEIGHT_TARGET_INDEX]
    ranges = obs[:, _RANGE_START:_RANGE_START + 12].reshape(-1, 6, 2)
    leg_map = torch.tensor(_LEFT_RIGHT_LEG_MAP, dtype=torch.long, device=obs.device)
    mirrored[:, _RANGE_START:_RANGE_START + 12] = ranges.index_select(1, leg_map).reshape(-1, 12)
    return mirrored


def _mirror_actions(actions: torch.Tensor) -> torch.Tensor:
    if actions.ndim != 2 or actions.shape[1] != 18:
        raise ValueError(f"Expected actions [batch, 18], got {tuple(actions.shape)}")
    mirrored = actions.clone()
    mirrored[:, :_DOF_SPLIT] = actions[:, _DOF_SPLIT:]
    mirrored[:, _DOF_SPLIT:] = actions[:, :_DOF_SPLIT]
    return mirrored


@torch.no_grad()
def get_elair_lidar_xsym_obs_act(
    obs: Optional[torch.Tensor] = None,
    actions: Optional[torch.Tensor] = None,
    env=None,
    obs_type: str = "policy",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return original plus left/right-mirrored 79-D observations and actions."""
    del env, obs_type
    obs_augmented = torch.cat((obs, _mirror_observation(obs)), dim=0) if obs is not None else None
    actions_augmented = torch.cat((actions, _mirror_actions(actions)), dim=0) if actions is not None else None
    return obs_augmented, actions_augmented


__all__ = ["get_elair_lidar_xsym_obs_act"]
