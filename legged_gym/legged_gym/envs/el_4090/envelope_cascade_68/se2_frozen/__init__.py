"""Frozen 68-dim SE2 (spider_envelop_2) layer vendored for cascade_68.

Vendored verbatim from ``legged_gym/legged_gym/envs/el_4090/spider_envelop_2``
at commit ``84c08ca`` (HEAD of ``feat/el_4090_2``, tag ``c83m/final-merge``;
the SE2 sources last changed at ``6cb4e49``, the 68-dim observation refactor).
The live SE2 task keeps evolving upstream; this frozen copy preserves the
68-dim contract that the cascade_68 checkpoints (``policy_1.pt``: 68-dim
TorchScript input; HAA range network condition order) are pinned against.
Do not modify except by an explicit, reviewed re-vendor.

Allowed external dependencies of this package (deliberately NOT vendored,
unchanged on both branches): the v1 base layer ``spider_envelop/`` (env base
class, config base classes, symmetry) and
``legged_gym.utils.envelop.network.haa_swing_range``.
"""

from legged_gym.envs.el_4090.envelope_cascade_68.se2_frozen.config import (
    El4090Se2_68Cfg,
    El4090Se2_68CfgPPO,
)
from legged_gym.envs.el_4090.envelope_cascade_68.se2_frozen.env import EL_4090_SE2_68
from legged_gym.envs.el_4090.envelope_cascade_68.se2_frozen.envelope_condition import (
    EnvelopeConditionState,
)

__all__ = [
    "EL_4090_SE2_68",
    "El4090Se2_68Cfg",
    "El4090Se2_68CfgPPO",
    "EnvelopeConditionState",
]
