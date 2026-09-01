"""Frozen 83-dim SE2 (spider_envelop_2) layer vendored for cascade_83.

Vendored verbatim from ``legged_gym/legged_gym/envs/el_4090/spider_envelop_2``
at commit ``fd527a8`` (the merge base with ``feat/el_4090_2``).  The upstream
SE2 task has since moved to a 68-dim observation (range priors removed from
the policy observation); this frozen copy preserves the original 83-dim
contract that the cascade_83 checkpoints were trained against
(``policy_1.pt``: 83-dim TorchScript input; HAA range network condition
order).  Do not modify except by an explicit, reviewed re-vendor.

Allowed external dependencies of this package (deliberately NOT vendored,
unchanged on both branches): the v1 base layer ``spider_envelop/`` (env base
class, config base classes, symmetry) and
``legged_gym.utils.envelop.network.haa_swing_range``.
"""

from legged_gym.envs.el_4090.envelope_cascade_83.se2_frozen.config import (
    El4090Se2_83Cfg,
    El4090Se2_83CfgPPO,
)
from legged_gym.envs.el_4090.envelope_cascade_83.se2_frozen.env import EL_4090_SE2_83
from legged_gym.envs.el_4090.envelope_cascade_83.se2_frozen.envelope_condition import (
    EnvelopeConditionState,
)

__all__ = [
    "EL_4090_SE2_83",
    "El4090Se2_83Cfg",
    "El4090Se2_83CfgPPO",
    "EnvelopeConditionState",
]
