"""envelope_cascade_83: SE2 locomotion + EA2 point-cloud envelope perception.

Importing this package self-registers the ``el4090_cascade`` task with
``task_registry`` (same pattern as the tasks in ``legged_gym.envs``, but
kept inside the cascade package so no file outside it is modified).
"""

from legged_gym.envs.el_4090.envelope_cascade_83.el4090_cascade_config import (
    El4090CascadeCfg,
    El4090CascadeCfgPPO,
)
from legged_gym.envs.el_4090.envelope_cascade_83.el4090_cascade_env import (
    EL_4090_CASCADE,
)
from legged_gym.utils.task_registry import task_registry

TASK_NAME = "el4090_cascade"


def register_cascade() -> None:
    """Idempotently register the cascade task (name-checked)."""
    if TASK_NAME not in task_registry.task_classes:
        task_registry.register(
            TASK_NAME,
            EL_4090_CASCADE,
            El4090CascadeCfg(),
            El4090CascadeCfgPPO(),
        )


register_cascade()

__all__ = [
    "EL_4090_CASCADE",
    "El4090CascadeCfg",
    "El4090CascadeCfgPPO",
    "TASK_NAME",
    "register_cascade",
]
