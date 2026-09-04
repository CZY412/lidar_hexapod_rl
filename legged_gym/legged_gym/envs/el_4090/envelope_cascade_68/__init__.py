"""envelope_cascade_68: 68-dim SE2 locomotion + EA2 point-cloud envelope perception.

Importing this package self-registers the ``el4090_cascade_68`` task with
``task_registry`` (same pattern as the tasks in ``legged_gym.envs``, but
kept inside the cascade package so no file outside it is modified).

Structural twin of ``envelope_cascade_83`` (legacy 83-dim contract); the
two packages are fully independent — neither imports the other.
"""

from legged_gym.envs.el_4090.envelope_cascade_68.el4090_cascade_config import (
    El4090Cascade68Cfg,
    El4090Cascade68CfgPPO,
)
from legged_gym.envs.el_4090.envelope_cascade_68.el4090_cascade_env import (
    EL_4090_CASCADE_68,
)
from legged_gym.utils.task_registry import task_registry

TASK_NAME = "el4090_cascade_68"


def register_cascade() -> None:
    """Idempotently register the cascade task (name-checked)."""
    if TASK_NAME not in task_registry.task_classes:
        task_registry.register(
            TASK_NAME,
            EL_4090_CASCADE_68,
            El4090Cascade68Cfg(),
            El4090Cascade68CfgPPO(),
        )


register_cascade()

__all__ = [
    "EL_4090_CASCADE_68",
    "El4090Cascade68Cfg",
    "El4090Cascade68CfgPPO",
    "TASK_NAME",
    "register_cascade",
]
