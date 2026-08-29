"""Pytest configuration for EA2 tests.

Isaac Gym requires ``isaacgym`` to be imported before PyTorch.  Some test
modules import torch at module scope, so this is the conventional place to
ensure the import order is correct.
"""

import isaacgym  # noqa: F401
