"""Import helpers for running envelope data tools without importing Isaac Gym."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def load_haa_range_module():
    """Load the pure-Torch range module by path, bypassing ``utils.__init__``."""
    name = "_el4090_haa_swing_range_standalone"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "network" / "haa_swing_range.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
