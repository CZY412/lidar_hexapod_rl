"""Unit tests for v3 directional rejection-complement range selection."""

import importlib.util
from pathlib import Path
import unittest


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "utils/envelop/train/train_api_haa_range.py"
)
_SPEC = importlib.util.spec_from_file_location("train_api_haa_range_test", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


class TestDirectionalRunningRange(unittest.TestCase):
    def test_keeps_full_largest_contiguous_range(self) -> None:
        # The rejected gap must remain absent: the returned range is the
        # complete longest component, not a fixed-width trim or joined range.
        self.assertEqual(
            _MODULE.select_largest_running_segment([(-1.5, -1.2), (-1.0, -0.35)]),
            (-1.0, -0.35),
        )

    def test_reports_none_below_point_two(self) -> None:
        with self.assertRaisesRegex(_MODULE.NoFeasibleHaaEnvelopeError, "none"):
            _MODULE.select_largest_running_segment([(-1.5, -1.31), (-1.0, -0.81)])
