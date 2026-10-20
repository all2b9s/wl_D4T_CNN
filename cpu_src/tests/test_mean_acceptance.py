"""The production criterion averages sources, never different output components."""
import unittest
import numpy as np
from cpu_src.validation import error_stats


class MeanAcceptanceTest(unittest.TestCase):
    def check(self, actual):
        return error_stats(np.array(actual), np.zeros_like(actual), 1e-6, 1e-5, 1e-4)

    def test_source_cancellation_allowed_but_component_cancellation_not_allowed(self):
        report = self.check([[.01, -.02], [-.01, .02]])
        self.assertTrue(report["passed"])
        self.assertFalse(report["pointwise_passed"])
        self.assertFalse(self.check([[.001, -.001], [.001, -.001]])["passed"])

    def test_boundary_is_strict_and_nonfinite_or_empty_cannot_pass(self):
        self.assertFalse(self.check([[1e-4, 0.]])["passed"])
        self.assertTrue(self.check([[.999e-4, 0.]])["passed"])
        self.assertFalse(self.check([[np.nan, 0.]])["passed"])
        self.assertFalse(self.check(np.empty((0, 2)))["passed"])

    def test_response_components_remain_separate(self):
        value = np.zeros((8, 2, 2))
        value[:, 1, 0] = 2e-4
        self.assertFalse(self.check(value)["passed"])
        with self.assertRaises(ValueError):
            error_stats(np.zeros((2, 2)), np.zeros(2), 1e-6, 1e-5, 1e-4)


if __name__ == "__main__":
    unittest.main()
