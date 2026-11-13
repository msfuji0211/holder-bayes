"""Checks for the reported finite-draw decisions."""
import unittest

import numpy as np

from hb_experiments.apogee_audit.decisions import (
    bracket_content, cleanup_range, count_quantile, joint_prefix_bracket, prefix,
)


class AuditDecisions(unittest.TestCase):
    def test_bracket_matches_exhaustive_prefix_search(self):
        rng = np.random.default_rng(123)
        for flags in (np.zeros((8, 6), bool), np.ones((8, 6), bool), rng.random((83, 6)) < .45):
            order = rng.permutation(flags.shape[1])
            for level in (.50, .95, 1.):
                n = len(order)
                options = []
                for l in range(n+1):
                    for u in range(l, n+1):
                        if bracket_content(flags, prefix(order, l), prefix(order, u)) >= level:
                            options.append((u-l, u, l))
                lo, hi, content = joint_prefix_bracket(flags, order, level)
                self.assertEqual((int((hi & ~lo).sum()), int(hi.sum()), int(lo.sum())), min(options))
                self.assertEqual(content, bracket_content(flags, lo, hi))

    def test_capacity_and_joint_membership_are_distinct(self):
        flags = np.eye(3, dtype=bool)
        self.assertEqual(count_quantile(flags.sum(axis=1), .95), 1)
        lo, hi, content = joint_prefix_bracket(flags, np.arange(3))
        self.assertEqual((int(lo.sum()), int(hi.sum()), content), (0, 3, 1.))

    def test_retained_mean_uses_selected_membership(self):
        flags = np.array([[True, False, False], [False, True, False],
                          [False, False, True], [True, True, True]])
        result = cleanup_range(flags, np.array([-1., 1., 3.]))
        self.assertEqual(result["dex_median"], 1.)
        self.assertEqual(result["empty_draws"], 1)
        self.assertAlmostEqual(result["dex_width"], 1.9)


if __name__ == "__main__":
    unittest.main()
