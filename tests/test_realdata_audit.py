"""Checks for the scientific decisions and the common diagnostic comparison."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from hb_experiments.realdata_audit import decision, design, kf, report


class AuditChecks(unittest.TestCase):
    def test_cost_quantile_is_capacity_optimum(self):
        k = np.array([0, 1, 1, 2, 2, 3, 5, 5, 8, 10])
        b = decision.quantile(k, .90)
        risk = [0.1*j+np.maximum(k-j, 0).mean() for j in range(12)]
        self.assertEqual(b, 8)
        self.assertAlmostEqual(risk[b], min(risk))
        ranking = np.array([2, 0, 1])
        q = decision.fixed_frequency(3, np.array([0, 1, 2, 2]), ranking)
        np.testing.assert_allclose(q, [.5, 0, .75])
        np.testing.assert_array_equal(decision.actions(np.array([.99, .4, .03]), 5, 1, .1),
                                      [decision.REMOVE, decision.REVIEW, decision.KEEP])

    def test_paired_injection_changes_only_marked_responses(self):
        split = dict(y0=np.array([1, 2, 3], dtype=np.float32), reference_sd=np.array(.5),
                     injection_mask=np.array([True, False, True]), injection_sign=np.array([-1, 1, 1]),
                     test_indices=np.array([7, 4, 1]))
        clean, weak, strong = [design.with_injection(split, shift) for shift in [0., 2.5, 6.]]
        self.assertEqual(clean["Z"].sum(), 0)
        np.testing.assert_array_equal(weak["Z"], strong["Z"])
        np.testing.assert_allclose(weak["y"], [-.25, 2, 4.25])
        np.testing.assert_allclose(strong["y"], [-2, 2, 6])
        np.testing.assert_array_equal(weak["test_indices"], strong["test_indices"])

    def test_profile_gradient(self):
        rng = np.random.default_rng(123)
        X = np.column_stack((np.ones(36), rng.normal(size=36)))
        y = rng.normal(size=36)
        w = rng.integers(1, 5, size=36)
        w = w/w.sum()
        for theta in [np.array([.3, -.5, -.4]), np.array([0., 0., .7])]:
            val, grad, alpha = kf.profile(X, y, w, .5, theta)
            eps = np.eye(3)*1e-6
            fd = [(kf.profile(X, y, w, .5, theta+e)[0]-kf.profile(X, y, w, .5, theta-e)[0])/2e-6 for e in eps]
            np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-8)
            self.assertLessEqual(alpha, 1)
            self.assertLess(val, 0)

    def test_common_diagnostics_control_the_report(self):
        with TemporaryDirectory() as name:
            out = Path(name)
            c = design.config()
            c.update(datasets=["kin8nm"], replicates=2)
            manifest = design.open_run(out, c)
            for job in [j for j in manifest["jobs"] if j["shift"]==2.5]:
                valid = job["replicate"]==0
                d = out/"jobs"/job["id"]
                design.write_json(d/"result.json", dict(
                    fingerprint=manifest["fingerprint"], job=job, status=dict(paired_valid=valid),
                    capacity=[dict(method="DPD", rule="cost_quantile", loss=.02 if valid else 99.),
                              dict(method="DPD", rule="central", loss=.05)],
                    triage=[], removal=[], counts=[]))
            self.assertTrue(report.collect(out).is_file())
            p = pd.read_csv(out/"summary"/"paired_summary.csv")
            self.assertEqual(len(p), 1)
            self.assertEqual(p.iloc[0]["pair_n"], 1)
            self.assertAlmostEqual(p.iloc[0]["delta_loss_mean"], -.03)
            completion = pd.read_csv(out/"summary"/"completion.csv")
            self.assertEqual(len(completion), 6)
            self.assertEqual(completion["completed"].sum(), 2)
            self.assertEqual(completion["paired_valid"].sum(), 1)


if __name__ == "__main__":
    unittest.main()
