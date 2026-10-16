"""Numerical checks of target moments, tails, and deterministic fixed sampling."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import expit

from hb_experiments.kernels.repair_kernel import run_composed_chain, local_cholesky, longest_identical_run

class NormalProposal:
    def __init__(self, covariance):
        self.covariance = np.asarray(covariance)
        self.factor = np.linalg.cholesky(covariance)
        self.inverse = np.linalg.inv(covariance)
        self.dimension = len(covariance)

    def draw(self, rng, n):
        return rng.normal(size=(n, self.dimension)) @ self.factor.T, np.zeros(n)

    def logq(self, values):
        x = np.atleast_2d(values)
        return -.5*np.einsum("ni,ij,nj->n", x, self.inverse, x)


class TargetChecks(unittest.TestCase):
    def test_correlated_normal_target_moments(self):
        covariance = np.array([[1., .7], [.7, 1.8]])
        inverse = np.linalg.inv(covariance)
        target = lambda u: -.5*float(u @ inverse @ u)
        batch = lambda u: -.5*np.einsum("ni,ij,nj->n", u, inverse, u)
        proposal = NormalProposal(np.array([[.7, .1], [.1, .7]]))
        rng = np.random.default_rng(701)
        pilot = rng.multivariate_normal(np.zeros(2), covariance, size=(4, 500))
        factor = local_cholesky(pilot)
        parts = []
        for chain in range(4):
            out, lp, _ = run_composed_chain(target, batch, proposal, np.array([4., -4.]),
                seed=710+chain, warmup=500, draws=6000, local_chol=factor)
            np.testing.assert_allclose(lp, batch(out), rtol=0., atol=1e-14)
            parts.append(out)
        draws = np.concatenate(parts)
        np.testing.assert_allclose(draws.mean(axis=0), 0., atol=.055)
        np.testing.assert_allclose(np.cov(draws.T), covariance, rtol=.07, atol=.06)

    def test_logistic_tail_and_escape_from_rare_state(self):
        # rho has an exact logistic law: P(rho>3)=sigmoid(-3), epsilon=sigmoid(-rho) is uniform.
        target = lambda u: -.5*u[0]**2-np.logaddexp(0., -u[1])-np.logaddexp(0., u[1])
        batch = lambda u: -.5*u[:, 0]**2-np.logaddexp(0., -u[:, 1])-np.logaddexp(0., u[:, 1])
        proposal = NormalProposal(np.diag([1., .5]))
        parts = []
        for chain in range(4):
            out, _, info = run_composed_chain(target, batch, proposal, np.array([0., 8.]),
                seed=801+chain, warmup=1000, draws=8000, local_chol=np.diag([1., 1.2]), rho_step=1.)
            parts.append(out)
            self.assertLess(info["longest_identical_position_run"], 30)
        draws = np.concatenate(parts)
        self.assertLess(abs(draws[:, 0].mean()), .05)
        self.assertLess(abs(np.mean(draws[:, 1] > 3)-expit(-3)), .008)
        epsilon = expit(-draws[:, 1])
        self.assertLess(abs(epsilon.mean()-.5), .02)
        np.testing.assert_allclose(np.quantile(epsilon, [.025, .975]), [.025, .975], atol=.012)

    def test_rejected_states_and_deterministic_replay(self):
        target = lambda u: -.5*float(u @ u) if np.all(np.abs(u) < 1.) else -np.inf
        batch = lambda u: np.array([target(x) for x in u])
        proposal = NormalProposal(np.eye(2)*4.)
        kwargs = dict(seed=999, warmup=0, draws=200, local_chol=np.eye(2)*10., rho_step=10.)
        a, _, _ = run_composed_chain(target, batch, proposal, np.zeros(2), **kwargs)
        b, _, _ = run_composed_chain(target, batch, proposal, np.zeros(2), **kwargs)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all(np.abs(a) < 1.))
        self.assertGreater(longest_identical_run(a), 1)
        self.assertEqual(len(a), 200)


if __name__ == "__main__":
    unittest.main()
