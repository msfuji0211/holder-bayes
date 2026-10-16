"""Predictive diagnostic tests using generated observations and draws."""
import unittest
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import norm
from hb_experiments import predictive_diagnostics as core

class NumericalChecks(unittest.TestCase):
    def test_normal_predictive_intervals_cover_response_not_only_mean(self):
        y = norm.ppf((np.arange(1000)+0.5)/1000)
        X = np.ones((len(y), 1))
        z = np.zeros((8, 2))
        a, s = core.diagnose(z, X, y, dict(y_mean=0., y_std=1.), observation_batch=37)
        np.testing.assert_allclose(a['nlpd_mc'], -norm.logpdf(y), atol=1e-13)
        np.testing.assert_allclose(a['pit'], norm.cdf(y), atol=1e-14)
        np.testing.assert_allclose(a['q025'], norm.ppf(.025), atol=1e-13)
        self.assertEqual(s['coverage_80'], .8)
        self.assertEqual(s['coverage_95'], .95)
        self.assertEqual(s['coefficient_variance_mean'], 0.)
        self.assertEqual(s['residual_variance_mean'], 1.)

    def test_mixture_quantiles_match_independent_scalar_root_solver(self):
        mu = np.array([[-2., 3.], [2., -1.], [-2., 3.], [2., -1.]])
        sigma = np.array([.5, 2., 2., .5])
        q, error, _ = core.mixture_quantiles(mu, sigma, cdf_tolerance=1e-11)
        for i in range(2):
            for j, p in enumerate(core.PROBABILITIES):
                expected = brentq(lambda x: norm.cdf(x, loc=mu[:, i], scale=sigma).mean()-p, -30, 30, xtol=1e-13)
                self.assertAlmostEqual(q[i, j], expected, places=8)
        self.assertLessEqual(error, 1e-11)

    def test_total_variance_uses_mean_sigma_squared_not_geometric_sigma(self):
        z = np.array([[-2., np.log(.5)], [2., np.log(2.)], [-2., np.log(2.)], [2., np.log(.5)]])
        a, s = core.diagnose(z, np.ones((3, 1)), np.array([0., 1., 2.]), dict(y_mean=0., y_std=1.))
        self.assertAlmostEqual(s['sigma_hat'], 1.)
        self.assertAlmostEqual(s['residual_variance_mean'], 2.125)
        np.testing.assert_allclose(a['coefficient_variance'], 4., atol=1e-14)
        np.testing.assert_allclose(a['predictive_variance'], 6.125, atol=1e-14)

    def test_mixture_log_score_is_log_of_mean_density(self):
        z = np.array([[-2.,np.log(.5)],[2.,np.log(2.)]])
        y = np.array([0.,1.,-3.])
        a,_ = core.diagnose(z,np.ones((3,1)),y,dict(y_mean=0.,y_std=1.))
        expected = -np.log(.5*norm.pdf(y,loc=-2.,scale=.5)+.5*norm.pdf(y,loc=2.,scale=2.))
        np.testing.assert_allclose(a['nlpd_mc'],expected,atol=1e-13)
        mean_log = -.5*(norm.logpdf(y,loc=-2.,scale=.5)+norm.logpdf(y,loc=2.,scale=2.))
        self.assertGreater(float(np.max(np.abs(mean_log-expected))),1.)

    def test_overconfident_model_has_low_coverage(self):
        y = 2*norm.ppf((np.arange(1000)+.5)/1000)
        z = np.column_stack([np.zeros(8), np.full(8, np.log(.5))])
        a, s = core.diagnose(z, np.ones((1000, 1)), y, dict(y_mean=0., y_std=1.))
        self.assertLess(s['coverage_80'], .3)
        self.assertLess(s['coverage_95'], .4)
        self.assertEqual(s['coverage_95'], float((np.abs(y) <= .5*norm.ppf(.975)).mean()))
        np.testing.assert_array_equal(a['covered_80'] | (y < a['q10']) | (y > a['q90']), True)

    def test_log_density_remains_finite_in_extreme_tail(self):
        z = np.zeros((4, 2))
        a, _ = core.diagnose(z, np.ones((1, 1)), np.array([10000.]), dict(y_mean=0., y_std=1.))
        self.assertTrue(np.isfinite(a['nlpd_mc']).all())
        self.assertAlmostEqual(a['nlpd_mc'][0], -norm.logpdf(10000.), places=6)

    def test_batching_and_coordinate_scaling(self):
        rng = np.random.default_rng(18)
        z = np.column_stack([rng.normal(0, .1, (20, 2)), rng.normal(-.2, .05, 20)])
        X = np.column_stack([np.ones(17), rng.normal(size=17)])
        y = rng.normal(size=17)
        a, _ = core.diagnose(z, X, y, dict(y_mean=0., y_std=1.), observation_batch=3)
        b, _ = core.diagnose(z, X, y, dict(y_mean=0., y_std=1.), observation_batch=17)
        c, _ = core.diagnose(z, X, 5+2*y, dict(y_mean=5., y_std=2.), observation_batch=5)
        for key in a:
            np.testing.assert_allclose(a[key], b[key], atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(c['nlpd_mc'], a['nlpd_mc']+np.log(2.), atol=1e-11)
        np.testing.assert_allclose(c['q025'], 5+2*a['q025'], atol=1e-10)
        np.testing.assert_allclose(c['predictive_variance'], 4*a['predictive_variance'], atol=1e-11)
        np.testing.assert_array_equal(a['covered_95'], c['covered_95'])

    def test_positive_excess_mass_handles_negative_nlpd(self):
        result = core.pairwise_excess({'nlpd_mc':np.array([-3., 8., -4.])},
                                      {'nlpd_mc':np.array([-2., 1., -2.])})
        self.assertAlmostEqual(result['mean_nlpd_difference'], 4/3)
        self.assertEqual(result['top20_test_indices'][0], 1)
        self.assertEqual(result['top_positive_excess_groups'][0]['fraction_of_positive_excess_mass'], 1.)
