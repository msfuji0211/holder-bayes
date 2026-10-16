"""Public-interface numerical fixtures; no new scientific MCMC results."""
import ast,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.special import expit
from hbayes import SamplerConfig,audit_probabilities,fit_density,fit_regression
from hbayes.api import _regression_inputs,_sampler_context

ROOT=Path(__file__).resolve().parents[1]

def original_functions(path,names):
    tree=ast.parse((ROOT/path).read_text())
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    namespace=dict(np=np,jnp=np,expit=expit)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),namespace)
    return namespace

class PublicAPIChecks(unittest.TestCase):
    def test_regression_standardisation_matches_reviewed_function(self):
        original=original_functions(Path('experiments/exp1_regression/utils_exp1_regression.py'),{'standardize_regression_data'})
        rng=np.random.default_rng(31);X=3+2*rng.normal(size=(70,3));y=7+rng.normal(size=70)
        design,response,scaler=_regression_inputs(X,y)
        expected,ey,_,es=original['standardize_regression_data'](X,y,X)
        np.testing.assert_array_equal(design,expected);np.testing.assert_array_equal(response,ey)
        for name in scaler:np.testing.assert_array_equal(scaler[name],es[name])

    def test_density_audit_matches_reviewed_bottom_k_algorithm(self):
        original=original_functions(Path('experiments/exp1_simulation/utils_exp1.py'),{'compute_dpd_outlier_prob'})
        rng=np.random.default_rng(19);x=rng.normal(size=(90,2));z=rng.normal(size=(4,140,5));z[...,2:4]*=.3
        z.reshape(-1,5)[:2,-1]=[-1000.,1000.]
        expected=original['compute_dpd_outlier_prob'](z.reshape(-1,5),x,S_eval_max=400,seed=37)['p_out']
        actual=audit_probabilities(z,[x],'density',seed=37)
        np.testing.assert_array_equal(actual,expected)

    def test_regression_audit_matches_reviewed_bottom_k_algorithm(self):
        wanted={'compute_dpd_outlier_prob','compute_log_likelihoods_gaussian','summarize_k_budgets'}
        original=original_functions(Path('experiments/exp1_regression/utils_exp1_regression.py'),wanted)
        rng=np.random.default_rng(21);X=np.c_[np.ones(100),rng.normal(size=(100,2))];y=rng.normal(size=100)
        z=rng.normal(size=(4,140,5));z[...,3]*=.3
        expected=original['compute_dpd_outlier_prob'](z.reshape(-1,5),X,y,s_eval_max=400,seed=38)['p_out']
        actual=audit_probabilities(z,[X,y],'regression',seed=38)
        np.testing.assert_array_equal(actual,expected)

    def backend(self,model,inputs,**settings):
        cfg=settings['sampler'];d=inputs[0].shape[1]
        z=np.zeros((cfg.chains,cfg.draws,2*d+1 if model=='density' else d+2))
        z[...,0]=.25
        if model=='regression':z[...,1:d]=np.arange(1,d)*.3
        z[...,-1]=np.log(4.)
        self.last=(model,inputs,settings,z.copy())
        return dict(samples_chains=z,samples=z.reshape(-1,z.shape[-1]),diagnostics_pass=False,sampler='fixture_no_mcmc')

    def test_fit_preserves_all_draws_and_original_unit_intercept(self):
        rng=np.random.default_rng(11);X=rng.normal(size=(60,2))*[2.,3.]+[4.,-5.];y=8+2*rng.normal(size=60)
        cfg=SamplerConfig(chains=2,warmup=20,draws=40)
        with patch('hbayes.api._sample',side_effect=self.backend):fit=fit_regression(X,y,sampler=cfg)
        model,inputs,settings,z=self.last
        self.assertEqual(model,'regression');self.assertEqual(settings['phi'],'dpd')
        self.assertFalse(fit.diagnostics['diagnostics_pass']);self.assertEqual(fit.samples.shape,(2,40,5))
        np.testing.assert_array_equal(fit.samples,z)
        # Raw-unit and standardised predictions must agree for every draw.
        beta=fit.coefficient_samples
        from_raw=beta[...,0,None]+beta[...,1:]@X.T
        from_likelihood=fit.scaler['y_mean']+fit.scaler['y_std']*(z[...,:3]@inputs[0].T)
        np.testing.assert_allclose(from_raw,from_likelihood,rtol=1e-7,atol=1e-7)
        np.testing.assert_allclose(fit.predict(X),from_raw.mean(axis=(0,1)),atol=1e-13)
        with tempfile.TemporaryDirectory() as d:
            fit.save(d)
            with np.load(Path(d)/'posterior_samples.npz',allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['samples_physical'],z)
                np.testing.assert_array_equal(saved['coefficients'],beta)
            self.assertTrue((Path(d)/'summary.json').is_file());self.assertTrue((Path(d)/'audit_scores.csv').is_file())

    def test_phi_defaults_and_covariance_temperature_reach_backend(self):
        rng=np.random.default_rng(12);x=rng.normal(size=(80,2))*[2.,3.]
        cfg=SamplerConfig(chains=2,warmup=20,draws=40)
        with patch('hbayes.api._sample',side_effect=self.backend):fit=fit_density(x,phi='rational_saturated',gamma=.5,sampler=cfg)
        _,_,settings,_=self.last
        self.assertEqual(settings['kappa'],.25)
        self.assertAlmostEqual(settings['eta'],np.linalg.det(np.cov(x,rowvar=False))**.25,places=12)
        self.assertEqual(fit.location_samples.shape,(2,40,2));self.assertEqual(fit.scale_samples.shape,(2,40,2))

    def test_context_restores_environment_and_legacy_job_on_error(self):
        from hbayes import paper_sampling
        original=dict(paper_sampling.CONTEXT)
        try:
            paper_sampling.CONTEXT.update(legacy_folder='original-job',force_methods=['DPD'])
            with patch.dict(os.environ,{'HBAYES_CACHE_ROOT':'original-cache','HBAYES_MH30_ROOT':'original-donor'}):
                with self.assertRaisesRegex(RuntimeError,'fixture interruption'):
                    with _sampler_context('temporary-cache',SamplerConfig(chains=2,warmup=20,draws=40)):
                        self.assertEqual(paper_sampling.CONTEXT,{})
                        self.assertNotIn('HBAYES_MH30_ROOT',os.environ)
                        raise RuntimeError('fixture interruption')
                self.assertEqual(os.environ['HBAYES_CACHE_ROOT'],'original-cache')
                self.assertEqual(os.environ['HBAYES_MH30_ROOT'],'original-donor')
            self.assertEqual(paper_sampling.CONTEXT['legacy_folder'],'original-job')
        finally:paper_sampling.CONTEXT.clear();paper_sampling.CONTEXT.update(original)

    def test_invalid_inputs_fail_before_sampling(self):
        with patch('hbayes.api._sample') as backend:
            for call in [lambda:fit_density([[1.,2.],[np.nan,3.]]),lambda:fit_regression(np.ones((3,2)),np.ones(2)),
                         lambda:fit_density(np.eye(3),phi='unknown'),lambda:SamplerConfig(chains=1)]:
                with self.assertRaises(ValueError):call()
            backend.assert_not_called()
