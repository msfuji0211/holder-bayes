"""Optional real inference smoke; no convergence claim for these short fits."""
import os
import tempfile
import unittest
import numpy as np
from hbayes import fit_density, fit_regression, SamplerConfig


@unittest.skipUnless(os.environ.get('HBAYES_TEST_MCMC')=='1', 'Set HBAYES_TEST_MCMC=1 in the inference environment')
class InferenceSmoke(unittest.TestCase):
    def test_generated_density_and_regression_fit_and_save(self):
        rng=np.random.default_rng(120)
        x=rng.normal(size=(40,2));x[-5:]+=5
        y=1+x@np.array([1.,-.5])+rng.normal(size=40);y[-5:]+=6
        settings=SamplerConfig(chains=2,warmup=40,draws=40,seed=120)
        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            for label,fn,args in [('density',fit_density,(x,)),('regression',fit_regression,(x,y))]:
                output=Path(d)/label
                fit=fn(*args,sampler=settings,output_dir=output)
                self.assertEqual(fit.samples.shape[:2],(2,40))
                self.assertTrue(np.isfinite(fit.samples).all())
                self.assertEqual(fit.outlier_probability.shape,(40,))
                self.assertTrue((output/'posterior_samples.npz').is_file())
