"""Numerical and decision checks for the additive experiments; no result fixtures."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize._numdiff import approx_derivative
from hb_experiments.additional import common as C
from hb_experiments.additional import frequentist as F
from hb_experiments.additional import auditing as A


class AdditionalChecks(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(441)
        self.X=np.c_[np.ones(50),self.rng.normal(size=50)]
        self.y=self.X@np.array([.5,-1.2])+self.rng.normal(size=50)
        self.y[-7:]+=5
        self.cfg=C.load_config(C.ROOT/'configs/additional_smoke.json')

    def test_profile_envelope_gradient_weighting_and_native_risk(self):
        import jax
        import jax.numpy as jnp
        from hbayes.models import LinearRegressionParams,MVNDiagParams
        from hbayes.posteriors import S_gamma_regression,S_holder_mvn_diag
        with jax.experimental.enable_x64():
            for model,X,y,t in [('regression',self.X,self.y,np.array([.4,-1.,.2])),
                                 ('density',self.X[:,1:],None,np.array([.1,.15]))]:
                for gamma in [.5,1.]:
                    w=self.rng.integers(0,4,size=len(X));v,g,alpha=F.profiled(t,X,y,model,gamma,w)
                    numerical=approx_derivative(lambda z:F.profiled(z,X,y,model,gamma,w)[0],t).ravel()
                    assert_allclose(g,numerical,rtol=2e-5,atol=2e-7)
                    ids=np.repeat(np.arange(len(X)),w);d=X.shape[1]
                    if model=='regression':
                        params=LinearRegressionParams(jnp.asarray(t[:d]),jnp.asarray(t[d]))
                        expected=S_gamma_regression(params,alpha,jnp.asarray(X[ids]),jnp.asarray(y[ids]),gamma)
                    else:
                        params=MVNDiagParams(jnp.asarray(t[:d]),jnp.asarray(t[d:]))
                        expected=S_holder_mvn_diag(params,alpha,jnp.asarray(X[ids]),gamma)
                    assert_allclose(v,expected,rtol=1e-12,atol=1e-12)
                    unweighted=F.profiled(t,X[ids],None if y is None else y[ids],model,gamma)
                    assert_allclose(g,unweighted[1],rtol=1e-12,atol=1e-12)

    def test_profile_mass_respects_boundary(self):
        X=np.zeros((12,1));v,g,alpha=F.profiled(np.array([0.,0.]),X,None,'density',.5)
        self.assertEqual(alpha,1.)
        assert_allclose(g,approx_derivative(lambda t:F.profiled(t,X,None,'density',.5)[0],np.zeros(2)).ravel(),atol=1e-7)

    def test_optimizer_records_stationarity_and_starts(self):
        fit=F.fit(self.X,self.y,'regression',.5,self.cfg['optimizer'],seed=15)
        self.assertTrue(fit.valid);self.assertLess(fit.projected_gradient,1e-4)
        self.assertEqual(len(fit.starts),4);self.assertTrue(0<=fit.epsilon<=1)
        # Estimator must improve on its initial values, not merely echo the start.
        self.assertLessEqual(fit.objective,min(F.profiled(t,self.X,self.y,'regression',.5)[0] for t in fit.starts)+1e-10)

    def test_bootstrap_scores_original_observations_and_resumes_atomically(self):
        from hb_experiments.additional.bootstrap import run
        fit=F.fit(self.X,self.y,'regression',.5,self.cfg['optimizer'])
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            r=run(self.X,self.y,'regression',.5,self.cfg['optimizer'],fit.theta,3,9,d)
            assert_array_equal(r['weights'].sum(axis=1),np.full(3,len(self.X)))
            self.assertEqual(r['selected'].shape,(3,len(self.X)))
            assert_array_equal(r['selected'].sum(axis=1),np.rint(len(self.X)*r['epsilon']))
            # Human-readable JSON is not required to restore a committed checkpoint.
            (Path(d)/'diagnostics.json').unlink()
            restored=run(self.X,self.y,'regression',.5,self.cfg['optimizer'],fit.theta,3,9,d)
            assert_allclose(restored['theta'],r['theta'])
            self.assertEqual(len(restored['diagnostics']),3)
            with self.assertRaises(ValueError):run(self.X,self.y+.01,'regression',.5,self.cfg['optimizer'],fit.theta,3,9,d)

    def test_fod_identity_and_uncertainty_ablations(self):
        freq=F.fit(self.X,self.y,'regression',.5,self.cfg['optimizer'])
        position=np.tile(np.r_[freq.theta,1.5],(2,30,1))
        position[:,:,1]+=self.rng.normal(0,.4,size=(2,30));position[:,:,-1]+=self.rng.normal(0,.5,size=(2,30))
        audit=A.score_fit(position,freq,None,self.X,self.y,'regression',60)
        self.assertAlmostEqual(audit['scores']['FoD'].sum(),audit['posterior_K'].mean())
        self.assertTrue(np.any((audit['scores']['FoD']>0)&(audit['scores']['FoD']<1)))
        self.assertEqual(set(audit['scores']),set(A.METHODS)-{'Bootstrap'})
        self.assertFalse(np.allclose(audit['scores']['FoD'],audit['scores']['Epsilon only']))

    def test_policy_costs_and_imperfect_review_by_hand(self):
        q=np.array([0.,.3,.95]);action=A.policy_actions(q,5.,1.,.1)
        assert_array_equal(action,[0,2,2])
        m=A.policy_metrics(np.array([0,1,2,2]),np.array([1,0,1,0]),5,1,.1,.2)
        self.assertEqual(m['false_removed'],1);self.assertEqual(m['missed'],1);self.assertEqual(m['reviewed'],2)
        self.assertAlmostEqual(m['loss'],(1+5+.2+.2*(1+5))/4)

    def test_clean_population_error_and_rank_deficient_refit(self):
        data=dict(beta_true=np.array([0.,1.]),scaler_y_std=2.,scaler_y_mean=3.,scaler_x_mean=np.array([4.]),scaler_x_std=np.array([2.]),population_cov=np.array([[2.6]]))
        # Standardised beta [0.5,1] transforms exactly to original [0,1].
        self.assertAlmostEqual(A.regression_quality(np.array([.5,1.]),data)['clean_population_RMSE'],1.)
        self.assertIsNone(A.weighted_refit(np.ones((8,2)),np.ones(8),np.ones(8)))

    def test_cost_data_are_paired_across_gamma_and_shift(self):
        from hb_experiments.additional.datasets import prepare
        j=next(x for x in C.jobs(self.cfg,'costs') if x['epsilon']>0)
        a=prepare(j,self.cfg);b=prepare(dict(j,gamma=1.),self.cfg);d=prepare(dict(j,shift=6.),self.cfg)
        for k in ['X','y','Z']:assert_array_equal(a[k],b[k])
        for k in ['x_raw','Z','beta_true']:assert_array_equal(a[k],d[k])
        self.assertEqual(a['seed'],self.cfg['costs']['seed']+10000+5000)

    def test_original_generators_and_coordinates_are_preserved(self):
        from hb_experiments.additional.datasets import prepare
        from experiments.exp1_regression.utils_exp1_regression import simulate_regression_dataset,standardize_regression_data
        from experiments.exp1_simulation.utils_exp1 import simulate_dataset
        for job in C.jobs(self.cfg,'direct'):
            data=prepare(job,self.cfg)
            if job['family']=='density':
                X,Z=simulate_dataset(60,.2,'A',int(data['seed']))
                assert_array_equal(X,data['X']);assert_array_equal(Z,data['Z'])
            else:
                raw=simulate_regression_dataset(60,100,.2,6.,int(data['seed']),x_dim=2,contamination_mode='response_homogeneous')
                X,y,_,_=standardize_regression_data(raw['x_train'],raw['y_train'],raw['x_test'])
                assert_array_equal(X,data['X']);assert_array_equal(y,data['y'])
            self.assertEqual(data['X'].dtype,np.float32)

    def test_manifest_rejects_changed_inputs_config_and_source(self):
        from hb_experiments.additional.runner import initialize,verify_complete
        with tempfile.TemporaryDirectory() as d:
            value=initialize(d,self.cfg);self.assertEqual(value,initialize(d,self.cfg))
            cfg=C.merge(self.cfg,{'seed':43})
            with self.assertRaises(ValueError):initialize(d,cfg)
            C.dump(Path(d)/'result.json',dict(fingerprint=value,artifacts={'missing.csv':'abc'}))
            with self.assertRaises(ValueError):verify_complete(Path(d),value)

    def test_macro_summary_requires_complete_replication_blocks(self):
        import pandas as pd
        from hb_experiments.additional.summary import paired_costs
        rows=[]
        cfg=C.merge(self.cfg,{'costs':{'replicates':2}})
        for rep in range(2):
            for shift in [2.5,6.]:
                for method,loss in [('FoD',.1),('Frequentist',.2)]:
                    rows.append(dict(id=f'{rep}_{shift}',rule='risk_proxy',false_remove_cost=5.,miss_cost=1.,review_cost=.1,review_error_rate=0.,method=method,
                                     loss=loss,false_removed=0,missed=1,reviewed=3,gamma=.5,n=50,p=1,shift=shift,epsilon=.15,rep=rep,paired_valid=not(rep==1 and shift==6.)))
        diffs,blocks,macro=paired_costs(pd.DataFrame(rows),cfg)
        self.assertEqual(len(diffs),16);self.assertEqual(len(blocks),4)
        loss=next(r for r in macro if r['metric']=='loss');self.assertAlmostEqual(loss['difference'],-.1)
        self.assertEqual(loss['n_complete_replicates'],1);self.assertTrue(np.isnan(loss['ci_low']))

    def test_grids_and_duplicate_validation(self):
        c=C.load_config();self.assertEqual(len(C.jobs(c,'direct')),360);self.assertEqual(len(C.jobs(c,'costs')),400)
        for study in ['direct','costs']:
            ids=[j['id'] for j in C.jobs(c,study)];self.assertEqual(len(ids),len(set(ids)))
        c['costs']['shifts']=[2.5,2.5]
        with self.assertRaises(ValueError):C.validate(c)

    def test_benchmark_loader_uses_full_local_raw_data_and_fixed_clean_scaler(self):
        from hb_experiments.additional.datasets import prepare
        from experiments.exp1_regression.utils_exp1_regression import sample_train_test_with_split_cfg,make_realdata_split_seed,kanamori_standardize_regression_data
        # Generated raw data exercise the local-file route; no claim to be kin8nm results.
        x=self.rng.normal(size=(8192,8));y=x[:,0]+self.rng.normal(size=8192)
        cfg=C.merge(self.cfg,{'direct':{'benchmark':True,'datasets':['kin8nm'],'realdata_replicates':1}})
        jobs=[j for j in C.jobs(cfg,'direct') if j['family']=='benchmark']
        with tempfile.TemporaryDirectory() as d:
            np.savetxt(Path(d)/'kin8nm.csv',np.c_[x,y],delimiter=',',header=','.join([f'x{i}' for i in range(8)]+['target']),comments='')
            data=prepare(jobs[-1],cfg,local_data_dir=d,dataset_cache=Path(d)/'cache')
            self.assertEqual(data['X'].shape,(6553,9));self.assertEqual(data['X_test'].shape,(1639,9))
            rx,ry,tx,ty=sample_train_test_with_split_cfg(x,y,make_realdata_split_seed(cfg['seed'],0),split_sizes='ratio',train_ratio=.8,test_ratio=.2)
            X,_,Xt,scaler=kanamori_standardize_regression_data(rx,ry,tx)
            assert_allclose(data['X'],X,atol=1e-6);assert_allclose(data['X_test'],Xt,atol=1e-6)
            self.assertAlmostEqual(float(data['scaler_y_std']),scaler['y_std'])
            self.assertTrue(data['Z'].any());self.assertTrue((Path(d)/'cache/kin8nm.npz').exists())

    def test_full_execution_collection_and_valid_plot_branches_with_generated_posterior_fixture(self):
        from hb_experiments.additional.runner import initialize,run_job
        from hb_experiments.additional.summary import collect,read_table
        from hb_experiments.additional.plotting import plot
        cfg=C.merge(self.cfg,{'direct':{'cleaning_refit':'nll'}})
        refit_calls=[]
        def generated_posterior(X,y,model,gamma,seed,cfg,output,**kwargs):
            d=X.shape[1];r=np.random.default_rng(seed)
            if model=='density':center=np.r_[np.median(X,axis=0),np.zeros(d)]
            else:center=np.r_[np.linalg.lstsq(X,y,rcond=None)[0],0.]
            if not kwargs.get('likelihood_only'):center=np.r_[center,1.7]
            else:refit_calls.append((X.copy(),y.copy()))
            position=center+r.normal(0,.05,size=(2,40,len(center)))
            C.save_npz(Path(output)/'attempt_00/samples.npz',position=position)
            return position,dict(additional_diagnostics_pass=True,sampler='generated_test_fixture',rhat_max=1.,ess_bulk_min=1000.,ess_tail_min=1000.)
        with tempfile.TemporaryDirectory() as d,patch('hb_experiments.additional.posterior.fit',side_effect=generated_posterior),contextlib.redirect_stdout(io.StringIO()):
            fingerprint=initialize(d,cfg)
            for study in ['direct','costs']:
                for job in C.jobs(cfg,study):self.assertEqual(run_job(job,cfg,d,fingerprint),'complete')
                report=collect(d,cfg,study)
                self.assertEqual(report['complete'],report['expected']);self.assertEqual(report['paired_valid'],report['expected'])
                plot(d,cfg,study)
                self.assertGreater(len(list((Path(d)/study/'summary/figures').glob('*.png'))),1)
            macro=read_table(Path(d)/'costs/summary/macro_paired.csv');self.assertFalse(macro.empty)
            self.assertTrue(refit_calls)
            job=C.jobs(cfg,'direct')[0]
            self.assertEqual(run_job(job,cfg,d,fingerprint),'reuse')


if __name__=='__main__':unittest.main()
