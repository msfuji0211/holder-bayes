"""Exercise the real dispatcher and post-processing without archived result files."""
import contextlib
import io
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from hbayes.cli import main
from hb_experiments.common import ROOT, read, write
from hb_experiments.collection import collect
from hb_experiments.inference import run
from hb_experiments.tables import render
from hb_experiments.native_run import validate_dataset
from fixtures import collect_fixture


class SourceOnlyChecks(unittest.TestCase):
    def test_check_plan_doctor_do_not_need_previous_results(self):
        self.assertFalse((ROOT/'reference').exists());self.assertFalse((ROOT/'provenance').exists())
        for argv in [['check'],['paper','plan'],['doctor']]:
            with contextlib.redirect_stdout(io.StringIO()) as output:status=main(argv)
            self.assertEqual(status,0);self.assertIsInstance(json.loads(output.getvalue()),dict)
        self.assertFalse((ROOT/'results').exists())

    def test_missing_runtime_has_installation_error_without_reading_results(self):
        with tempfile.TemporaryDirectory() as d, patch('hb_experiments.inference.doctor',return_value={'missing':['jax']}):
            with self.assertRaisesRegex(RuntimeError,'pip install'):run(Path(d)/'new','synthetic')

    def test_missing_raw_data_is_actionable(self):
        with tempfile.TemporaryDirectory() as d, patch('hb_experiments.inference.doctor',return_value={'missing':[]}):
            with self.assertRaisesRegex(FileNotFoundError,'download-data'):run(Path(d)/'new','benchmark')

    def test_full_dispatch_collect_plot_tables_pack_with_generated_data(self):
        # Fits are replaced by generated fixtures, but the execution branch,
        # complete-grid collector, eleven plots, table exporter and ZIP are real.
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'new';called=[];original_run=subprocess.run
            def backend(command, **kw):
                if len(command)<4 or command[1:3]!=['-m','hb_experiments.native_run']:
                    return original_run(command,**kw)
                phase=command[3];called.append(phase)
                self.assertNotIn('HB_REPRO_ASSETS',kw['env'])
                self.assertEqual(kw['cwd'],ROOT)
                if phase=='collect':collect_fixture(output/'runs')
                return SimpleNamespace(returncode=0)
            with patch('hb_experiments.inference.doctor',return_value={'missing':[]}), \
                 patch('hb_experiments.inference.subprocess.run',side_effect=backend), \
                 patch.dict(os.environ,{'HB_REPRO_ASSETS':'nonexistent_old_results'}):
                report=run(output,'all',allow_download=True)
            self.assertEqual(len(called),9)
            self.assertEqual(report['current_layout_figures']['figures'],11)
            self.assertTrue(report['current_layout_figures']['regression_intercept_included'])
            self.assertEqual(len(list((output/'summaries').rglob('*.csv'))),20)
            self.assertTrue((output/'figures/main/posterior.png').is_file())
            self.assertEqual(len(pd.read_csv(output/'figures/main/posterior_summary.csv')),12)
            self.assertGreater(report['tables']['numeric_cells'],0)
            table=(output/'tables/benchmark_aggregate.tex').read_text()
            self.assertIn(r'\mathbf',table)
            csv=output/'summaries/density/gamma_0.500/runs.csv'
            self.assertEqual(len(pd.read_csv(csv)),5*30*6)
            agg=pd.read_csv(output/'summaries/regression/gamma_1.000/aggregate.csv')
            row=agg[(agg.method=='DPD') & np.isclose(agg.eps_true,.2)].iloc[0]
            expected=.9+.2+np.arange(30)/10000
            self.assertAlmostEqual(row.rmse_test_mean,expected.mean())
            self.assertAlmostEqual(row.rmse_test_std,expected.std(ddof=1))
            from hb_experiments.artifacts import pack
            archive=Path(d)/'paper-results.zip';pack(output,archive)
            self.assertTrue(archive.is_file())
            # Collection rejects missing and failed fits; a data-dependent
            # problem cannot be hidden by removing one replicate from a CSV.
            a=SimpleNamespace(output_dir=output/'runs',gammas=[.1,.5,1.],datasets=['cal_housing','pumadyn32fh','kin8nm'],replicates=30,real_replicates=5)
            metric=a.output_dir/'exp1_regression_eps_sweep_mu6_gamma0p1/eps0.00/rep000/metrics.json'
            value=read(metric);value['dpd']['rhat_max']=1.2;write(metric,value)
            with self.assertRaisesRegex(ValueError,'diagnostics'):collect(a)
            metric.unlink()
            with self.assertRaises(FileNotFoundError):collect(a)

    def test_best_displayed_ties_are_bold_and_oracle_is_excluded(self):
        frame=pd.DataFrame(dict(dataset=['cal_housing']*3,gamma=[.5]*3,eps_true=[.2]*3,
            cleaning_method=['NoCleaning','ProposedDPD','OracleCleaning'],
            rmse_test_mean=[1.0001,1.0002,.5],rmse_test_std=[.1]*3,
            nlpd_test_mc_mean=[3.,2.,.1],nlpd_test_mc_std=[.2]*3))
        text,cells=render(frame)
        rmse=[c for c in cells if c['metric']=='rmse_test']
        self.assertEqual([c['bold'] for c in rmse],[True,True,False])
        nlpd=[c for c in cells if c['metric']=='nlpd_test_mc']
        self.assertEqual([c['bold'] for c in nlpd],[False,True,False])
        self.assertNotIn('@@',text)

    def test_real_dataset_dimensions_reject_substitutes(self):
        with self.assertRaises(ValueError):validate_dataset('pumadyn32fh',np.zeros((8192,8)),np.zeros(8192))
        x=np.zeros((8192,32));y=np.zeros(8192)
        self.assertIs(validate_dataset('pumadyn32fh',x,y)[0],x)

    def test_new_predictive_analysis_retains_all_splits_and_uses_generated_caches(self):
        from fixtures import predictive_campaign
        from hb_experiments.predictive import run as predictive_run
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'paper';predictive_campaign(output)
            report=predictive_run(output,exclude_split=2)
            self.assertEqual(report['fits'],30);self.assertEqual(report['primary_splits'],5)
            target=Path(report['output']);rows=pd.read_csv(target/'split_metrics.csv')
            self.assertEqual(set(rows.split),set(range(5)))
            main=pd.read_csv(target/'all_splits_aggregate.csv');sensitivity=pd.read_csv(target/'sensitivity_aggregate.csv')
            self.assertTrue((main.nlpd_test_mc_count==5).all())
            self.assertTrue((sensitivity.nlpd_test_mc_count==4).all())
            self.assertEqual(len(list(target.glob('*_pointwise.csv'))),30)
            saved=read(target/'report.json');self.assertTrue(saved['primary_retains_all_splits'])
            self.assertTrue(saved['sensitivity_is_post_hoc']);self.assertFalse(saved['inference_run'])
            # The numerical analysis checks the actual training-input contract.
            path=next((output/'cache').rglob('contract.json'));contract=read(path)
            contract['input_hashes'][0]='corrupted';write(path,contract)
            with self.assertRaisesRegex(ValueError,'cache target'):predictive_run(output)
