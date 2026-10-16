"""Public command/output conventions; fixture files are not paper results."""
import contextlib,io,json,tempfile,unittest,zipfile
from pathlib import Path
from hbayes.cli import main
from hb_experiments.artifacts import publish_paper,pack

class PublicPackageChecks(unittest.TestCase):
    def test_public_plan_uses_general_output_and_module_names(self):
        buffer=io.StringIO()
        with contextlib.redirect_stdout(buffer):code=main(['paper','run','--download-data','--dry-run'])
        self.assertEqual(code,0);report=json.loads(buffer.getvalue());self.assertEqual(len(report['commands']),9)
        command=report['commands'][0]
        self.assertEqual(command[2],'hb_experiments.native_run')
        self.assertTrue(command[command.index('--output-dir')+1].endswith('results/paper/runs'))
        self.assertNotIn('mcmc_repair_pilot_20261012',' '.join(command))

    def test_public_artifacts_are_copies_with_exact_source_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'paper';runs=root/'runs';csv=runs/'exp1_simulation_gamma_sweep_gamma0p5/summary_agg.csv'
            csv.parent.mkdir(parents=True);csv.write_text('method,rmse\nDPD,1\n')
            picture=runs/'paper_exp1_suite_section6_revised/main_text/fig1_posterior_regression_simulation.png'
            picture.parent.mkdir(parents=True);picture.write_bytes(b'layout fixture, not an experimental figure')
            result=publish_paper(root)
            self.assertEqual((root/'summaries/density/gamma_0.500/aggregate.csv').read_bytes(),csv.read_bytes())
            self.assertEqual((root/'figures/main/posterior.png').read_bytes(),b'layout fixture, not an experimental figure')
            self.assertFalse((runs/'paper_exp1_suite_section6_revised').exists())
            self.assertTrue((root/'figures/additional/main_text'/picture.name).is_file());self.assertEqual(result['files'],2)

    def test_result_archive_excludes_cache_and_only_opted_in_run_samples(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'paper'
            for name in ['experiment.json','cache/contract.json','calibration/posterior.npz','runs/example/metrics.json','runs/example/samples.npz','figures/.DS_Store']:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'archive fixture')
            for include in [False,True]:
                archive=Path(d)/('samples.zip' if include else 'paper-results.zip');pack(root,archive,include_samples=include)
                with zipfile.ZipFile(archive) as z:
                    names=z.namelist();self.assertIsNone(z.testzip())
                    self.assertFalse(any('/cache/' in n or '/calibration/' in n for n in names))
                    self.assertFalse(any('.DS_Store' in n for n in names))
                    self.assertEqual(any(n.endswith('/samples.npz') for n in names),include)
                    self.assertIn('paper-results/runs/example/metrics.json',names)
            with self.assertRaises(ValueError):pack(root,root/'recursive.zip')

