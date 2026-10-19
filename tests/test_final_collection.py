"""Saved-result collection, source selection and posterior/metric agreement."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from fixtures import campaign
from hb_experiments.common import read, write
from hb_experiments.final_collection import collect
from hb_experiments.figures import REGRESSION
from hb_experiments.grid import diagnostic_issues


def options(source, output):
    return SimpleNamespace(results_root=source, synthetic_root=None, real_root=None,
                           output_root=output, no_plots=False, dry_run=False)


class FinalCollectionChecks(unittest.TestCase):
    def test_public_campaign_collects_all_runs_and_draws_eleven_figures(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root/'paper'; campaign(source/'runs')
            output = root/'collection'; result = collect(options(source, output))
            self.assertEqual(result['total_runs'], 1170)
            self.assertEqual(result['aggregate_csvs'], 20)
            self.assertEqual(result['public_csvs'], 20)
            self.assertEqual(result['plot_report']['figures'], 11)
            self.assertTrue((output/'figures/main/fig3_likelihood_selection_combined.png').is_file())
            self.assertFalse(result['inference_run'])
            self.assertFalse(result['diagnostics_recomputed'])
            self.assertTrue(read(output/'final_collection.json')['execution_complete'])
            self.assertFalse((source/'runs/workflow').exists())

    def test_overlay_has_priority_and_failed_overlay_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); base = root/'base'; campaign(base)
            rel = Path('realdata_benchmark/dataset_cal_housing/gamma_0.500/eps_0.20/muout_6.0/rep_000/metrics.json')
            payload = read(base/rel); payload['dpd']['rmse_test'] = .321
            overlay = root/'rebuilt'; write(overlay/rel, payload)
            a = options(base, root/'collection'); a.real_root = overlay; a.no_plots = True
            self.assertEqual(collect(a)['total_runs'], 1170)
            self.assertEqual(read(a.output_root/rel)['dpd']['rmse_test'], .321)
            payload['dpd']['rhat_max'] = 1.2; write(overlay/rel, payload)
            a.output_root = root/'other'
            with self.assertRaisesRegex(ValueError, 'diagnostics'):
                collect(a)
            self.assertFalse(a.output_root.exists())

    def test_missing_run_and_mismatching_sample_pair_stop_before_writing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); base = root/'base'; campaign(base)
            a = options(base, root/'collection'); overlay = root/'rebuilt'
            rel = Path(REGRESSION)/'metrics.json'
            payload = read(base/rel); payload['dpd']['rmse_test'] += .1
            write(overlay/rel, payload); a.synthetic_root = overlay
            with self.assertRaisesRegex(FileNotFoundError, 'Matching representative'):
                collect(a)
            self.assertFalse(a.output_root.exists())
            a.synthetic_root = None; (base/rel).unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'Missing planned run'):
                collect(a)
            self.assertFalse(a.output_root.exists())

    def test_dry_run_writes_nothing_and_changed_input_cannot_replace_collection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); base = root/'base'; campaign(base)
            a = options(base, root/'collection'); a.no_plots = True; a.dry_run = True
            self.assertTrue(collect(a)['dry_run']); self.assertFalse(a.output_root.exists())
            a.dry_run = False; collect(a)
            rel = Path(REGRESSION)/'metrics.json'
            payload = read(base/rel); payload['dpd']['rmse_test'] += .1; write(base/rel, payload)
            with self.assertRaisesRegex(ValueError, 'inputs changed'):
                collect(a)

    def test_composed_mh_diagnostics_still_enforce_rhat_and_ess(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d); campaign(base)
            payload = read(base/REGRESSION/'metrics.json')
            payload['student_t'].update(sampler='verified_composed_mh', nuts_divergences=None)
            self.assertEqual(diagnostic_issues(payload, 'regression'), [])
            payload['student_t']['ess_tail_min'] = 10
            self.assertTrue(diagnostic_issues(payload, 'regression'))


if __name__ == '__main__':
    unittest.main()
