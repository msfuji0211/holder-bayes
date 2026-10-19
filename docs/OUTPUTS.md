# Saved outputs

No date or one-off repair label is required in an output filename. Set `output_dir` in the Python API, or `--output` in the paper command, to keep independent analyses separate.

| Scope | Default location / filename | Contents |
| --- | --- | --- |
| Reusable model fit | `results/fits/<chosen-name>/posterior_samples.npz` | Full chain arrays, epsilon, original-unit parameters, and audit probabilities |
| Reusable model fit | `summary.json` | Configuration, fraction interval, and diagnostics |
| Reusable model fit | `audit_scores.csv` | Observation index and posterior flagging frequency |
| Paper run | `results/paper/experiment.json` | Explicit command sequence, input route, runtime, and wrapper hashes |
| Paper run | `results/paper/runs/` | Per-setting metrics, raw samples, and resumable run receipts |
| Paper run | `results/paper/summaries/density/gamma_0.500/runs.csv` | Per-run density metrics; other gamma values use the same structure |
| Paper run | `results/paper/summaries/regression/gamma_1.000/aggregate.csv` | Across-replicate aggregates |
| Paper run | `results/paper/summaries/benchmark/` | `runs.csv`, `aggregate.csv`, and `paper_setting.csv` |
| Paper run | `results/paper/summaries/cleaning/` | Run/aggregate metrics and audit summaries |
| Paper figures | `results/paper/figures/main/posterior.png` | Density posterior and regression posterior including the intercept |
| Paper figures | `accuracy.png`, `audit_scores.png` | Main accuracy and representative observation/audit plots |
| Appendix figures | `results/paper/figures/appendix/` | Descriptive phi, ranking, and gamma-sensitivity filenames |
| Paper tables | `results/paper/tables/` | LaTeX tables and numerical cells computed from current aggregate CSVs |
| Paper run | `results/paper/artifact_manifest.json` | SHA-256 hashes of public CSV and figure copies |
| Paper run | `results/paper/run_report.json` | Completion and generated-artifact report |
| Paper archive | `paper-results.zip` | Public outputs and per-fit metrics, with an archive manifest |

Default model fits with `output_dir=None` return an in-memory `FitResult` and use `results/cache/` for resumable sampler state; no `results/fits` name is invented automatically. The examples choose their output directory explicitly.

Per-setting native identifiers (gamma/contamination/replicate and recorded fit contracts) remain stable inside `runs/` so collection and recovery can find the correct observations and checkpoints. Public summary/figure names are independent of those identifiers. Advanced native plots remain available under `figures/additional/`.

`paper pack` excludes `cache/`, `calibration/`, staging directories, and native sample files by default. `--include-samples` includes `.npz` run samples, not fit caches. Standard figures may include representative posterior plotting arrays. Generated result archives are separate from the source distribution and are not needed to start a new experiment.

`hbayes paper diagnose` writes `diagnostics/predictive/<dataset>/gamma_<value>/eps_<value>/` with pointwise CSVs, per-fit summaries, `split_metrics.csv`, `all_splits_aggregate.csv`, optional `sensitivity_aggregate.csv`, and a paired-diagnostic report. These are generated results; none is shipped with the source.

The paper plotting workflow also writes a vector `.pdf` beside every `.png`. PNG exports use at least 600 dpi. Multipanel figures use uppercase panel labels, excluding colour bars. The public artifact-copying step retains both formats with matching descriptive stems.
