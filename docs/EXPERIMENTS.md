# Paper experiments

Use a Python 3.12.8 environment and install `python -m pip install -e ".[experiments]"`. Run commands from the source checkout, or use an editable installation so `hbayes` can locate it.

```bash
hbayes paper plan
hbayes paper run --download-data --dry-run
hbayes paper run --download-data
```

The dry run prints nine native commands and starts no MCMC or download. The complete grid is configured in `configs/paper.json`. It retains 30 synthetic replicates and five real-data splits: density 450, regression 540, benchmark 90, and cleaning 90 settings. The native stage names are `gamma_pilot`, `pilot`, `pilot_realdata`, `synthetic`, `benchmark`, `cleaning`, `review`, `collect`, and `plot`.

Use `--output results/paper` to choose a run directory. Production results use `runs/`, pilot/calibration fits use `calibration/`, and sampler caches use `cache/`. Existing experiment results are not required. Reusing that directory resumes validated native checkpoints. Mismatched settings or altered checkpoints are refused.

To resume a specific stage:

```bash
hbayes paper run --phase synthetic --download-data
hbayes paper run --phase benchmark --download-data
hbayes paper run --phase cleaning --download-data
hbayes paper run --phase review
hbayes paper run --phase collect
hbayes paper run --phase plot
```

New stochastic campaigns can have different failed fits. Unresolved diagnostic failures stop final collection, rather than removing chains, replicate datasets, or splits. The diagnostic queue and fixed-transition kernels are described in [INFERENCE.md](INFERENCE.md). Historical repair/adoption recipes are not part of the public runner. A fresh campaign can need additional diagnostics or sampling; a successful code installation does not guarantee every stochastic fit passes.

## Real-data inputs

`--download-data` permits the original loaders to obtain the real datasets. It checks the paper dimensions and rejects an eight-feature fallback for the 32-feature pumadyn experiment. Upstream fallback files may differ in row order or response units from a historical snapshot.

For existing raw data:

```bash
hbayes paper run --local-data-dir /path/to/datasets --dry-run
hbayes paper run --local-data-dir /path/to/datasets
```

The directory needs `cal_housing.csv`, `pumadyn32fh.csv`, and `kin8nm.csv`, or corresponding `.tsv` files, with a header. Use a column named `target`, or put the response last. Keep the intended row order and units. Raw data and the new dataset cache are sufficient; no earlier fit or result archive is required.

## Figures and result archive

The final plot stage uses the new campaign's saved draws, including for the original-unit regression intercept panel. It checks collected summary hashes and both representative inputs before finalising figures. Public figures are written under `figures/main/` and `figures/appendix/`; the complete native suite is retained under `figures/additional/`.

```bash
hbayes paper pack --archive paper-results.zip
# Include retained sample files as well, while still excluding sampler caches:
hbayes paper pack --archive paper-results-with-samples.zip --include-samples
```

The source package can be published without either archive. Full MCMC is computational work for the experiment environment and was not rerun during this packaging refactor. The original complete transitive environment lock is unavailable; cache contracts record the actual runtime of each new fit.

## Tables

To combine already saved runs (including optional rebuilt metric trees) in a separate directory without MCMC, use `hbayes paper collect --results-root /path/to/results --output-root results/collection`. The Python and bash launchers are `scripts/run_final_collection.py` and `scripts/run_final_collection.sh`. See [the saved-result collection guide](FINAL_COLLECTION_ja.md) for legacy input paths, required representative samples, and the `--no-plots` option. This post-processing command is independent of historical reference CSVs and manifests.

`collect` and `plot` also export LaTeX tables from the new aggregate CSVs. To regenerate them independently:

```bash
hbayes paper tables --output results/paper
```

Each table retains the collected conditions, method labels, means and sample SDs. Displayed ties are bold; oracle rows are excluded from practical-method comparisons. Counts, coverage and epsilon estimates receive no monotone performance ranking. Tables do not compare their cells against frozen paper numbers.

For runs with `--local-data-dir`, repeat that input option when resuming a stage, including review, collect and plot: the local data route is part of the run contract.

## Pointwise predictive diagnostics

After generating benchmark and cleaning fits, retain the new campaign's cache and raw observations, then run:

```bash
hbayes paper diagnose --output results/paper
# Optional common-four-split sensitivity; the primary summary still uses all five:
hbayes paper diagnose --output results/paper --exclude-split 2
```

Defaults are cal-housing, gamma .5 and epsilon .2. The command uses the initial DPD fit and NoCleaning, Oracle, lower, central and upper DPD cleaning fits (30 fits over five splits). It checks each newly generated posterior's cache contract against its training observations and retains all draws/test observations. It writes pointwise NLPD/PIT/interval/variance CSVs, coverage summaries, paired excess-NLPD contributions and all-split aggregates under `diagnostics/predictive/`. It starts no MCMC and has no numerical reference-file dependency. Missing *newly generated* inputs or caches are reported as errors.

`--exclude-split` is a zero-based index explicitly selected for an exploratory sensitivity analysis. It adds a separate common-split aggregate; it never replaces the all-five primary result or excludes individual test observations. Identify an influential split from the new diagnostics before interpreting an exclusion in a new campaign.
