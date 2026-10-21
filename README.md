# Hölder–Bayes

Python implementation of Hölder–Bayes inference and probabilistic outlier auditing. The library fits Gaussian location–scale and linear regression models, jointly estimates the contamination/audit fraction, and computes observation-level posterior flagging frequencies. The repository also contains data generators, experiment runners, collectors and plots for the accompanying paper. No fitted samples, result CSVs, reference images or historical checksum inventories are shipped or required.

## Installation

Use Python 3.12. From this source checkout, install the inference dependencies:

```bash
python -m pip install -e ".[inference]"
```

The source distribution is named `holder-bayes`; the Python module and command are `hbayes`. This repository does not assume a published PyPI release. The wheel contains the reusable `hbayes` library; paper commands additionally use the source checkout and its experiment configurations.

## Fit a model

```python
import numpy as np
from hbayes import fit_density, fit_regression

rng = np.random.default_rng(42)
x = np.vstack([rng.normal(size=(180, 2)), rng.normal(5, 1, size=(20, 2))])
fit = fit_density(x, gamma=0.5, output_dir="results/fits/density")
print(fit.summary())
print(fit.outlier_probability)

X = rng.normal(size=(200, 2))
y = 1.0 + X @ np.array([1.0, -1.5]) + rng.normal(size=200)
y[-40:] += 6.0
regression = fit_regression(X, y, gamma=1.0, output_dir="results/fits/regression")
print(regression.coefficient_samples.mean(axis=(0, 1)))  # intercept, then slopes
print(regression.predict(X[:5]))
```

Regression adds an intercept and standardises using training observations. Reported coefficients, scales, and predictions use the original units. Density fitting uses original data coordinates and a diagonal covariance model. `phi` selects `dpd`, `exponential_saturated`, or `rational_saturated`. `SamplerConfig` controls seed, chains, warm-up, and retained draws; defaults match the nominal paper settings. These examples perform MCMC.

Every retained chain and rejected MH state is preserved. Diagnostics are returned even when their convergence gate is not met; inspect them before interpreting a fit. An audit probability is a posterior frequency of bottom-K likelihood membership for the supplied batch, rather than a known ground-truth contamination label. See [the Python API](docs/API.md) and the runnable [examples](examples/).

Each fitted model can save `posterior_samples.npz`, `summary.json`, and `audit_scores.csv`. Optional sampler checkpoints live under `cache/` inside its output directory. The top-level `hbayes` import does not load JAX.

## Paper experiments

Install the complete experiment environment in a separate environment if needed. Python 3.12.8 and the top-level dependency versions recorded with the paper are specified by this extra and `requirements-experiments.txt`.

```bash
python -m pip install -e ".[experiments]"
hbayes paper plan
hbayes paper run --download-data --dry-run
hbayes paper run --download-data
```

`python -m hbayes ...` is equivalent to `hbayes ...`. The source-checkout entry point `python run_experiments.py run --download-data` is also available. No earlier experiment outputs are required. Synthetic observations use the fixed seed rules. To read existing raw datasets, replace `--download-data` with `--local-data-dir /path/to/datasets`.

The resolved grid has 450 density, 540 regression, 90 real-data benchmark, and 90 cleaning settings. Pilot fits, production fits, diagnostic review, collection, and plotting are separate resumable stages. Unresolved diagnostic failures prevent final collection. The [experiment guide](docs/EXPERIMENTS.md) describes the stages and recovery tools.

Results are written under `results/paper/`, with `experiment.json`, `runs/`, `summaries/`, `figures/`, and `run_report.json`. Tables are generated from the current aggregate CSVs under `tables/`; `hbayes paper tables` regenerates them without fitting. Main figures are `posterior.png` (including the regression intercept), `accuracy.png`, and `audit_scores.png`. Source run identifiers remain stable for checkpoint recovery; public summaries and figures have descriptive names. See [output formats](docs/OUTPUTS.md).

```bash
hbayes paper pack --archive paper-results.zip
```

The result archive contains metrics, summaries, figures, and manifests. Calibration and sampler caches are excluded; `--include-samples` additionally includes retained run sample files. Generated outputs are ignored by Git.

## Development and validation

```bash
python -m pip install -e ".[test]"
hbayes check
python -m unittest discover -s tests -v
```

`hbayes check` checks source/configuration availability. `hbayes doctor` reports the installed inference environment. Neither opens a previous result or a frozen checksum inventory. Test observations, metrics and draws are generated in temporary directories; the post-processing integration test exercises all planned run identities, collection, 11 figures, tables and packaging without running MCMC. To also exercise short real inference fits after installing the inference extra:

```bash
HBAYES_TEST_MCMC=1 python -m unittest discover -s tests -p test_inference_smoke.py -v
```

The full paper MCMC campaign is separate computational work. It was not rerun while preparing this source release. A short integration fit does not certify convergence of a paper campaign. Newly generated input/cache hashes protect restart consistency; they are created locally during a run and are unrelated to any historical result bundle. Changes in code, settings or runtime require a new output directory.

The complete original transitive dependency lock and hardware/BLAS build information are unavailable. New posterior draw bytes need not be identical across machines. The [protocol](docs/PROTOCOL.md) records scoring definitions, scale support, diagnostics, and sensitivity-analysis conventions.

## License and citation

This software is released under the [MIT License](LICENSE). Add the final paper citation before public distribution.
