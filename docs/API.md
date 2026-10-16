# Python API

Install from the checkout with `python -m pip install -e ".[inference]"`. The import name is `hbayes`.

## Fitting

`fit_density(x, gamma=0.5, phi="dpd", kappa=None, sampler=None, cache_dir=None, output_dir=None, priors=None)` fits the diagonal-Gaussian location–scale model. `x` is an n-by-d matrix (a vector is treated as one feature), with finite values and positive-definite sample covariance. Density temperature is computed from that covariance in the original coordinates.

`fit_regression(X, y, gamma=1.0, ...)` accepts raw covariates without an intercept column. It adds the intercept, computes the training mean/SD transformation, and uses the original standardised float32 likelihood inputs with locally promoted float64 inference. The regression scale floor is 0.05 in standardised response coordinates. Its response temperature is computed in those same likelihood coordinates. Density H-B has no scale floor.

`phi` accepts `dpd`, `exponential_saturated`, and `rational_saturated`. Their default kappa values are 1, 0.5, and 0.25 respectively. The same target constructors and frozen-proposal MH implementation used in the paper are called; this interface does not define a different posterior.

```python
from hbayes import SamplerConfig, fit_regression

settings = SamplerConfig(chains=4, warmup=4000, draws=4000, seed=42,
                         target_accept=0.99, initial_step=0.1)
fit = fit_regression(X, y, sampler=settings, output_dir="results/fits/regression")
```

`priors` may be an existing `PriorsConfig` / `MVNPriorsConfig` instance from `hbayes.posteriors`; the reviewed defaults are used when omitted. The sampler keeps its resumable cache. With `output_dir`, the default cache is its `cache/` subdirectory; otherwise it is `results/cache/`. A supplied `cache_dir` overrides that location. Cache identity binds data, prior, target, runtime, and sampler settings. The API scopes and restores its process environment and existing job context.

## Returned result

| Attribute / method | Meaning |
| --- | --- |
| `samples` | All retained physical positions, shape `(chains, draws, parameters)`; regression positions use standardised likelihood coordinates |
| `epsilon_samples` | `sigmoid(-rho)` for every retained state, shape `(chains, draws)` |
| `location_samples` | Density location in original units |
| `coefficient_samples` | Regression coefficients in original units; intercept first, then slopes; shape `(chains, draws, p+1)` |
| `scale_samples` | Gaussian standard deviation in original units; density has one scale per feature |
| `outlier_probability` | Training-batch posterior bottom-K flagging frequency for each observation |
| `diagnostics` | Original R-hat, ESS, sampler statistics, and diagnostic status, without filtering failed chains |
| `summary()` | Global fraction mean/95% interval, configuration, and diagnostics |
| `predict(X_new)` | Regression posterior mean response in original units |
| `save(directory)` | Save `posterior_samples.npz`, `summary.json`, and `audit_scores.csv` |

Scores use at most 400 fixed-seed draws and `K=round(n*epsilon)` as in the paper. `epsilon_samples` and its summary retain all draws; in the historical density experiment, the saved scoring/fraction summary used the scoring subset. Thus the reusable API's full-draw fraction summary is explicitly distinct from that historical summary, while the paper workflow remains unchanged.

The low-level `audit_probabilities(samples, inputs, model, max_draws=400, seed=42)` accepts the actual likelihood inputs: density points, or regression design matrix (including intercept) and standardised response. Its scores depend on the supplied batch, not on a fixed universal threshold. A passing parameter-chain diagnostic does not establish accuracy of an extreme-tail predictive density integral.

The two public fitting functions invoke MCMC. No MCMC is started on import. The `examples/` programs are complete fitting examples; unit-test fixtures are numerical checks rather than experimental results.

Release-preparation tests compare numerical auditing and preprocessing with the original functions and test backend dispatch with fixtures. A new end-to-end JAX fit was not run during this interface refactor; the unchanged sampler has separate numerical and integrity tests.
