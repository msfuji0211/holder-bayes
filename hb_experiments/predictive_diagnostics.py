"""Gaussian posterior predictive diagnostics, using every saved draw."""
from __future__ import annotations

import math
import numpy as np
from scipy.special import logsumexp, ndtr, ndtri

LOG2PI = math.log(2.0 * math.pi)
PROBABILITIES = np.array([0.025, 0.1, 0.9, 0.975])


def fixed_scaler(raw):
    """Match clean-training standardisation and its float32 design exactly."""
    x = np.asarray(raw['x_train_raw'], dtype=float)
    y = np.asarray(raw['y_train_raw_clean'], dtype=float)
    xt = np.asarray(raw['x_test_raw'], dtype=float)
    if x.ndim != 2 or xt.ndim != 2 or x.shape[1] != xt.shape[1]:
        raise ValueError('Invalid raw covariate shapes')
    if not all(np.isfinite(a).all() for a in [x, y, xt, raw['y_test_raw']]):
        raise ValueError('Non-finite saved observations')
    xs = np.where(x.std(axis=0) < 1e-9, 1.0, x.std(axis=0))
    ys = float(y.std())
    scaler = dict(x_mean=x.mean(axis=0), x_std=xs, y_mean=float(y.mean()),
                  y_std=1.0 if ys < 1e-12 else ys)
    design = lambda a: np.column_stack([np.ones(len(a)), (a-scaler['x_mean'])/xs]).astype(np.float32)
    return design(x), design(xt), scaler


def training_inputs(raw, label, X, scaler):
    if label == 'HBinitial':
        np.testing.assert_array_equal(raw['X'], X)
        return np.asarray(raw['X'], dtype=np.float64), np.asarray(raw['y'], dtype=np.float64), np.arange(len(X))
    if label == 'HBinitial_from_cleaning':
        y = ((np.asarray(raw['y_train_raw_contam'], dtype=float)-scaler['y_mean'])/scaler['y_std']).astype(np.float32)
        return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), np.arange(len(X))
    removed = np.asarray(raw['removed_indices_'+label])
    if removed.ndim != 1 or removed.dtype.kind not in 'iu':
        raise ValueError('Invalid saved removal indices')
    if len(np.unique(removed)) != len(removed) or np.any(removed < 0) or np.any(removed >= len(X)):
        raise ValueError('Repeated or out-of-range removal index')
    keep = np.ones(len(X), dtype=bool)
    keep[removed] = False
    y = ((np.asarray(raw['y_train_raw_contam'], dtype=float)-scaler['y_mean'])/scaler['y_std']).astype(np.float32)
    return np.asarray(X[keep], dtype=np.float64), np.asarray(y[keep], dtype=np.float64), np.flatnonzero(keep)


def mixture_quantiles(mu, sigma, *, probabilities=PROBABILITIES, cdf_tolerance=1e-8, max_iterations=80):
    """Invert the finite Gaussian-mixture CDF by safeguarded Newton steps.

    These are response predictive quantiles, not quantiles of fitted means.
    No new predictive simulation, parameter sampling, or draw thinning occurs.
    """
    mu, sigma = np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if mu.ndim != 2 or sigma.shape != (len(mu),) or not np.all(sigma > 0):
        raise ValueError('Invalid Gaussian mixture')
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        raise ValueError('Non-finite mixture')
    if not np.all((probabilities > 0) & (probabilities < 1)) or not 0 < cdf_tolerance < 1e-3:
        raise ValueError('Invalid quantile probabilities or CDF tolerance')
    centre = mu.mean(axis=0)
    sd = np.sqrt(np.mean(sigma**2)+mu.var(axis=0, ddof=0))
    out = np.empty((mu.shape[1], len(probabilities)))
    max_error = 0.0
    max_used = 0
    for j, p in enumerate(probabilities):
        zp = float(ndtri(p))
        component_q = mu+sigma[:, None]*zp
        lo, hi = component_q.min(axis=0), component_q.max(axis=0)
        current = np.clip(centre+sd*zp, lo, hi)
        active = np.ones(mu.shape[1], dtype=bool)
        for iteration in range(max_iterations):
            idx = np.flatnonzero(active)
            if not len(idx):
                break
            z = (current[idx][None]-mu[:, idx])/sigma[:, None]
            residual = ndtr(z).mean(axis=0)-p
            done = np.abs(residual) <= cdf_tolerance
            if np.any(done):
                max_error = max(max_error, float(np.abs(residual[done]).max()))
                active[idx[done]] = False
            remaining = ~done
            if not np.any(remaining):
                max_used = max(max_used, iteration+1)
                continue
            ii, rr, zz = idx[remaining], residual[remaining], z[:, remaining]
            hi[ii] = np.where(rr > 0, current[ii], hi[ii])
            lo[ii] = np.where(rr < 0, current[ii], lo[ii])
            pdf = (np.exp(-0.5*zz**2)/sigma[:, None]).mean(axis=0)/math.sqrt(2*math.pi)
            with np.errstate(divide='ignore', invalid='ignore'):
                candidate = current[ii]-rr/pdf
            valid = np.isfinite(candidate) & (candidate > lo[ii]) & (candidate < hi[ii])
            current[ii] = np.where(valid, candidate, 0.5*(lo[ii]+hi[ii]))
        if np.any(active):
            raise ArithmeticError(f'Predictive quantiles did not meet CDF tolerance for p={p}; no interval reported')
        out[:, j] = current
    if np.any(np.diff(out, axis=1) < 0):
        raise ArithmeticError('Non-monotone predictive quantiles')
    return out, max_error, max_used


def diagnose(samples, X_test, y_test, scaler, *, observation_batch=64, cdf_tolerance=1e-8):
    samples = np.asarray(samples, dtype=np.float64)
    X = np.asarray(X_test, dtype=np.float64)
    y = np.asarray(y_test, dtype=np.float64)
    d = X.shape[1]
    if samples.ndim != 2 or samples.shape[1] not in [d+1, d+2] or len(samples) < 1:
        raise ValueError('Unexpected physical-coordinate regression draws')
    if y.shape != (len(X),) or not all(np.isfinite(a).all() for a in [samples, X, y]):
        raise ValueError('Invalid test data or posterior draws')
    if observation_batch < 1:
        raise ValueError('Observation batch must be positive')
    sigma = np.maximum(float(scaler['y_std'])*np.exp(samples[:, d]), 1e-8)
    if not np.isfinite(sigma).all():
        raise ValueError('Non-finite residual scales')
    sigma_hat = max(float(scaler['y_std'])*float(np.exp(samples[:, d].mean())), 1e-8)
    noise_var = float(np.mean(sigma**2))
    mean_beta = samples[:, :d].mean(axis=0)
    prediction = float(scaler['y_mean'])+float(scaler['y_std'])*(X@mean_beta)
    residual = y-prediction
    arrays = dict(test_index=np.arange(len(y), dtype=np.int64), predictive_mean=prediction,
                  residual=residual, squared_error=residual**2, abs_error=np.abs(residual),
                  nlpd_plugin=0.5*LOG2PI+np.log(sigma_hat)+0.5*(residual/sigma_hat)**2,
                  residual_variance=np.full(len(y), noise_var))
    for key in ['nlpd_mc', 'pit', 'coefficient_variance', 'q025', 'q10', 'q90', 'q975']:
        arrays[key] = np.empty(len(y))
    max_cdf_error, max_iterations = 0.0, 0
    for begin in range(0, len(y), observation_batch):
        sl = slice(begin, min(begin+observation_batch, len(y)))
        mu = float(scaler['y_mean'])+float(scaler['y_std'])*(samples[:, :d]@X[sl].T)
        z = (y[sl][None]-mu)/sigma[:, None]
        lp = -0.5*LOG2PI-np.log(sigma)[:, None]-0.5*z**2
        arrays['nlpd_mc'][sl] = -(logsumexp(lp, axis=0)-np.log(len(samples)))
        arrays['pit'][sl] = ndtr(z).mean(axis=0)
        arrays['coefficient_variance'][sl] = mu.var(axis=0, ddof=0)
        quantiles, error, iterations = mixture_quantiles(mu, sigma, cdf_tolerance=cdf_tolerance)
        for j, key in enumerate(['q025', 'q10', 'q90', 'q975']):
            arrays[key][sl] = quantiles[:, j]
        max_cdf_error, max_iterations = max(max_cdf_error, error), max(max_iterations, iterations)
    arrays['predictive_variance'] = arrays['residual_variance']+arrays['coefficient_variance']
    arrays['predictive_sd'] = np.sqrt(arrays['predictive_variance'])
    arrays['standardized_residual'] = residual/arrays['predictive_sd']
    arrays['covered_80'] = (y >= arrays['q10']) & (y <= arrays['q90'])
    arrays['covered_95'] = (y >= arrays['q025']) & (y <= arrays['q975'])
    summary = dict(n_test=len(y), posterior_draws=len(samples), sigma_hat=sigma_hat,
                   rmse_test=float(np.sqrt(np.mean(residual**2))), mae_test=float(np.mean(np.abs(residual))),
                   nlpd_test=float(arrays['nlpd_plugin'].mean()), nlpd_test_mc=float(arrays['nlpd_mc'].mean()),
                   residual_variance_mean=noise_var,
                   coefficient_variance_mean=float(arrays['coefficient_variance'].mean()),
                   coefficient_variance_max=float(arrays['coefficient_variance'].max()),
                   predictive_variance_mean=float(arrays['predictive_variance'].mean()),
                   quantile_max_cdf_error=max_cdf_error, quantile_max_iterations=max_iterations,
                   pit_histogram_10=np.histogram(arrays['pit'], bins=np.linspace(0, 1, 11))[0].tolist())
    for level, lower, upper in [(80, 'q10', 'q90'), (95, 'q025', 'q975')]:
        summary.update({f'coverage_{level}': float(arrays[f'covered_{level}'].mean()),
                        f'interval_width_{level}': float((arrays[upper]-arrays[lower]).mean()),
                        f'lower_miss_{level}': float((y < arrays[lower]).mean()),
                        f'upper_miss_{level}': float((y > arrays[upper]).mean())})
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ArithmeticError('Non-finite pointwise diagnostics')
    return arrays, summary


def add_design_diagnostics(arrays, X_train, X_test):
    """Design leverage is a diagnostic, not an estimated error probability."""
    X, Xt = np.asarray(X_train, dtype=float), np.asarray(X_test, dtype=float)
    inv = np.linalg.pinv(X.T@X, rcond=1e-12, hermitian=True)
    arrays['design_leverage'] = np.einsum('ij,jk,ik->i', Xt, inv, Xt)
    arrays['max_abs_standardized_covariate'] = np.max(np.abs(Xt[:, 1:]), axis=1)
    arrays['outside_retained_covariate_range'] = np.any((Xt[:, 1:] < X[:, 1:].min(axis=0)) |
                                                      (Xt[:, 1:] > X[:, 1:].max(axis=0)), axis=1)
    condition = float(np.linalg.cond(X))
    return dict(retained_design_rank=int(np.linalg.matrix_rank(X)),
                retained_design_condition_number=condition if np.isfinite(condition) else None)


def pairwise_excess(case, baseline):
    """Contributions to the mean score difference; all observations are kept."""
    delta = np.asarray(case['nlpd_mc'])-np.asarray(baseline['nlpd_mc'])
    order = np.argsort(-delta, kind='stable')
    positive = float(np.maximum(delta, 0).sum())
    groups = []
    for fraction in [0.001, 0.01, 0.05]:
        k = max(1, int(np.ceil(fraction*len(delta))))
        idx = order[:k]
        groups.append(dict(fraction=fraction, count=k, test_indices=idx.tolist(),
                           contribution_to_mean_difference=float(delta[idx].sum()/len(delta)),
                           fraction_of_positive_excess_mass=None if positive == 0 else
                           float(np.maximum(delta[idx], 0).sum()/positive)))
    return dict(mean_nlpd_difference=float(delta.mean()), positive_excess_count=int((delta > 0).sum()),
                top_positive_excess_groups=groups, top20_test_indices=order[:20].tolist())
