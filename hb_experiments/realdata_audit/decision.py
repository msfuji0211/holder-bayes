"""Score and evaluate prespecified policies; injection labels enter evaluation only."""
from itertools import product

import numpy as np
from scipy.special import expit


KEEP, REMOVE, REVIEW = 0, 1, 2
KF = "K-F"
BOOT = "K-F bootstrap"


def counts(n, epsilon):
    return np.clip(np.rint(n*np.asarray(epsilon)).astype(np.int64), 0, n)


def quantile(k, q):
    """Left inverse of the empirical CDF (the smallest feasible integer)."""
    k = np.sort(k)
    if not len(k) or not 0 < q <= 1:
        raise ValueError("A count quantile needs nonempty draws and q in (0,1]")
    idx = max(0, int(np.ceil(q*len(k) - 1e-10)) - 1)
    return int(k[idx])


def balanced_indices(shape, limit):
    chains, draws = shape[:2]
    per = min(draws, limit//chains)
    idx = np.rint(np.linspace(0, draws-1, per)).astype(int)
    return (np.arange(chains)[:, None]*draws+idx).ravel()


def order(q, severity):
    return np.lexsort((np.arange(len(q)), -severity, -q))


def severity(X, y, theta):
    r = (y-X @ theta[:-1])*np.exp(-theta[-1])
    return theta[-1] + .5*r*r  # additive log(2*pi)/2 cancels in every ranking


def frequencies(X, y, theta, k, chunk_size=32):
    n = len(y)
    hits, average = np.zeros(n, dtype=int), np.zeros(n)
    for start in range(0, len(k), chunk_size):
        t = theta[start:start+chunk_size]
        r = (y[None, :] - t[:, :-1] @ X.T)*np.exp(-t[:, -1, None])
        scores = t[:, -1, None] + .5*r*r
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite Gaussian log density while scoring")
        average += scores.sum(axis=0)
        for s, budget in zip(scores, k[start:start+chunk_size]):
            hits[np.argsort(-s, kind="stable")[:budget]] += 1
    if hits.sum() != k.sum():
        raise AssertionError("Selected memberships and draw-specific counts disagree")
    return hits/len(k), average/len(k)


def fixed_frequency(n, k, ranking):
    # Rank r (1-based) is selected exactly when K >= r.
    survival = np.bincount(k, minlength=n+1)[::-1].cumsum()[::-1][1:]/len(k)
    q = np.empty(n)
    q[ranking] = survival
    return q


def hard_q(n, ranking, budget):
    q = np.zeros(n)
    q[ranking[:budget]] = 1
    return q


def score_models(X, y, fit_point, fit_boot, posteriors, s):
    """No labels or clean responses are accepted by the score builder."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    n = len(y)
    point_sev = severity(X, y, np.asarray(fit_point["theta"]))
    point_rank = np.argsort(-point_sev, kind="stable")
    p = np.array([fit_point["epsilon"]])
    point_k = counts(n, p)
    fits = {KF: (p, point_k, point_rank)}
    triage = {f"{KF} point": (hard_q(n, point_rank, point_k[0]), point_rank)}
    theta, eps, diagnostics = fit_boot
    use = np.array([v["valid"] for v in diagnostics], dtype=bool)
    if use.any():
        eps = eps[use]
        bk = counts(n, eps)
        q, _ = frequencies(X, y, theta[use], bk)
        ranks = order(q, point_sev)
        fits[BOOT] = (eps, bk, ranks)
        triage[BOOT] = (q, ranks)
    for name, pos in posteriors.items():
        flat = pos.reshape(-1, pos.shape[-1])
        eps = expit(-flat[:, -1])
        k = counts(n, eps)
        sev = severity(X, y, flat[:, :-1].mean(axis=0))
        rank = np.argsort(-sev, kind="stable")
        idx = balanced_indices(pos.shape, s["max_fod_draws"])
        q, av = frequencies(X, y, flat[idx, :-1], k[idx])
        fits[name] = (eps, k, rank)
        triage[f"{name} central"] = (hard_q(n, rank, int(np.rint(k.mean()))), rank)
        triage[f"{name} count only"] = (fixed_frequency(n, k[idx], rank), rank)
        triage[f"{name} FoD"] = (q, order(q, av))
    return fits, triage


def actions(q, fp, miss, review):
    # A proxy loss for disagreement with audit selection, not calibrated label probabilities.
    return np.argmin(np.stack([miss*q, fp*(1-q), np.full(len(q), review)]), axis=0)


def matched_actions(ranking, d, r):
    a = np.full(len(ranking), KEEP)
    a[ranking[:d]] = REMOVE
    a[ranking[d:d+r]] = REVIEW
    return a


def triage_loss(a, z, fp, miss, review):
    d, v, k = (a == REMOVE), (a == REVIEW), (a == KEEP)
    err_remove, err_keep = int((d & ~z).sum()), int((k & z).sum())
    n = len(a)
    return dict(removed=int(d.sum()), reviewed=int(v.sum()), retained=int(k.sum()),
                false_removed=err_remove, missed=err_keep, review_rate=float(v.mean()),
                false_removal_rate=err_remove/n, miss_rate=err_keep/n,
                loss=(fp*err_remove+miss*err_keep+review*int(v.sum()))/n)


def capacity_loss(b, m, n, review, shortage):
    return dict(budget=b, budget_rate=b/n, shortage=max(m-b, 0), excess=max(b-m, 0),
                shortage_rate=max(m-b, 0)/n, excess_rate=max(b-m, 0)/n, covers=float(b >= m),
                loss=(review*b+shortage*max(m-b, 0))/n)


def evaluate(data, fits, triage, costs):
    z = data["Z"]
    n, m = len(z), int(z.sum())
    X, y, xt, yt = (np.asarray(data[key], dtype=float) for key in ("X", "y", "X_test", "y_test"))
    cache = {}

    def refit(keep):
        key = np.packbits(keep).tobytes()
        if key not in cache:
            ans = dict(refit_n=int(keep.sum()), rmse=None, mae=None)
            if keep.sum() > X.shape[1]:
                coef, _, rank, _ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
                if rank == X.shape[1]:
                    err = (xt @ coef)*float(data["scaler_y_std"]) + float(data["scaler_y_mean"]) - yt
                    ans.update(rmse=float(np.sqrt(np.mean(err**2))), mae=float(np.mean(abs(err))))
            cache[key] = ans
        return cache[key]

    summary, capacity, removal, policies = [], [], [], []
    q = 1-costs["review"]/costs["shortage"]
    for method, (eps, k, ranking) in fits.items():
        central = int(np.rint(k.mean()))
        lo, hi = quantile(k, .025), quantile(k, .975)
        summary.append(dict(method=method, count_draws=len(k), epsilon_mean=float(eps.mean()),
                            count_mean=float(k.mean()), count_sd=float(k.std()),
                            count_q025=lo, count_q975=hi, count_width=hi-lo))
        budgets = {"point": central} if method == KF else {"central": central, "cost_quantile": quantile(k, q)}
        for rule, b in budgets.items():
            capacity.append(dict(method=method, rule=rule, quantile=q if rule=="cost_quantile" else None,
                                 **capacity_loss(b, m, n, costs["review"], costs["shortage"])))
        cleaning = {"point": central} if method == KF else {"lower": lo, "central": central, "upper": hi}
        for rule, b in cleaning.items():
            remove = hard_q(n, ranking, b).astype(bool)
            tp = int((remove & z).sum())
            removal.append(dict(method=method, rule=rule, removed=b, true_removed=tp, false_removed=b-tp,
                                precision=tp/b if m and b else None, recall=tp/m if m else None,
                                f1=2*tp/(b+m) if m else None,
                                noninjected_removal_rate=(b-tp)/(n-m) if m<n else None, **refit(~remove)))
    removal.append(dict(method="Keep all", rule="unfiltered", removed=0, **refit(np.ones(n, dtype=bool))))
    fps = sorted({costs["false_remove"], *costs["false_remove_grid"]})
    rvs = sorted({costs["review"], *costs["review_grid"]})
    for fp, rv in product(fps, rvs):
        primary = fp == costs["false_remove"] and rv == costs["review"]
        miss = costs["miss"]
        policy = {name: actions(q, fp, miss, rv) for name, (q, _) in triage.items()}
        policy.update({"Keep all": np.full(n, KEEP), "Review all": np.full(n, REVIEW)})
        by_rule = {"risk_proxy": policy}
        if "DPD FoD" in policy:
            a = policy["DPD FoD"]
            d, r = int((a==REMOVE).sum()), int((a==REVIEW).sum())
            by_rule["matched_DPD_workload"] = {name: matched_actions(ranking, d, r)
                                               for name, (_, ranking) in triage.items()}
        for rule, by_method in by_rule.items():
            for name, a in by_method.items():
                detail = dict(method=name, rule=rule, false_remove_cost=fp, miss_cost=miss,
                              review_cost=rv, primary_cost=primary, **triage_loss(a, z, fp, miss, rv))
                if primary:
                    # Oracle verification: drop reviewed injections, retain verified unmodified rows.
                    detail.update(refit((a == KEEP) | ((a == REVIEW) & ~z)))
                policies.append(detail)
    return dict(capacity=capacity, triage=policies, removal=removal, counts=summary)
