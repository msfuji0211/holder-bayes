"""Experiment grid and metric diagnostic definitions; no inference imports."""
from pathlib import Path
METHOD_KEYS = {"DPD": "dpd", "ExpSatDPD": "exp_sat_dpd", "RatSatDPD": "rat_sat_dpd",
               "NLL": "nll", "GammaDiv": "gamma", "StudentT": "student_t"}
GAMMA_DEFINITION = "nakagawa_hashimoto_monotone_gamma_v1"
GAMMA_FLOOR = .05

def relative(kind, gamma, eps, rep, dataset=None):
    tag = format(gamma, ".12g").replace(".", "p")
    if kind == "regression":
        # Keep the established gamma1p0 folder name.
        tag = str(float(gamma)).replace(".", "p")
        return Path(f"exp1_regression_eps_sweep_mu6_gamma{tag}")/f"eps{eps:.2f}"/f"rep{rep:03d}"
    if kind == "density":
        tag = str(float(gamma)).replace(".", "p")
        return Path(f"exp1_simulation_gamma_sweep_gamma{tag}")/"A"/f"eps{eps:.2f}"/f"rep{rep:03d}"
    root = "realdata_benchmark" if kind == "benchmark" else "realdata_cleaning"
    mu = 0. if eps == 0 else 6.
    return Path(root)/f"dataset_{dataset}"/f"gamma_{gamma:.3f}"/f"eps_{eps:.2f}"/f"muout_{mu:.1f}"/f"rep_{rep:03d}"

def jobs(a, phase=None):
    phase = phase or a.phase
    out = []
    def add(kind, gamma, eps, rep, dataset=None):
        out.append({"kind": kind, "gamma": gamma, "eps": eps, "rep": rep, "dataset": dataset,
                    "relative_dir": str(relative(kind, gamma, eps, rep, dataset))})
    if phase in ["pilot", "gamma_pilot"]:
        for g, e in [(.5, .3), (1., .4)]:
            for rep in a.pilot_reps:
                add("regression", g, e, rep)
        add("regression", .1, 0., a.pilot_reps[0])
        add("density", .1, .4, a.pilot_reps[0])
        if phase == "gamma_pilot":
            add("regression", .1, .4, a.pilot_reps[0])
            for gamma in [.1, .5, 1.]:
                add("density", gamma, 0., a.pilot_reps[0])
                if gamma != .1:
                    add("density", gamma, .4, a.pilot_reps[0])
    elif phase == "pilot_realdata":
        dataset = "cal_housing" if "cal_housing" in a.datasets else a.datasets[0]
        for kind in ["benchmark", "cleaning"]:
            add(kind, .5, .2, 0, dataset)
    elif phase == "synthetic":
        for g in a.gammas:
            for kind, eps_list in [("regression", [0., .05, .1, .2, .3, .4]), ("density", [0., .1, .2, .3, .4])]:
                for eps in eps_list:
                    for rep in range(a.replicates):
                        add(kind, g, eps, rep)
    elif phase in ["benchmark", "cleaning"]:
        for dataset in a.datasets:
            for g in a.gammas:
                for eps in [0., .2]:
                    for rep in range(a.real_replicates):
                        add(phase, g, eps, rep, dataset)
    return out

def blocks(payload, kind):
    if kind == "cleaning":
        yield from payload.get("audits", {}).items()
        yield from payload.get("cleaning", {}).items()
    else:
        for method, key in METHOD_KEYS.items():
            if kind == "density" and method == "NLL":
                key = "standard"
            if key in payload:
                yield method, payload[key]

def diagnostic_issues(payload, kind):
    issues = []
    found = 0
    if kind == "cleaning":
        expected_audits = ["DPD", "ExpSatDPD", "RatSatDPD"]
        expected_cleaning = ["NoCleaning", "IFCleaning", "LOFCleaning", "OneClassSVMCleaning"]
        expected_cleaning += [f"Proposed{method}{suffix}" for method in expected_audits
                             for suffix in ["", "_optimistic", "_pessimistic"]]
        if payload.get("eps_true", 0.) > 0:
            expected_cleaning += ["RandomCleaning", "OracleCleaning"]
        for section, expected in [("audits", expected_audits), ("cleaning", expected_cleaning)]:
            for name in expected:
                if name not in payload.get(section, {}):
                    issues.append({"method": name, "issue": "missing_method_metrics", "section": section})
    else:
        for method, key in METHOD_KEYS.items():
            if kind == "density" and method == "NLL":
                key = "standard"
            if key not in payload:
                issues.append({"method": method, "issue": "missing_method_metrics"})
    for name, block in blocks(payload, kind):
        found += 1
        values = [block.get(n) for n in ["rhat_max", "ess_bulk_min", "ess_tail_min"]]
        import math
        good = all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
        if not good or values[0] > 1.01 or min(values[1:]) < 400:
            issues.append({"method": name, "issue": "mixing", "values": values})
        if name == "GammaDiv" and (block.get("gamma_posterior_definition") != GAMMA_DEFINITION
                                   or block.get("sigma_min") != GAMMA_FLOOR or block.get("eta") != 1.):
            issues.append({"method": name, "issue": "gamma_target_definition"})
        final_mh = name in ["DPD", "ExpSatDPD", "RatSatDPD", "GammaDiv"] and block.get("sampler") == "frozen_mixture_independence_mh"
        if not final_mh and block.get("nuts_divergences") != 0:
            issues.append({"method": name, "issue": "nuts_divergences", "value": block.get("nuts_divergences")})
    if not found:
        issues.append({"issue": "missing_method_metrics"})
    return issues
