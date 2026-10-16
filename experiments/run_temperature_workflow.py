"""Paper workflow: finish experiment grids, then repair recorded MCMC failures.

Use hbayes paper run. Synthetic observations are generated from fixed seeds.
Raw real datasets may be downloaded or supplied as local CSV/TSV files.
"""
from __future__ import annotations

import argparse
import ast
import csv
import copy
from dataclasses import asdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
from contextlib import contextmanager
import socket

from hbayes.paper_sampling import VERSION, array_hash, file_hash, job_context, CONTEXT
from hb_experiments.grid import relative, jobs, blocks, diagnostic_issues
from experiments.storage_safety import (atomic_json, atomic_npz, atomic_text, atomic_bytes,
    ensure_capacity, install_storage_safety, metadata as storage_metadata, save_error_report)
from hbayes.gamma_posterior import DEFINITION as GAMMA_DEFINITION, SIGMA_MIN as GAMMA_FLOOR, specification as gamma_specification

METHOD_KEYS = {"DPD": "dpd", "ExpSatDPD": "exp_sat_dpd", "RatSatDPD": "rat_sat_dpd",
               "NLL": "nll", "GammaDiv": "gamma", "StudentT": "student_t"}

POLICY = {"version": "deferred_mcmc_diagnostics_v2", "rhat_max": 1.01,
          "ess_bulk_min": 400., "ess_tail_warning_min": 300.,
          "ess_tail_pass_min": 400., "final_nuts_divergences": 0,
          "completed_mcmc_failures": "save_and_continue",
          "integrity_or_definition_errors": "stop",
          "pilot_admission": "retain_v1_criteria", "final_collection": "require_all_resolved"}
_HELD_WORKFLOW_LOCKS = set()
JOB_FIELDS = ["kind", "gamma", "eps", "rep", "dataset", "relative_dir"]


def numbers(text, cast=float):
    return [cast(s) for s in text.replace(",", " ").split()]


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["gamma_pilot", "pilot", "pilot_realdata", "synthetic", "benchmark", "cleaning", "review", "collect", "plot", "preflight", "status", "restore_pilots"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--pilot-policy", choices=["auto", "require"], default=os.environ.get("PILOT_POLICY", "auto"),
                   help="auto: reuse/restore verified pilot results and run missing pilots automatically; require: manual pilot admission")
    p.add_argument("--output-dir", type=Path, default=Path(os.environ.get("RESULTS_ROOT", "outputs/temperature_gamma_nh_v3")))
    p.add_argument("--pilot-dir", type=Path, default=Path(os.environ.get("PILOT_ROOT", "outputs/temperature_gamma_nh_pilot_v3")))
    p.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("FIT_CACHE_ROOT", "outputs/temperature_floor005_cache_v2")))
    p.add_argument("--existing-root", type=Path, default=Path(os.environ.get("EXISTING_RESULTS_ROOT", "outputs/temperature_corrected")))
    p.add_argument("--reuse-pilot-dir", type=Path, default=Path(os.environ.get("REUSE_PILOT_ROOT", "outputs/temperature_floor005_pilot_v2")),
                   help="Previous pilot observations and non-GammaDiv results; treated as read-only")
    p.add_argument("--mh30-root", type=Path, default=Path(os.environ["MH30_ROOT"]) if os.environ.get("MH30_ROOT") else None)
    p.add_argument("--proposal-root", type=Path, default=Path(os.environ["PROPOSAL_ROOT"]) if os.environ.get("PROPOSAL_ROOT") else None,
                   help="Original all30 proposals directory containing inputs/repXXX/data.npz")
    p.add_argument("--local-data-dir", default=os.environ.get("LOCAL_DATA_DIR"))
    p.add_argument("--gammas", default=os.environ.get("GAMMAS", "0.1 0.5 1.0"))
    p.add_argument("--datasets", default=os.environ.get("DATASETS", "cal_housing,pumadyn32fh,kin8nm"))
    p.add_argument("--replicates", type=int, default=int(os.environ.get("REPLICATES", "30")))
    p.add_argument("--real-replicates", type=int, default=int(os.environ.get("REAL_REPLICATES", "5")))
    p.add_argument("--pilot-reps", default=os.environ.get("PILOT_REPS", "0,1,4"))
    p.add_argument("--chains", type=int, default=int(os.environ.get("CHAINS", "4")))
    p.add_argument("--warmup", type=int, default=int(os.environ.get("NUM_WARMUP", "4000")))
    p.add_argument("--samples", type=int, default=int(os.environ.get("NUM_SAMPLES", "4000")))
    p.add_argument("--mh-warmup", type=int, default=int(os.environ.get("MH_WARMUP", "4000")))
    p.add_argument("--mh-samples", type=int, default=int(os.environ.get("MH_SAMPLES", "4000")))
    p.add_argument("--target-accept", type=float, default=float(os.environ.get("TARGET_ACCEPT", "0.99")))
    p.add_argument("--init-step", type=float, default=float(os.environ.get("INIT_STEP", "0.1")))
    a = p.parse_args(argv)
    a.gammas = numbers(a.gammas)
    a.datasets = a.datasets.replace(",", " ").split()
    a.pilot_reps = numbers(a.pilot_reps, int)
    if (a.chains < 2 or min(a.warmup, a.samples, a.mh_warmup, a.mh_samples) < 20
            or not 0 < a.target_accept < 1 or a.init_step <= 0
            or not a.gammas or any(g <= 0 for g in a.gammas)
            or not a.datasets or min(a.replicates, a.real_replicates) < 1
            or not a.pilot_reps or any(r < 0 for r in a.pilot_reps)
            or len(set(a.gammas)) != len(a.gammas) or len(set(a.datasets)) != len(a.datasets)
            or len(set(a.pilot_reps)) != len(a.pilot_reps)):
        p.error("Invalid experiment or sampler settings")
    for field in ["output_dir", "pilot_dir", "cache_dir", "existing_root", "reuse_pilot_dir"]:
        setattr(a, field, getattr(a, field).resolve())
    if (a.existing_root/"results").is_dir() and not any(a.existing_root.glob("exp1_*")):
        a.existing_root = a.existing_root/"results"
    if a.mh30_root is not None:
        a.mh30_root = a.mh30_root.resolve()
    if a.proposal_root is not None:
        a.proposal_root = a.proposal_root.resolve()
    elif a.mh30_root:
        for candidate in [a.mh30_root/"proposals", Path("outputs/regression_gamma1_eps04_all30_v1/proposals")]:
            if (candidate/"inputs/rep000/data.npz").exists():
                a.proposal_root = candidate.resolve()
                break
    protected = [a.existing_root, a.reuse_pilot_dir] + ([a.mh30_root] if a.mh30_root else []) + ([a.proposal_root] if a.proposal_root else [])
    for output in [a.output_dir, a.pilot_dir, a.cache_dir]:
        for source in protected:
            if output == source or output.is_relative_to(source) or source.is_relative_to(output):
                p.error("Source results and new outputs/cache must use separate, non-nested directories")
    if any(x == y or x.is_relative_to(y) or y.is_relative_to(x)
           for i, x in enumerate([a.output_dir, a.pilot_dir, a.cache_dir])
           for y in [a.output_dir, a.pilot_dir, a.cache_dir][i+1:]):
        p.error("New results, pilot results and fit cache need separate directories")
    return a


def code_sources():
    root = Path(__file__).resolve().parent.parent
    paths = [root/"hbayes"/n for n in ["models.py", "dpd.py", "posteriors.py", "sampling.py",
             "bounded_scale.py", "paper_sampling.py", "independence_mh.py", "diagnostics.py", "experiment_temperature.py",
             "gamma_posterior.py", "gamma_sampling.py"]]
    paths += [root/"hb_experiments/grid.py", root/"configs/paper.json", Path(__file__), *sorted((root/"experiments/exp1_regression").glob("*.py")),
              *sorted((root/"experiments/exp1_simulation").glob("*.py"))]
    return {str(p.relative_to(root)): file_hash(p) for p in paths if p.is_file()}


def sampler_contract(a):
    import importlib.metadata
    runtime = {"python": sys.version.split()[0]}
    for name in ["numpy", "scipy", "jax", "jaxlib", "blackjax", "arviz"]:
        try:
            runtime[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            runtime[name] = None
    return {"version": VERSION, "regression_holder_sigma_min_standardized": .05, "density_holder_sigma_min": None,
            "gamma_div_protocol": gamma_specification(),
            "pilot_reps": a.pilot_reps,
            "chains": a.chains, "warmup": a.warmup, "samples": a.samples,
            "mh_warmup": a.mh_warmup, "mh_samples": a.mh_samples,
            "target_accept": a.target_accept, "init_step": a.init_step,
            "source_sha256": code_sources(), "runtime": runtime}










def classify_issues(issues):
    """Separate incomplete/invalid results from completed statistical failures."""
    blocking, warnings, failures = [], [], []
    for issue in issues:
        values = issue.get("values", [])
        tail_only = (issue.get("issue") == "mixing" and len(values) == 3
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) for v in values)
            and 0 < values[0] <= POLICY["rhat_max"]
            and values[1] >= POLICY["ess_bulk_min"]
            and POLICY["ess_tail_warning_min"] <= values[2] < POLICY["ess_tail_pass_min"])
        if tail_only:
            warnings.append({**issue, "issue": "tail_ess_warning", "severity": "warning",
                             "action": "Review tail precision or append a declared fixed segment before final collection."})
        elif (issue.get("issue") == "mixing" and len(values) == 3
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)):
            failures.append({**issue, "severity": "needs_repair"})
        elif (issue.get("issue") == "nuts_divergences"
                and isinstance(issue.get("value"), (int, float)) and not isinstance(issue["value"], bool)
                and math.isfinite(issue["value"]) and issue["value"] > 0
                and issue["value"] == int(issue["value"])):
            failures.append({**issue, "severity": "needs_repair"})
        else:
            blocking.append(issue)
    return blocking, warnings, failures


def diagnostic_row(job, issues, **extra):
    blocking, warnings, failures = classify_issues(issues)
    return {**job, **extra, "diagnostic_issues": issues, "blocking_issues": blocking,
            "diagnostic_warnings": warnings, "diagnostic_failures": failures,
            "diagnostic_status": "blocked" if blocking else "needs_repair" if failures else "warning" if warnings else "pass"}


def pending_fit_details(payload, kind, issues):
    """Keep cache locations for later per-method repair, without changing draws."""
    names = {i.get("method") for i in issues}
    fields = ["sampler", "fit_cache", "fit_contract_sha256", "rhat_max", "ess_bulk_min", "ess_tail_min",
              "nuts_divergences", "mh_acceptance_mean", "retained_draws_per_chain", "additional_draws_per_chain"]
    return {name: {k: block.get(k) for k in fields} for name, block in blocks(payload, kind) if name in names}


def verify_nonfinite_diagnostics(payload, kind, issues):
    """A nonfinite diagnostic may be a constant chain; nonfinite states are fatal."""
    import numpy as np
    bad = {i.get("method") for i in issues if i.get("issue") == "mixing"
           and all(isinstance(v, (int, float)) for v in i.get("values", []))
           and any(not math.isfinite(v) for v in i.get("values", []))}
    if not bad:
        return issues
    from hbayes.paper_sampling import _restore
    checks = []
    for name, block in blocks(payload, kind):
        if name not in bad:
            continue
        cache = block.get("fit_cache")
        saved = _restore(Path(cache)/"posterior.npz") if cache else None
        if (saved is None or "samples_physical" not in saved or "log_posterior_physical" not in saved):
            checks.append({"method": name, "issue": "nonfinite_diagnostics_without_verified_draws"})
        elif (not np.isfinite(saved["samples_physical"]).all()
              or not np.isfinite(saved["log_posterior_physical"]).all()):
            checks.append({"method": name, "issue": "nonfinite_saved_states"})
    return issues + checks


def update_report(report, complete=False):
    rows = report["jobs"]
    warnings = sum(len(r.get("diagnostic_warnings", [])) for r in rows)
    failures = sum(len(r.get("diagnostic_failures", [])) for r in rows)
    blocking = sum(len(r.get("blocking_issues", [])) for r in rows)
    report.update(diagnostic_policy=POLICY, storage_policy=storage_metadata(), execution_complete=complete,
                  all_diagnostics_pass=bool(complete and not warnings and not failures and not blocking),
                  diagnostic_summary={"saved_datasets": len(rows), "warning_methods": warnings,
                      "blocking_issues": blocking,
                      "deferred_failures": failures,
                      "repair_datasets": sum(bool(r.get("diagnostic_failures")) for r in rows),
                      "warning_datasets": sum(bool(r.get("diagnostic_warnings")) for r in rows)})


def contracts_compatible(saved, current):
    """Resume only the same code, inputs, sampler settings and runtime."""
    return isinstance(saved, dict) and saved == current


def write_config(path, cfg):
    if path.exists():
        old = json.loads(path.read_text())
        if (old.get("experiment_contract") != cfg["experiment_contract"]
                or not contracts_compatible(old.get("sampler_contract"), cfg["sampler_contract"])):
            raise ValueError("Output contains different experiment settings; choose a new output-dir/pilot-dir")
        if old != cfg:
            backup = path.with_name("workflow_config.before_deferred_diagnostics.json")
            if not backup.exists():
                atomic_json(backup, old)
    atomic_json(path, cfg)


def migrate_receipt(path, receipt, current):
    """Record real current source hashes without changing any numerical outputs."""
    if not contracts_compatible(receipt.get("sampler_contract"), current):
        raise ValueError(f"Run receipt differs: {path}")
    if receipt["sampler_contract"] != current:
        history = receipt.setdefault("sampler_contract_history", [])
        for prior in [receipt.get("previous_sampler_contract"), receipt["sampler_contract"]]:
            if prior is not None and prior not in history:
                history.append(prior)
        receipt["previous_sampler_contract"] = receipt["sampler_contract"]
        receipt["contract_migration_reason"] = "storage_io_v1: inference and data-generation sources unchanged"
        receipt["sampler_contract"] = current
        atomic_json(path, receipt)


def forced_conditions(a):
    """Retry entire method/condition groups, including seeds which passed before."""
    groups = {}
    for job in jobs(a, "synthetic")+jobs(a, "benchmark"):
        path = a.existing_root/job["relative_dir"]/"metrics.json"
        if path.exists():
            payload = json.loads(path.read_text())
            key = (job["kind"], job["gamma"], job["eps"], job["dataset"])
            blocking, _, _ = classify_issues(diagnostic_issues(payload, job["kind"]))
            groups.setdefault(key, set()).update(i["method"] for i in blocking if "method" in i)
    return groups


def verified_report(root, name, a):
    path = root/"workflow"/f"{name}.json"
    if not path.exists():
        raise RuntimeError(f"Missing {name} report: {path}. "
                           "Run the missing phase with the same output and input settings. "
                           f"If its result files are also missing, run phase {name} with the same roots/settings; "
                           "verified fit caches are reused. Do not delete production outputs/cache.")
    report = json.loads(path.read_text())
    current = sampler_contract(a)
    if not contracts_compatible(report.get("sampler_contract"), current):
        raise RuntimeError(f"{name} code/sampler settings differ: {path}")
    if name in ["synthetic", "benchmark", "cleaning"] and report.get("experiment_contract") != experiment_contract(a):
        raise ValueError(f"Experiment grid differs: {path}")
    complete = report.get("execution_complete", report.get("all_diagnostics_pass", False))
    refreshed = []
    for row in report.get("jobs", []):
        job = {k: row[k] for k in JOB_FIELDS}
        if any(i.get("issue") == "execution_error" for i in row.get("diagnostic_issues", [])):
            refreshed.append(diagnostic_row(job, row["diagnostic_issues"], origin=row.get("origin")))
            continue
        folder = root/row["relative_dir"]
        receipt = json.loads((folder/"workflow_receipt.json").read_text())
        if receipt.get("job") != job or not contracts_compatible(receipt.get("sampler_contract"), current):
            raise ValueError(f"Dataset receipt differs: {folder}")
        metrics_name = "gamma_check.json" if name == "gamma_pilot" else "metrics.json"
        if metrics_name not in receipt.get("output_sha256", {}):
            raise ValueError(f"Metrics are not covered by the saved receipt: {folder}")
        for output_name, digest in receipt["output_sha256"].items():
            if not (folder/output_name).is_file() or file_hash(folder/output_name) != digest:
                raise ValueError(f"Completed result changed: {folder/output_name}")
        payload = json.loads((folder/metrics_name).read_text())
        if name == "gamma_pilot":
            issues = gamma_diagnostic_issues(payload)
            fit_payload = {"gamma": payload}
        else:
            issues = diagnostic_issues(payload, job["kind"])
            fit_payload = payload
        issues = verify_nonfinite_diagnostics(fit_payload, job["kind"], issues)
        refreshed.append(diagnostic_row(job, issues, origin=row.get("origin"),
                         pending_fits=pending_fit_details(fit_payload, job["kind"], issues)))
    report["jobs"] = refreshed
    actual_jobs = [{k: row[k] for k in JOB_FIELDS} for row in refreshed]
    complete = bool(complete and actual_jobs == jobs(a, name))
    update_report(report, complete)
    return report


def read_phase(root, name, a):
    report = verified_report(root, name, a)
    if not report["execution_complete"] or report["diagnostic_summary"]["blocking_issues"]:
        raise RuntimeError(f"{name} is incomplete or has blocking diagnostic issues: {root/'workflow'/f'{name}.json'}")
    if name in ["gamma_pilot", "pilot", "pilot_realdata"] and report["diagnostic_summary"]["deferred_failures"]:
        raise RuntimeError(f"{name} completed but needs MCMC repair before admitting the full experiment grid: {root/'workflow'/f'{name}.json'}")
    return report


def ensure_pilot(a, phase, root):
    if a.pilot_policy == "require":
        return read_phase(root, phase, a)
    path = root/"workflow"/f"{phase}.json"
    if path.is_file():
        try:
            report = verified_report(root, phase, a)
        except json.JSONDecodeError:
            report = None  # Rebuild a damaged report only from verified outputs.
        if report is not None and report["execution_complete"]:
            return read_phase(root, phase, a)
    with exclusive(a.pilot_dir):
        print(f"Checking saved {phase} outputs before automatic recovery.", flush=True)
        restored = restore_pilots(a, phases=[phase])["phases"][phase]
        if restored["invalid_cases"]:
            raise RuntimeError(f"{phase} has changed/corrupt outputs or incompatible settings. "
                               f"Inspect {a.pilot_dir/'workflow/pilot_restoration.json'}; automatic recovery preserves these files.")
        if not restored["missing_cases"]:
            return read_phase(root, phase, a)
        print(f"Automatic {phase}: {len(restored['missing_cases'])} cases need outputs; verified fit caches are reused.", flush=True)
        pilot_args = copy.copy(a)
        pilot_args.phase = phase
        if phase == "gamma_pilot":
            run_gamma_pilot(pilot_args)
        else:
            run_phase(pilot_args)
        return read_phase(root, phase, a)


def check_gates(a):
    if a.phase in ["collect", "plot"]:
        review = write_review(a, verify=True)
        if not review["ready_for_collection"]:
            state = ("experiment phases incomplete" if not review["experiments_complete"]
                     else f"{len(review['pending'])} unresolved issues")
            raise RuntimeError(f"Final diagnostic review required: {state}. "
                               f"Inspect {a.output_dir/'workflow/diagnostic_review.json'}; pending failures are not passes.")
        return
    if a.phase in ["pilot", "pilot_realdata", "synthetic", "benchmark", "cleaning", "collect", "plot"]:
        ensure_pilot(a, "gamma_pilot", a.pilot_dir/"gamma_checks")
    if a.phase in ["synthetic", "benchmark", "cleaning", "collect", "plot"]:
        ensure_pilot(a, "pilot", a.pilot_dir)
    if a.phase in ["benchmark", "cleaning", "collect", "plot"]:
        ensure_pilot(a, "pilot_realdata", a.pilot_dir)
    if a.phase == "cleaning":
        read_phase(a.output_dir, "benchmark", a)


def experiment_contract(a):
    return {"gammas": a.gammas, "datasets": a.datasets, "replicates": a.replicates,
            "real_replicates": a.real_replicates, "n": 1000, "n_test": 1000,
            "base_seed": 42, "mu_out": 6., "preprocessing": "kanamori", "split_mode": "fixed",
            "local_data_dir": a.local_data_dir}


def gamma_diagnostic_issues(summary):
    return [i for i in diagnostic_issues({"gamma": summary}, "regression")
            if i.get("method") == "GammaDiv"]


def restored_case(root, phase, job, a):
    """Validate existing completed outputs without sampling or changing them."""
    folder = root/job["relative_dir"]
    path = folder/"workflow_receipt.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed-case receipt: {path}")
    receipt = json.loads(path.read_text())
    if receipt.get("job") != job or not contracts_compatible(receipt.get("sampler_contract"), sampler_contract(a)):
        raise ValueError(f"Saved case settings or numerical sources differ: {folder}")
    metrics_name = "gamma_check.json" if phase == "gamma_pilot" else "metrics.json"
    outputs = receipt.get("output_sha256", {})
    draw_name = ("samples.npz" if job["kind"] in ["regression", "density"]
                 else "posterior_samples.npz" if job["kind"] == "benchmark" else "raw.npz")
    if metrics_name not in outputs or draw_name not in outputs:
        raise ValueError(f"Saved pilot receipt does not cover both metrics and draws: {folder}")
    for name, digest in outputs.items():
        candidate = folder/name
        if not candidate.is_file() or file_hash(candidate) != digest:
            raise ValueError(f"Saved pilot output missing or changed: {candidate}")
    payload = json.loads((folder/metrics_name).read_text())
    fit_payload = {"gamma": payload} if phase == "gamma_pilot" else payload
    issues = gamma_diagnostic_issues(payload) if phase == "gamma_pilot" else diagnostic_issues(payload, job["kind"])
    issues = verify_nonfinite_diagnostics(fit_payload, job["kind"], issues)
    return diagnostic_row(job, issues, origin=receipt.get("origin", "verified_existing_pilot"),
                          pending_fits=pending_fit_details(fit_payload, job["kind"], issues))


def restore_pilots(a, phases=None):
    """Rebuild only pilot reports, using checksummed outputs; never run MCMC."""
    current = sampler_contract(a)
    summary = {"inference_run": False, "phases": {}, "required_phases": []}
    for phase, root in [("gamma_pilot", a.pilot_dir/"gamma_checks"),
                        ("pilot", a.pilot_dir), ("pilot_realdata", a.pilot_dir)]:
        if phases is not None and phase not in phases:
            continue
        path = root/"workflow"/f"{phase}.json"
        rows, missing, invalid = [], [], []
        for job in jobs(a, phase):
            try:
                rows.append(restored_case(root, phase, job, a))
            except FileNotFoundError as exc:
                missing.append({**job, "reason": str(exc)})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                invalid.append({**job, "reason": str(exc)})
        report = {"phase": phase, "sampler_contract": current, "experiment_contract": experiment_contract(a),
                  "jobs": rows, "restored_from_verified_case_receipts": True,
                  "restoration_missing": missing, "restoration_invalid": invalid}
        update_report(report, complete=not missing and not invalid)
        if path.is_file():
            backup = path.with_name(path.name+".before_storage_recovery")
            if not backup.exists():
                atomic_bytes(backup, path.read_bytes())
        atomic_json(path, report)
        state = report["diagnostic_summary"]
        ready = bool(report["execution_complete"] and not state["blocking_issues"] and not state["deferred_failures"])
        summary["phases"][phase] = {"verified_cases": len(rows), "expected_cases": len(jobs(a, phase)),
            "missing_cases": missing, "invalid_cases": invalid, "admission_ready": ready,
            "report_path": str(path)}
        if not ready:
            summary["required_phases"].append(phase)
        print(f"{phase}: verified={len(rows)}/{len(jobs(a, phase))}, missing={len(missing)}, "
              f"invalid={len(invalid)}, admission_ready={ready}", flush=True)
    atomic_json(a.pilot_dir/"workflow/pilot_restoration.json", summary)
    if summary["required_phases"]:
        print("Pilot phases requiring attention, in order: "+", ".join(summary["required_phases"]), flush=True)
        print("Run those phases using the same roots/environment/settings; verified fit caches are reused. "
              "Missing or corrupt outputs are not admitted as passes.", flush=True)
    else:
        print("Requested pilot reports restored and admission checks passed: True", flush=True)
    return summary


def report_locations(a):
    return [(a.pilot_dir/"gamma_checks", "gamma_pilot"), (a.pilot_dir, "pilot"),
            (a.pilot_dir, "pilot_realdata"), *[(a.output_dir, phase)
             for phase in ["synthetic", "benchmark", "cleaning"]]]


def recovery_queue(pending):
    """One review task per method/case; suggestions are not automatic retries."""
    grouped = {}
    for item in pending:
        if not item.get("method") or not item.get("relative_dir"):
            continue
        key = (item["phase"], item["relative_dir"], item["method"])
        if key not in grouped:
            grouped[key] = {k: item.get(k) for k in ["phase", *JOB_FIELDS, "method", "condition_path",
                "fit_cache", "fit_contract_sha256", "sampler", "rhat_max", "ess_bulk_min", "ess_tail_min",
                "nuts_divergences", "retained_draws_per_chain", "additional_draws_per_chain"]}
            grouped[key].update(issue_types=[], severities=[])
        row = grouped[key]
        row["issue_types"].append(item["issue"])
        row["severities"].append(item["severity"])
        if item["issue"] == "nuts_divergences":
            row["nuts_divergences"] = item.get("value")
    queue = []
    for row in grouped.values():
        stats = [row["rhat_max"], row["ess_bulk_min"], row["ess_tail_min"]]
        finite = all(isinstance(v, (int, float)) and math.isfinite(v) for v in stats)
        if "blocking" in row["severities"]:
            action = "resolve_integrity_or_definition_error"
        elif ("nuts_divergences" in row["issue_types"] or not finite or stats[0] > POLICY["rhat_max"]):
            action = "inspect_chains_and_revalidate_sampler"
        elif all(s == "warning" for s in row["severities"]):
            action = "review_tail_precision_or_fixed_draw_extension"
        elif row["sampler"] == "frozen_mixture_independence_mh":
            action = "fixed_mh_draw_extension_candidate"
        else:
            action = "fixed_draw_extension_or_target_preserving_refit_candidate"
        row["suggested_action"] = action
        row["severity"] = ("blocking" if "blocking" in row["severities"] else
                           "needs_repair" if "needs_repair" in row["severities"] else "warning")
        row["metrics_path"] = str(Path(row["condition_path"])/
                                  ("gamma_check.json" if row["phase"] == "gamma_pilot" else "metrics.json"))
        downstream = ["rebuild_this_run_metrics_and_rankings"]
        if row["kind"] == "cleaning":
            downstream.append("rebuild_affected_cleaning_selections_and_downstream_fits")
        elif row["kind"] == "benchmark":
            downstream.append("check_matching_cleaning_runs_if_they_reuse_this_audit_fit")
        row["after_repair"] = downstream
        row.pop("severities")
        queue.append(row)
    return queue


def write_csv(path, columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def write_review(a, verify=True):
    """Gather pending cases; a review never silently waives a warning."""
    pending, phases = [], {}
    for root, phase in report_locations(a):
        path = root/"workflow"/f"{phase}.json"
        if not path.exists():
            phases[phase] = {"state": "missing", "execution_complete": False,
                             "expected_datasets": len(jobs(a, phase)), "saved_datasets": 0}
            continue
        try:
            if verify:
                report = verified_report(root, phase, a)
            else:
                report = json.loads(path.read_text())
                complete = report.get("execution_complete", report.get("all_diagnostics_pass", False))
                report["jobs"] = [diagnostic_row({k: r[k] for k in JOB_FIELDS}, r.get("diagnostic_issues", []),
                                                 pending_fits=r.get("pending_fits", {}))
                                  for r in report.get("jobs", [])]
                actual = [{k: r[k] for k in JOB_FIELDS} for r in report["jobs"]]
                update_report(report, bool(complete and actual == jobs(a, phase)))
            stats = report["diagnostic_summary"]
            phases[phase] = {**stats, "execution_complete": report["execution_complete"],
                            "all_diagnostics_pass": report["all_diagnostics_pass"],
                            "expected_datasets": len(jobs(a, phase)), "report_path": str(path)}
            for row in report["jobs"]:
                issues = [(i, "blocking") for i in row["blocking_issues"]]
                issues += [(i, "warning") for i in row["diagnostic_warnings"]]
                issues += [(i, "needs_repair") for i in row["diagnostic_failures"]]
                for issue, severity in issues:
                    values = issue.get("values", [None, None, None])
                    details = dict(row.get("pending_fits", {}).get(issue.get("method"), {}))
                    if issue.get("values"):
                        details.update(rhat_max=values[0], ess_bulk_min=values[1], ess_tail_min=values[2])
                    pending.append({"phase": phase, **{k: row[k] for k in JOB_FIELDS}, **issue,
                        "severity": severity, **details, "condition_path": str(root/row["relative_dir"]),
                        "report_path": str(path)})
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            phases[phase] = {"state": "invalid", "execution_complete": False, "report_path": str(path)}
            pending.append({"phase": phase, "issue": "report_validation_error", "severity": "blocking",
                            "message": str(exc), "report_path": str(path)})
    complete = all(p.get("execution_complete", False) for p in phases.values())
    report = {"diagnostic_policy": POLICY, "sampler_contract": sampler_contract(a),
              "experiment_contract": experiment_contract(a),
              "verification": "receipts_and_output_hashes" if verify else "saved_phase_reports",
              "phases": phases, "experiments_complete": complete, "pending": pending,
              "remaining_phases": [name for name, p in phases.items() if not p.get("execution_complete", False)],
              "warning_count": sum(i["severity"] == "warning" for i in pending),
              "deferred_failure_count": sum(i["severity"] == "needs_repair" for i in pending),
              "blocking_count": sum(i["severity"] == "blocking" for i in pending),
              "ready_for_collection": bool(verify and complete and not pending)}
    directory = a.output_dir/"workflow"
    atomic_json(directory/"diagnostic_review.json", report)
    columns = ["phase", "kind", "gamma", "eps", "rep", "dataset", "relative_dir", "method", "issue",
               "severity", "rhat_max", "ess_bulk_min", "ess_tail_min", "value", "nuts_divergences",
               "sampler", "fit_cache", "condition_path", "message"]
    path = directory/"diagnostic_pending.csv"
    write_csv(path, columns, pending)
    queue = recovery_queue(pending)
    atomic_json(directory/"mcmc_repair_queue.json", {"diagnostic_policy": POLICY,
                "verification": report["verification"], "automatic_sampling": False,
                "tasks": queue, "task_count": len(queue)})
    queue_rows = [{**r, "issue_types": ";".join(r["issue_types"]), "after_repair": ";".join(r["after_repair"])} for r in queue]
    write_csv(directory/"mcmc_repair_queue.csv", ["phase", *JOB_FIELDS, "method", "severity", "issue_types",
        "rhat_max", "ess_bulk_min", "ess_tail_min", "nuts_divergences", "sampler", "suggested_action",
        "fit_cache", "fit_contract_sha256", "metrics_path", "retained_draws_per_chain", "additional_draws_per_chain",
        "after_repair"], queue_rows)
    readiness = (f"ready_for_collection={report['ready_for_collection']}" if verify
                 else "verification_pending: run the review phase")
    print(f"Diagnostic review: warnings={report['warning_count']}, deferred_failures={report['deferred_failure_count']}, "
          f"blocking={report['blocking_count']}, "
          f"experiments_complete={complete}, {readiness}", flush=True)
    print(f"Pending cases: {path}", flush=True)
    if not complete:
        print("Incomplete phases: " + ", ".join(report["remaining_phases"]), flush=True)
    return report


def report_progress(row):
    for warning in row["diagnostic_warnings"]:
        print(f"WARNING (continuing): {warning['method']} tail ESS={warning['values'][2]:.3f} "
              f"is in [300, 400); pending final review.", flush=True)
    for issue in row["diagnostic_failures"]:
        print(f"MCMC NEEDS REPAIR (continuing): {issue.get('method')}, {issue['issue']}, "
              f"values={issue.get('values', issue.get('value'))}; saved for end-of-grid review.", flush=True)


def finish_phase(a, report, path):
    update_report(report, complete=True)
    atomic_json(path, report)
    if report["all_diagnostics_pass"]:
        print(f"All requested diagnostic checks passed: True ({len(report['jobs'])} datasets)", flush=True)
    else:
        print(f"Execution complete with {report['diagnostic_summary']['warning_methods']} warnings and "
              f"{report['diagnostic_summary']['deferred_failures']} deferred MCMC failures. "
              "All requested diagnostic checks passed: False; final review is pending.", flush=True)
    if a.phase in ["synthetic", "benchmark", "cleaning"]:
        write_review(a, verify=False)


@contextmanager
def exclusive(root):
    root = Path(root).resolve()
    path = root/"workflow.lock"
    if path in _HELD_WORKFLOW_LOCKS:
        yield
        return
    root.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        prior = json.loads(path.read_text())
        if prior.get("host") != socket.gethostname():
            raise RuntimeError(f"A workflow lock from another host exists: {path}")
        try:
            os.kill(prior["pid"], 0)
        except ProcessLookupError:
            path.unlink()
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        else:
            raise RuntimeError(f"Another workflow process is running: PID {prior['pid']}")
    with os.fdopen(fd, "w") as f:
        json.dump({"pid": os.getpid(), "host": socket.gethostname()}, f)
    _HELD_WORKFLOW_LOCKS.add(path)
    try:
        yield
    finally:
        _HELD_WORKFLOW_LOCKS.discard(path)
        path.unlink(missing_ok=True)


def gamma_inputs(a, job):
    """Recover exactly the data and fit seed used by the ordinary workflow."""
    import numpy as np
    from experiments.exp1_regression.utils_exp1_regression import simulate_regression_dataset, standardize_regression_data, regression_mcmc_var_names
    kind, eps, rep = job["kind"], job["eps"], job["rep"]
    source = a.existing_root/job["relative_dir"]
    if (a.reuse_pilot_dir/job["relative_dir"]/"metrics.json").is_file():
        source = a.reuse_pilot_dir/job["relative_dir"]
    if kind == "regression":
        seed = 42 + 1000*rep + 10000*int(round(eps*100))
        if a.proposal_root is not None and eps == .4:
            path = a.proposal_root/"inputs"/f"rep{rep:03d}"/"data.npz"
            meta = json.loads(path.with_suffix(".json").read_text())
            with np.load(path, allow_pickle=False) as f:
                raw = {k: f[k] for k in f.files}
            if {k: array_hash(v) for k, v in raw.items()} != meta["array_hashes"]:
                raise ValueError("GammaDiv proposal input hashes differ")
            expected = {"n": 1000, "n_test": 1000, "x_dim": 2, "eps": .4, "mu_out": 6.,
                        "contamination_mode": "intercept_shift", "seed": seed}
            if any(meta["settings"].get(k) != v for k, v in expected.items()):
                raise ValueError("GammaDiv recorded regression condition differs")
            X, y = raw["X"], raw["y"]
            origin = str(path)
        elif (source/"samples.npz").is_file():
            path = source/"samples.npz"
            meta = json.loads((source/"metrics.json").read_text())
            expected = {"n": 1000, "n_test": 1000, "x_dim": 2, "mu_out": 6.,
                        "contamination_mode": "intercept_shift", "base_seed": 42}
            if (meta.get("eps_true") != eps or meta.get("rep") != rep
                    or any(meta.get("run_config", {}).get(k) != v for k, v in expected.items())):
                raise ValueError("GammaDiv previous pilot regression condition differs")
            with np.load(path, allow_pickle=False) as f:
                X, y, _, _ = standardize_regression_data(f["x_train"], f["y_train"], f["x_test"])
            origin = str(path)
        else:
            raw = simulate_regression_dataset(n=1000, n_test=1000, eps=eps, mu_out=6., seed=seed,
                                              x_dim=2, contamination_mode="intercept_shift")
            X, y, _, _ = standardize_regression_data(raw["x_train"], raw["y_train"], raw["x_test"])
            origin = "generated_float32_with_original_seed_rule"
        inputs = [X, y]
        names = regression_mcmc_var_names(X.shape[1], False)
    else:
        from experiments.exp1_simulation.utils_exp1 import simulate_dataset
        seed = 42 + 1000*rep + 10*int(round(eps*10))
        if (source/"samples.npz").is_file():
            path = source/"samples.npz"
            meta = json.loads((source/"metrics.json").read_text())
            expected = {"n": 1000, "base_seed": 42, "scenarios": ["A"], "out_dist": "gaussian",
                        "out_mean_x": 5., "out_mean_y": 5., "out_scale_x": 1., "out_scale_y": 1.}
            if (meta.get("eps_true") != eps or meta.get("rep") != rep or meta.get("scenario") != "A"
                    or any(meta.get("run_config", {}).get(k) != v for k, v in expected.items())):
                raise ValueError("GammaDiv previous pilot density condition differs")
            with np.load(path, allow_pickle=False) as f:
                x = f["x"]
            origin = str(path)
        else:
            x, _ = simulate_dataset(1000, eps, "A", seed, out_dist="gaussian",
                                    out_mean=(5., 5.), out_scale=(1., 1.))
            origin = "generated_float32_with_original_seed_rule"
        inputs = [x]
        names = [f"mu_{i+1}" for i in range(x.shape[1])] + [f"log_scale_{i+1}" for i in range(x.shape[1])]
    return inputs, seed+202, names, origin


def run_gamma_pilot(a):
    """Check GammaDiv alone before reassembling the six-method pilot results."""
    import jax
    import jax.numpy as jnp
    from hbayes.gamma_posterior import logprob_regression, logprob_density
    from hbayes.gamma_sampling import fit_gamma
    from hbayes.posteriors import MVNPriorsConfig, MVNGibbsConfig
    root = a.pilot_dir/"gamma_checks"
    root.mkdir(parents=True, exist_ok=True)
    cfg = {"sampler_contract": sampler_contract(a), "experiment_contract": experiment_contract(a)}
    config_path = root/"workflow_config.json"
    write_config(config_path, cfg)
    report = {"phase": "gamma_pilot", **cfg, "jobs": [], "all_diagnostics_pass": False}
    update_report(report)
    report_path = root/"workflow/gamma_pilot.json"
    for job in jobs(a, "gamma_pilot"):
        print(f"gamma_pilot: {job['relative_dir']}", flush=True)
        folder = root/job["relative_dir"]
        try:
            receipt_path = folder/"workflow_receipt.json"
            if receipt_path.is_file():
                restored_case(root, "gamma_pilot", job, a)
                fitted = summary = json.loads((folder/"gamma_check.json").read_text())
                origin = summary.get("input_origin", "verified_saved_gamma_check")
                receipt = json.loads(receipt_path.read_text())
                migrate_receipt(receipt_path, receipt, cfg["sampler_contract"])
                print("Reuse verified gamma pilot: "+str(folder), flush=True)
            else:
                inputs, seed, names, origin = gamma_inputs(a, job)
                with jax.experimental.enable_x64():
                    priors = MVNPriorsConfig()
                    config = MVNGibbsConfig(gamma=job["gamma"], eta=1., priors=priors)
                    arrays = [jnp.asarray(v, dtype=jnp.float64) for v in inputs]
                    target = (logprob_regression(*arrays, config) if job["kind"] == "regression"
                              else logprob_density(arrays[0], config))
                    fitted = fit_gamma(target, inputs, method="GammaDiv", model=job["kind"], seed=seed,
                        chains=a.chains, num_warmup=a.warmup, num_samples=a.samples,
                        target_accept=a.target_accept, init_step=a.init_step, priors=priors,
                        eta=1., gamma=job["gamma"], names=names)
                summary = {k: v for k, v in fitted.items() if k not in ["samples", "samples_chains"]}
                summary.update(input_origin=origin, input_hashes=[array_hash(v) for v in inputs])
                atomic_json(folder/"gamma_check.json", summary)
                atomic_npz(folder/"samples.npz", samples_gamma=fitted["samples_chains"])
                outputs = {name: file_hash(folder/name) for name in ["gamma_check.json", "samples.npz"]}
                atomic_json(folder/"workflow_receipt.json", {"sampler_contract": cfg["sampler_contract"],
                            "job": job, "origin": "gamma_div_only_check", "output_sha256": outputs})
            issues = gamma_diagnostic_issues(summary)
            issues = verify_nonfinite_diagnostics({"gamma": summary}, job["kind"], issues)
            row = diagnostic_row(job, issues, origin=origin,
                                 pending_fits=pending_fit_details({"gamma": summary}, job["kind"], issues),
                                 **{n: fitted[n] for n in ["rhat_max", "ess_bulk_min", "ess_tail_min", "mode_count"] if n in fitted})
            report["jobs"].append(row)
        except Exception as exc:
            report["jobs"].append(diagnostic_row(job, [{"issue": "execution_error", "type": type(exc).__name__, "message": str(exc)}]))
            update_report(report)
            save_error_report(atomic_json, report_path, report, exc)
            raise
        update_report(report)
        atomic_json(report_path, report)
        report_progress(row)
        if row["blocking_issues"]:
            raise RuntimeError(f"GammaDiv diagnostic gate failed: {folder}; inspect {report_path}")
        jax.clear_caches()
    finish_phase(a, report, report_path)


def _run_job(a, job, output):
    import jax
    kind, g, e, r = (job[n] for n in ["kind", "gamma", "eps", "rep"])
    opts = {"chains": a.chains, "num_warmup": a.warmup, "num_samples": a.samples,
            "target_accept": a.target_accept, "init_step": a.init_step, "holder_phi": None}
    if kind == "regression":
        from experiments.exp1_regression import run_exp1_regression_eps_sweep as run
        if a.proposal_root and e == .4:
            import numpy as np
            path = a.proposal_root/"inputs"/f"rep{r:03d}"/"data.npz"
            meta = json.loads(path.with_suffix(".json").read_text())
            with np.load(path, allow_pickle=False) as data:
                recorded = {k: data[k] for k in data.files}
            if {k: array_hash(v) for k, v in recorded.items()} != meta["array_hashes"]:
                raise ValueError("Recorded experiment input hashes differ")
            expected = {"n": 1000, "n_test": 1000, "x_dim": 2, "eps": .4, "mu_out": 6.,
                        "contamination_mode": "intercept_shift", "seed": 400042+1000*r}
            if any(meta["settings"].get(k) != v for k, v in expected.items()):
                raise ValueError("Recorded data generation settings differ")
            CONTEXT["recorded_inputs"] = recorded
        elif CONTEXT.get("legacy_folder"):
            # Saved observations keep comparisons paired even if a new JAX
            # version generates different random values for the same seed.
            import numpy as np
            path = Path(CONTEXT["legacy_folder"])/"samples.npz"
            if path.is_file():
                meta = CONTEXT["legacy_payload"]
                expected = {"n": 1000, "n_test": 1000, "x_dim": 2,
                            "mu_out": 6., "contamination_mode": "intercept_shift", "base_seed": 42}
                if any(meta.get("run_config", {}).get(k) != v for k, v in expected.items()):
                    raise ValueError("Legacy regression data-generation settings differ")
                if meta.get("eps_true") != e or meta.get("rep") != r:
                    raise ValueError("Legacy regression data condition/replicate differs")
                with np.load(path, allow_pickle=False) as data:
                    required = ["x_train", "y_train", "x_test", "y_test", "mask_outlier", "beta_true", "sigma_true"]
                    if all(k in data.files for k in required):
                        recorded = {k: data[k] for k in required}
                        X, y, X_test, scaler = run.standardize_regression_data(
                            recorded["x_train"], recorded["y_train"], recorded["x_test"])
                        recorded.update(X=X, y=y, X_test=X_test,
                                        **{f"scaler_{k}": v for k, v in scaler.items()})
                        CONTEXT["recorded_inputs"] = recorded
        cfg = run.SweepConfig(**opts, gamma=g, output_dir=str(output/Path(job["relative_dir"]).parts[0]), n=1000, n_test=1000)
        result = run.run_single_replicate(cfg, e, r)
        run.save_single_run_results(cfg.output_dir, e, r, cfg, result, save_samples=True)
    elif kind == "density":
        from experiments.exp1_simulation import run_exp1 as run
        if CONTEXT.get("legacy_folder"):
            import numpy as np
            path = Path(CONTEXT["legacy_folder"])/"samples.npz"
            if path.is_file():
                meta = CONTEXT["legacy_payload"]
                expected = {"n": 1000, "base_seed": 42, "scenarios": ["A"],
                            "out_dist": "gaussian", "out_mean_x": 5., "out_mean_y": 5.,
                            "out_scale_x": 1., "out_scale_y": 1.}
                if any(meta.get("run_config", {}).get(k) != v for k, v in expected.items()):
                    raise ValueError("Legacy density data-generation settings differ")
                if meta.get("eps_true") != e or meta.get("rep") != r or meta.get("scenario") != "A":
                    raise ValueError("Legacy density data condition/replicate differs")
                with np.load(path, allow_pickle=False) as data:
                    if all(k in data.files for k in ["x", "y_true"]):
                        CONTEXT["recorded_density_inputs"] = {k: data[k] for k in ["x", "y_true"]}
        cfg = run.RunConfig(**opts, gamma=g, output_dir=str(output/Path(job["relative_dir"]).parts[0]), n=1000,
                            out_dist="gaussian", out_mean_x=5., out_mean_y=5., out_scale_x=1., out_scale_y=1., scenarios=["A"])
        result = run.run_single_replicate(cfg, "A", e, r)
        run.save_single_run_results(cfg.output_dir, "A", e, r, cfg, result, save_samples=True)
    else:
        from experiments.exp1_regression import run_exp1_realdata_benchmark as benchmark
        dataset_cache = a.cache_dir/"datasets"/f"{job['dataset']}.npz"
        data_receipt = dataset_cache.with_suffix(".json")
        import numpy as np
        if dataset_cache.exists():
            record = json.loads(data_receipt.read_text())
            if record["sha256"] != file_hash(dataset_cache) or record["local_data_dir"] != a.local_data_dir:
                raise ValueError("Dataset cache changed or uses a different data source")
            with np.load(dataset_cache, allow_pickle=False) as data:
                x, y = data["X"], data["y"]
        else:
            x, y = benchmark.load_benchmark_dataset(job["dataset"], a.local_data_dir)
            atomic_npz(dataset_cache, X=x, y=y)
            atomic_json(data_receipt, {"dataset": job["dataset"], "local_data_dir": a.local_data_dir,
                                      "sha256": file_hash(dataset_cache)})
        common = {**opts, "train_ratio": .8, "split_sizes": "ratio", "preprocessing": "kanamori", "split_mode": "fixed",
                  "local_data_dir": a.local_data_dir, "output_dir": str(output/Path(job["relative_dir"]).parts[0])}
        mu = 0. if e == 0 else 6.
        if kind == "benchmark":
            cfg = benchmark.RealDataConfig(**common, save_samples=True)
            result = benchmark.run_single_setting(cfg, job["dataset"], g, e, mu, r, x, y)
            benchmark.save_single_run_results(cfg, result)
        else:
            from experiments.exp1_regression import run_exp1_realdata_cleaning as run
            cfg = run.CleaningConfig(**common, save_raw=True)
            result, raw = run.run_single_cleaning_run(cfg, job["dataset"], g, e, mu, r, x, y)
            run._save_run_json(cfg, result, job["dataset"], g, e, mu, r, raw_bundle=raw)
    jax.clear_caches()
    return json.loads((output/job["relative_dir"]/"metrics.json").read_text())


def run_phase(a):
    check_gates(a)
    output = a.pilot_dir if a.phase.startswith("pilot") else a.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cfg = {"sampler_contract": sampler_contract(a), "experiment_contract": experiment_contract(a)}
    cfg_path = output/"workflow_config.json"
    write_config(cfg_path, cfg)
    report = {"phase": a.phase, **cfg, "jobs": [], "all_diagnostics_pass": False}
    update_report(report)
    report_path = output/"workflow"/f"{a.phase}.json"
    groups = forced_conditions(a)
    for job in jobs(a):
        folder = output/job["relative_dir"]
        receipt_path = folder/"workflow_receipt.json"
        source = a.existing_root/job["relative_dir"]
        pilot_source = a.reuse_pilot_dir/job["relative_dir"]
        if (pilot_source/"metrics.json").is_file():
            source = pilot_source
        print(f"{a.phase}: {job['relative_dir']}", flush=True)
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if not contracts_compatible(receipt.get("sampler_contract"), cfg["sampler_contract"]) or receipt["job"] != job:
                raise ValueError("Run receipt differs; choose a new output directory")
            for name, digest in receipt["output_sha256"].items():
                if file_hash(folder/name) != digest:
                    raise ValueError(f"Run output hash differs: {folder/name}")
            payload = json.loads((folder/"metrics.json").read_text())
            origin = receipt["origin"]
            migrate_receipt(receipt_path, receipt, cfg["sampler_contract"])
        else:
            old = json.loads((source/"metrics.json").read_text()) if (source/"metrics.json").exists() else None
            group = (job["kind"], job["gamma"], job["eps"], job["dataset"])
            # Entire unaffected density groups can reuse metric-only results.
            # Regression H-B always changes its support and must use new draws.
            if (job["kind"] == "density" and old is not None
                    and old.get("gamma", {}).get("gamma_posterior_definition") == GAMMA_DEFINITION and not groups.get(group)
                    and (job["rep"] != 0 or (source/"samples.npz").is_file())
                    and not diagnostic_issues(old, "density")):
                folder.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source/"metrics.json", folder/"metrics.json")
                if (source/"samples.npz").exists():
                    shutil.copy2(source/"samples.npz", folder/"samples.npz")
                payload, origin = old, "unchanged_density_group"
            else:
                data_seed = 42+1000*job["rep"]+10000*int(round(job["eps"]*100)) if job["kind"] == "regression" else None
                with job_context(**job, data_seed=data_seed, base_seed=42, eta_gamma_div=1.,
                                 legacy_folder=str(source) if old else None,
                                 legacy_payload=old, force_methods=sorted(groups.get(group, []))):
                    try:
                        payload = _run_job(a, job, output)
                    except Exception as exc:
                        report["jobs"].append(diagnostic_row(job, [{"issue": "execution_error",
                            "type": type(exc).__name__, "message": str(exc)}], origin="interrupted_or_failed_fit"))
                        update_report(report)
                        save_error_report(atomic_json, report_path, report, exc)
                        raise
                origin = "new_or_verified_cached_fits"
            outputs = {p.name: file_hash(p) for p in folder.iterdir()
                       if p.is_file() and p.name != "workflow_receipt.json" and not p.name.endswith(".tmp")}
            atomic_json(receipt_path, {"sampler_contract": cfg["sampler_contract"], "job": job,
                                      "origin": origin, "output_sha256": outputs})
        issues = verify_nonfinite_diagnostics(payload, job["kind"], diagnostic_issues(payload, job["kind"]))
        row = diagnostic_row(job, issues, origin=origin, pending_fits=pending_fit_details(payload, job["kind"], issues))
        report["jobs"].append(row)
        update_report(report)
        atomic_json(report_path, report)
        report_progress(row)
        if row["blocking_issues"]:
            write_review(a, verify=False)
            raise RuntimeError(f"Integrity/definition gate failed at {folder}. Results/checkpoints are saved; inspect {report_path}.")
    finish_phase(a, report, report_path)


def collect(a):
    check_gates(a)
    from hb_experiments.collection import collect as collect_metrics
    files = collect_metrics(a)
    atomic_json(a.output_dir/"workflow/collect.json", {"complete": True, "sampler_contract": sampler_contract(a),
                "experiment_contract": experiment_contract(a),
                "csv_sha256": {name: file_hash(path) for name, path in files.items()}})
    print("Collection complete; mean/std/count rebuilt from per-run JSON.")


def preflight(a):
    """Check required project modules and reuse files without running inference."""
    import importlib
    import numpy as np
    from hbayes.paper_sampling import runtime, target_sources
    required = {
        "hbayes.simulate": ["ContamConfig", "simulate_contaminated_gaussian"],
        "experiments.exp1_regression.run_exp1_realdata_benchmark": ["load_benchmark_dataset"],
        "experiments.exp1_regression.run_exp1_realdata_cleaning": ["run_single_cleaning_run"],
        "experiments.plot_exp1_paper_suite_section6_revised": ["main"],
    }
    for module, names in required.items():
        try:
            loaded = importlib.import_module(module)
        except ImportError as exc:
            raise RuntimeError(f"Required existing-project module/dependency is missing: {module}. "
                               'Install the source checkout with python -m pip install -e ".[experiments]".') from exc
        for name in names:
            if not hasattr(loaded, name):
                raise RuntimeError(f"Required existing-project function is missing: {module}.{name}")
    reused = 0
    if a.mh30_root:
        if a.proposal_root is None:
            raise RuntimeError("Set PROPOSAL_ROOT to the original all30 proposals/ directory containing inputs/.")
        if a.chains != 4:
            raise ValueError("Completed all30 reuse requires CHAINS=4")
        for rep in range(30):
            folder = a.mh30_root/"posterior"/f"rep{rep:03d}"
            if not folder.is_dir():
                folder = a.mh30_root/f"rep{rep:03d}"
            summary = json.loads((folder/"summary.json").read_text())
            receipt = json.loads((folder/"completion.json").read_text())
            for name, digest in receipt["output_sha256"].items():
                if file_hash(folder/name) != digest:
                    raise ValueError(f"Completed all30 output changed: {folder/name}")
            for name, digest in target_sources().items():
                if summary.get("original_source_sha256", {}).get("hbayes."+name[:-3]) != digest:
                    raise ValueError(f"Completed all30 target source differs: {name}")
            if not summary.get("diagnostics_pass") or summary["gamma"] != 1. or summary["sigma_min"] != .05:
                raise ValueError(f"Completed all30 settings/diagnostics differ: {folder}")
            with np.load(folder/"samples.npz", allow_pickle=False) as data:
                if data["samples_physical"].shape != (4, a.mh_samples, 5):
                    raise ValueError("Completed all30 reuse requires the saved MH draw count (normally MH_SAMPLES=4000)")
            path = a.proposal_root/"inputs"/f"rep{rep:03d}"/"data.npz"
            meta = json.loads(path.with_suffix(".json").read_text())
            with np.load(path, allow_pickle=False) as data:
                hashes = {k: array_hash(data[k]) for k in data.files}
            if hashes != meta["array_hashes"]:
                raise ValueError(f"Original all30 observations changed: {path}")
            for name in ["X", "y"]:
                if hashes[name] != summary["input_array_hashes"][name]:
                    raise ValueError(f"Original observations do not match completed all30: {path}")
            reused += 1
    print(json.dumps({"runtime": runtime(), "sampler_contract": sampler_contract(a),
                      "existing_results_found": a.existing_root.is_dir(),
                      "completed_all30_files_checked": reused,
                      "proposal_root": str(a.proposal_root),
                      "note": "File/runtime checks only; run both pilots to verify new diagnostics."}, indent=2))
    print("Preflight checks passed: True", flush=True)


def main(argv=None):
    a = parse(argv)
    if a.dry_run:
        print(json.dumps({"phase": a.phase, "jobs": jobs(a), "sampler_contract": sampler_contract(a),
                          "diagnostic_policy": POLICY,
                          "existing_root": str(a.existing_root), "mh30_root": str(a.mh30_root),
                          "proposal_root": str(a.proposal_root),
                          "note": "No inference, data download, or output mutation performed."}, indent=2))
        return
    install_storage_safety()
    if a.phase in ["gamma_pilot", "pilot", "pilot_realdata", "synthetic", "benchmark", "cleaning"]:
        ensure_capacity(a.pilot_dir if "pilot" in a.phase else a.output_dir)
        ensure_capacity(a.cache_dir)
    os.environ["HBAYES_CACHE_ROOT"] = str(a.cache_dir)
    os.environ["HBAYES_MH_WARMUP"] = str(a.mh_warmup)
    os.environ["HBAYES_MH_SAMPLES"] = str(a.mh_samples)
    if a.mh30_root:
        if not a.mh30_root.is_dir():
            raise FileNotFoundError(a.mh30_root)
        os.environ["HBAYES_MH30_ROOT"] = str(a.mh30_root)
    else:
        os.environ.pop("HBAYES_MH30_ROOT", None)
    import jax
    jax.config.update("jax_enable_x64", False)
    if a.phase == "preflight":
        preflight(a)
        return
    if a.phase == "restore_pilots":
        with exclusive(a.pilot_dir):
            restore_pilots(a)
        return
    if a.phase == "status":
        for root, phase in report_locations(a):
            path = root/"workflow"/f"{phase}.json"
            report = json.loads(path.read_text()) if path.exists() else {}
            summary = report.get("diagnostic_summary", {})
            print(f"{phase}: saved={len(report.get('jobs', []))}, "
                  f"execution_complete={report.get('execution_complete', report.get('all_diagnostics_pass', False))}, "
                  f"pass={report.get('all_diagnostics_pass', False)}, "
                  f"warnings={summary.get('warning_methods', 'run review')}, "
                  f"deferred_failures={summary.get('deferred_failures', 'run review')}")
        return
    if a.phase == "review":
        with exclusive(a.output_dir):
            write_review(a, verify=True)
        return
    if a.phase == "gamma_pilot":
        with exclusive(a.pilot_dir):
            run_gamma_pilot(a)
        return
    if a.phase == "collect":
        with exclusive(a.output_dir):
            collect(a)
        return
    if a.phase == "plot":
        with exclusive(a.output_dir):
            run_plot(a)
        return
    with exclusive(a.pilot_dir if a.phase.startswith("pilot") else a.output_dir):
        run_phase(a)


def run_plot(a):
    check_gates(a)
    report = json.loads((a.output_dir/"workflow/collect.json").read_text())
    if not report["complete"] or not contracts_compatible(report["sampler_contract"], sampler_contract(a)):
        raise ValueError("Run collect with these settings before plot")
    for name, digest in report["csv_sha256"].items():
        if file_hash(a.output_dir/name) != digest:
            raise ValueError("Collected CSV changed; rerun collect")
    from experiments import plot_exp1_paper_suite_section6_revised as plot
    sys.argv = ["plot", "--outputs_root", str(a.output_dir), "--output_dir", str(a.output_dir/"paper_exp1_suite_section6_revised"),
                "--simulation_main_gamma", "0.5", "--regression_main_gamma", "1.0", "--simulation_scenario", "A",
                "--regression_condition", "mu6", "--posterior_eps", "0.2", "--rep", "0"]
    plot.main()


if __name__ == "__main__":
    main()
