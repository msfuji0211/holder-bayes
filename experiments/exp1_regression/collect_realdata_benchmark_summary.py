"""
Rebuild summary_metrics.csv (and summary_agg.csv) from per-run metrics.json files.

Use after a partial run, a resume, or SPLIT_BY_DATASET jobs that share one output_dir.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

METHOD_JSON_KEYS = {"DPD": "dpd", "ExpSatDPD": "exp_sat_dpd", "RatSatDPD": "rat_sat_dpd",
                   "NLL": "nll", "GammaDiv": "gamma", "StudentT": "student_t"}

JSON_KEY_TO_METHOD = {v: k for k, v in METHOD_JSON_KEYS.items()}

_RUN_DIR_RE = re.compile(
    r"dataset_(?P<dataset>.+)/gamma_(?P<gamma>[0-9.]+)/eps_(?P<eps>[0-9.]+)/muout_(?P<muout>[0-9.]+)/rep_(?P<rep>[0-9]+)$"
)


def _flatten_block(block: dict) -> dict:
    """Drop nested diagnostics; keep scalars for CSV."""
    out = {}
    for k, v in block.items():
        if k == "mcmc_diagnostics_by_param":
            continue
        if isinstance(v, (dict, list)):
            continue
        out[k] = v
    return out


def collect_rows(output_dir: str) -> list[dict]:
    root = Path(output_dir)
    rows: list[dict] = []
    for path in sorted(root.glob("dataset_*/gamma_*/eps_*/muout_*/rep_*/metrics.json")):
        rel = path.parent.relative_to(root).as_posix()
        m = _RUN_DIR_RE.match(rel)
        if not m:
            continue
        with open(path) as f:
            payload = json.load(f)

        gamma_raw = payload.get("gamma_hyperparam", payload.get("gamma", m.group("gamma")))
        if isinstance(gamma_raw, dict):
            gamma_val = float(m.group("gamma"))
        else:
            gamma_val = float(gamma_raw)

        base = {
            "dataset": payload.get("dataset", m.group("dataset")),
            "gamma": gamma_val,
            "eps_true": float(payload.get("eps_true", m.group("eps"))),
            "mu_out": float(payload.get("mu_out", m.group("muout"))),
            "replicate": int(payload.get("replicate", m.group("rep"))),
            "scenario": payload.get(
                "scenario",
                "original" if float(payload.get("eps_true", 0)) == 0.0 else "synthetic_contamination",
            ),
            "k_oracle": payload.get("k_oracle"),
            "k_est_primary": payload.get("k_est_primary"),
            "preprocessing": payload.get("preprocessing"),
            "split_mode": payload.get("split_mode"),
            "temperature_version": payload.get("temperature_version", "legacy_unversioned"),
        }

        for json_key, method in JSON_KEY_TO_METHOD.items():
            if json_key not in payload:
                continue
            block = payload[json_key]
            if not isinstance(block, dict):
                continue
            row = {**base, "method": method, **_flatten_block(block)}
            rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Collect benchmark metrics.json into summary CSVs")
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/exp1_realdata_benchmark_gamma_sweep",
    )
    p.add_argument("--expected_runs", type=int)
    args = p.parse_args()

    rows = collect_rows(args.output_dir)
    if not rows:
        raise SystemExit(f"No metrics.json found under {args.output_dir}")

    df = pd.DataFrame(rows)
    identities = df[["dataset", "gamma", "eps_true", "mu_out", "replicate"]].drop_duplicates()
    if args.expected_runs is not None and len(identities) != args.expected_runs:
        raise SystemExit(f"Incomplete results: {len(identities)}/{args.expected_runs} runs")
    if df.duplicated(["dataset", "gamma", "eps_true", "mu_out", "replicate", "method"]).any():
        raise SystemExit("Duplicate benchmark rows")
    if df["temperature_version"].nunique() != 1:
        raise SystemExit("Mixed temperature versions in output_dir")
    out = Path(args.output_dir)
    summary_path = out / "summary_metrics.csv"
    df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path} ({len(df)} rows, {len(rows) // max(len(JSON_KEY_TO_METHOD), 1)} runs approx.)")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    group_keys = ["dataset", "gamma", "scenario", "eps_true", "mu_out", "method"]
    numeric_cols = [c for c in numeric_cols if c not in group_keys and c != "replicate"]
    if numeric_cols:
        df_agg = df.groupby(group_keys)[numeric_cols].agg(["mean", "std"]).reset_index()
        df_agg.columns = [
            "_".join(col).strip() if col[1] else col[0] for col in df_agg.columns.values
        ]
        agg_path = out / "summary_agg.csv"
        df_agg.to_csv(agg_path, index=False)
        print(f"Wrote {agg_path}")


if __name__ == "__main__":
    main()
