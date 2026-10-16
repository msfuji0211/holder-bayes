import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.exp1_regression.run_exp1_regression import (
    generate_summary_plots as generate_condition_summary_plots,
)
from experiments.exp1_regression.run_exp1_regression_eps_sweep import (
    generate_summary_plots as generate_eps_summary_plots,
)
from experiments.exp1_regression.utils_exp1_regression import (
    generate_representative_plots_from_raw,
)


def _detect_mode(df_summary: pd.DataFrame, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    if "condition" in df_summary.columns:
        return "conditions"
    if "eps_true" in df_summary.columns:
        return "eps_sweep"
    raise ValueError("Could not infer mode from summary_agg.csv")


def _plot_condition_mode(input_dir: Path) -> None:
    for condition_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        sample_path = condition_dir / "rep000" / "samples.npz"
        if not sample_path.exists():
            continue
        raw = np.load(sample_path, allow_pickle=True)
        generate_representative_plots_from_raw(
            raw,
            condition=condition_dir.name,
            output_dir=str(input_dir),
        )


def _plot_eps_sweep_mode(input_dir: Path, mu_out: float | None) -> None:
    for eps_dir in sorted(p for p in input_dir.iterdir() if p.is_dir() and p.name.startswith("eps")):
        sample_path = eps_dir / "rep000" / "samples.npz"
        if not sample_path.exists():
            continue
        eps_label = eps_dir.name.replace("eps", "")
        mu_label = f"mu{int(mu_out)}" if mu_out is not None and float(mu_out).is_integer() else f"mu{mu_out}"
        raw = np.load(sample_path, allow_pickle=True)
        generate_representative_plots_from_raw(
            raw,
            condition=f"{mu_label}_eps{eps_label}",
            output_dir=str(input_dir),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate plots from saved exp1_regression results")
    parser.add_argument("--input_dir", required=True, help="Result directory containing summary_agg.csv")
    parser.add_argument(
        "--mode",
        choices=["auto", "conditions", "eps_sweep"],
        default="auto",
        help="Result layout. Auto-detected by default.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    summary_path = input_dir / "summary_agg.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary_agg.csv not found under {input_dir}")

    df_summary = pd.read_csv(summary_path)
    mode = _detect_mode(df_summary, args.mode)

    if mode == "conditions":
        generate_condition_summary_plots(df_summary, str(input_dir))
        _plot_condition_mode(input_dir)
    elif mode == "eps_sweep":
        generate_eps_summary_plots(df_summary, str(input_dir))
        mu_out = None
        if "mu_out" in df_summary.columns and not df_summary["mu_out"].empty:
            mu_out = float(df_summary["mu_out"].iloc[0])
        _plot_eps_sweep_mode(input_dir, mu_out=mu_out)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"Regenerated plots in {input_dir}")


if __name__ == "__main__":
    main()
