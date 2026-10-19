"""Replot appendix gamma grids from saved aggregate CSVs; never runs inference."""
from pathlib import Path
if __package__:
    from .figure_export import save_figure
else:
    from figure_export import save_figure

import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STYLES = {
    "DPD": ("H-B (DPD)", "#4c78a8", "o"),
    "ExpSatDPD": ("H-B (ExpSat)", "#e45756", "X"),
    "RatSatDPD": ("H-B (RatSat)", "#72b7b2", "P"),
    "NLL": ("Standard Bayes", "#2ca02c", "s"),
    "GammaDiv": (r"$\gamma$-divergence", "#f58518", "^"),
    "StudentT": (r"Student-$t$", "#b279a2", "D"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_gamma_sensitivity(results_root, figure_dir):
    results_root, figure_dir = Path(results_root), Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    receipt = {}
    with plt.rc_context({"font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "xtick.labelsize": 11,
                         "ytick.labelsize": 11}):
        for kind in ("density", "regression"):
            prefix = "simulation" if kind == "density" else "regression"
            error = "mu_bias_norm" if kind == "density" else "rmse_test"
            specs = [(error, "Location error" if kind == "density" else "Clean-test RMSE"),
                     ("eps_post_mean", r"Posterior mean $\epsilon$"),
                     ("auprc", "AUPRC")]
            for metric, ylabel in specs:
                output = figure_dir / f"{prefix}_{metric}_gamma_grid.png"
                fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.25))
                fig.subplots_adjust(left=.075, right=.99, bottom=.20, top=.73, wspace=.30)
                points, source_hashes = [], {}
                for ax, gamma, tag in zip(axes, (.1, .5, 1.), ("0p1", "0p5", "1p0")):
                    directory = (f"exp1_simulation_gamma_sweep_gamma{tag}" if kind == "density"
                                 else f"exp1_regression_eps_sweep_mu6_gamma{tag}")
                    rel = f"{directory}/summary_agg.csv"
                    df = pd.read_csv(results_root / rel)
                    if kind == "density":
                        assert set(df.scenario) == {"A"}
                    source_hashes[rel] = sha(results_root / rel)
                    for method, (label, color, marker) in STYLES.items():
                        if metric == "eps_post_mean" and method not in ("DPD", "ExpSatDPD", "RatSatDPD"):
                            continue
                        rows = df[df.method == method].sort_values("eps_true")
                        if metric == "auprc":
                            rows = rows[rows.eps_true > 0]
                        mean, sd = rows[f"{metric}_mean"], rows[f"{metric}_std"]
                        assert np.isfinite(mean).all() and np.isfinite(sd).all()
                        assert (rows[f"{metric}_count"] == 30).all()
                        ax.errorbar(rows.eps_true, mean, yerr=sd, color=color, marker=marker,
                                    ms=4, lw=1.4, capsize=2.5, label=label)
                        for _, row in rows.iterrows():
                            points.append(dict(kind=kind, gamma=gamma, method=method,
                                               epsilon=float(row.eps_true), metric=metric,
                                               mean=float(row[f"{metric}_mean"]),
                                               sd=float(row[f"{metric}_std"]), source_csv=rel))
                    if metric == "eps_post_mean":
                        ax.plot([0, .4], [0, .4], color="black", ls="--", lw=1.1,
                                label=r"True $\epsilon_0$")
                        ax.set_ylim(-.025, .45)
                    ax.set_title(rf"$\gamma={gamma:g}$", fontsize=14, pad=7)
                    ax.set_xlabel(r"True contamination $\epsilon_0$", fontsize=11)
                    ax.grid(alpha=.2)
                axes[0].set_ylabel(ylabel, fontsize=12)
                handles, labels = axes[0].get_legend_handles_labels()
                legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, .995),
                                    ncol=4 if metric == "eps_post_mean" else 6,
                                    frameon=False, fontsize=10.5, columnspacing=1.1,
                                    handlelength=1.6, handletextpad=.4)
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                legend_box = legend.get_window_extent(renderer)
                assert legend_box.x0 >= 0 and legend_box.x1 <= fig.bbox.x1
                assert all(not legend_box.overlaps(ax.title.get_window_extent(renderer)) for ax in axes)
                title_boxes = [ax.title.get_window_extent(renderer) for ax in axes]
                assert all(not a.overlaps(b) for a, b in zip(title_boxes, title_boxes[1:]))
                save_figure(fig, output, dpi=260)
                plt.close(fig)
                receipt[output.name] = dict(output_sha256=sha(output), pdf_sha256=sha(output.with_suffix('.pdf')),
                    png_dpi=600, panel_labels='uppercase', values_changed=False,
                    layout_regenerated_from_verified_csv=True, points=points,
                    source_csv_sha256=source_hashes, legend_and_titles_do_not_overlap=True)
    return receipt


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=root / "results")
    parser.add_argument("--figure-dir", type=Path, default=root / "fig")
    args = parser.parse_args()
    receipt = render_gamma_sensitivity(args.results_root, args.figure_dir)
    path = args.figure_dir / "gamma_plot_receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Replotted {len(receipt)} figures from saved CSVs. No MCMC. Receipt: {path}")


if __name__ == "__main__":
    main()
