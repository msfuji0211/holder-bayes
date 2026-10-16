import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.plot_exp1_paper_figures import (
    discover_gamma_dirs,
    parse_float_list,
    select_gammas,
)
from experiments.exp1_regression.utils_exp1_regression import (
    recover_posterior_params_original_scale,
    standardize_regression_data,
)


METHOD_ORDER = ["DPD", "ExpSatDPD", "RatSatDPD", "NLL", "GammaDiv", "StudentT"]
PROPOSAL_METHODS = ["DPD", "ExpSatDPD", "RatSatDPD"]
METHOD_STYLES = {
    "DPD": {"color": "#4c78a8", "marker": "o"},
    "ExpSatDPD": {"color": "#e45756", "marker": "v"},
    "RatSatDPD": {"color": "#72b7b2", "marker": "P"},
    "NLL": {"color": "#2ca02c", "marker": "s"},
    "GammaDiv": {"color": "#f58518", "marker": "^"},
    "StudentT": {"color": "#b279a2", "marker": "D"},
}
PHI_LABEL_TO_METHOD = {
    "exp_sat": "ExpSatDPD",
    "rat_sat": "RatSatDPD",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def format_gamma_tag(gamma: float) -> str:
    return f"{float(gamma):g}".replace(".", "p")


def format_eps_tag(eps: float) -> str:
    return f"{float(eps):.2f}".replace(".", "p")


def find_gamma_dir(outputs_root: Path, prefix: str, gamma: float) -> Path:
    for gamma_found, path in discover_gamma_dirs(str(outputs_root), prefix):
        if math.isclose(float(gamma_found), float(gamma), rel_tol=0.0, abs_tol=1e-12):
            return path
    raise FileNotFoundError(f"Could not find {prefix}<gamma> for gamma={gamma} under {outputs_root}")


def find_gamma_dir_from_prefixes(outputs_root: Path, prefixes: list[str], gamma: float) -> Path:
    for prefix in prefixes:
        try:
            return find_gamma_dir(outputs_root, prefix, gamma)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        f"Could not find any of {prefixes}<gamma> for gamma={gamma} under {outputs_root}"
    )


def load_gamma_summaries(outputs_root: Path, prefix: str) -> pd.DataFrame:
    rows = []
    for gamma, path in discover_gamma_dirs(str(outputs_root), prefix):
        summary_path = path / "summary_agg.csv"
        if not summary_path.exists():
            continue
        df = pd.read_csv(summary_path)
        df["gamma_tag"] = float(gamma)
        df["source_dir"] = path.name
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def available_methods(df: pd.DataFrame) -> list[str]:
    return [m for m in METHOD_ORDER if "method" in df.columns and m in set(df["method"])]


def available_proposal_methods(df: pd.DataFrame) -> list[str]:
    methods = available_methods(df)
    if "eps_post_mean_mean" in df.columns:
        return [
            m for m in methods
            if m in PROPOSAL_METHODS and not df[df["method"] == m]["eps_post_mean_mean"].isna().all()
        ]
    return [m for m in methods if m in PROPOSAL_METHODS]


def plot_lines(
    ax,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    methods: list[str],
    *,
    yerr_col: str | None = None,
    title: str,
    xlabel: str,
    ylabel: str,
    proposal_only: bool = False,
    sort_by: str | None = None,
    identity: bool = False,
):
    plot_methods = available_proposal_methods(df) if proposal_only else methods
    if sort_by is None:
        sort_by = x_col
    if identity:
        x_vals = sorted(float(v) for v in pd.Series(df[x_col]).dropna().unique())
        if x_vals:
            max_x = max(x_vals)
            ax.plot([min(x_vals), max_x], [min(x_vals), max_x], "k--", linewidth=1.1, label="True")
    for method in plot_methods:
        df_m = df[df["method"] == method].sort_values(sort_by)
        if df_m.empty or y_col not in df_m.columns or df_m[y_col].isna().all():
            continue
        style = METHOD_STYLES.get(method, {"color": "gray", "marker": "x"})
        yerr = df_m[yerr_col] if yerr_col and yerr_col in df_m.columns else None
        ax.errorbar(
            df_m[x_col],
            df_m[y_col],
            yerr=yerr,
            fmt=f"-{style['marker']}",
            color=style["color"],
            capsize=3,
            label=method,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)


def plot_main_simulation_figure(df_sim: pd.DataFrame, gamma: float, scenario: str, output_path: Path) -> None:
    df = df_sim[
        np.isclose(df_sim["gamma_tag"], gamma)
        & (df_sim["scenario"] == scenario)
    ].copy()
    methods = available_methods(df)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    plot_lines(
        axes[0],
        df,
        "eps_true",
        "auprc_mean",
        methods,
        yerr_col="auprc_std",
        title=f"Simulation AUPRC ({scenario}, gamma={gamma:g})",
        xlabel="True contamination epsilon",
        ylabel="AUPRC",
    )
    plot_lines(
        axes[1],
        df,
        "eps_true",
        "eps_post_mean_mean",
        methods,
        yerr_col="eps_post_mean_std",
        title=f"Simulation epsilon estimation ({scenario}, gamma={gamma:g})",
        xlabel="True contamination epsilon",
        ylabel="Posterior mean epsilon",
        proposal_only=True,
        identity=True,
    )
    plot_lines(
        axes[2],
        df,
        "eps_true",
        "eps_ci95_cover_mean",
        methods,
        title=f"Simulation epsilon coverage ({scenario}, gamma={gamma:g})",
        xlabel="True contamination epsilon",
        ylabel="Coverage probability",
        proposal_only=True,
    )
    axes[2].axhline(0.95, color="k", linestyle="--", linewidth=1.1)
    axes[2].set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_main_regression_performance_figure(df_eps: pd.DataFrame, gamma: float, output_path: Path) -> None:
    df = df_eps[np.isclose(df_eps["gamma_tag"], gamma)].copy()
    methods = available_methods(df)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    plot_lines(
        axes[0],
        df,
        "eps_true",
        "rmse_test_mean",
        methods,
        yerr_col="rmse_test_std",
        title=f"Regression RMSE (gamma={gamma:g})",
        xlabel="True contamination epsilon",
        ylabel="Test RMSE",
    )
    plot_lines(
        axes[1],
        df,
        "eps_true",
        "auprc_mean",
        methods,
        yerr_col="auprc_std",
        title=f"Regression AUPRC (gamma={gamma:g})",
        xlabel="True contamination epsilon",
        ylabel="AUPRC",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_main_regression_epsilon_figure(df_eps: pd.DataFrame, gamma: float, output_path: Path) -> None:
    df = df_eps[np.isclose(df_eps["gamma_tag"], gamma)].copy()
    methods = available_methods(df)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    plot_lines(
        axes[0],
        df,
        "eps_true",
        "eps_post_mean_mean",
        methods,
        yerr_col="eps_post_mean_std",
        title=f"Regression epsilon estimation (gamma={gamma:g})",
        xlabel="True contamination epsilon",
        ylabel="Posterior mean epsilon",
        proposal_only=True,
        identity=True,
    )
    plot_lines(
        axes[1],
        df,
        "eps_true",
        "eps_ci95_cover_mean",
        methods,
        title=f"Regression epsilon coverage (gamma={gamma:g})",
        xlabel="True contamination epsilon",
        ylabel="Coverage probability",
        proposal_only=True,
    )
    axes[1].axhline(0.95, color="k", linestyle="--", linewidth=1.1)
    axes[1].set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_gamma_grid(
    df: pd.DataFrame,
    gamma_values: list[float],
    x_col: str,
    y_col: str,
    yerr_col: str | None,
    methods: list[str],
    *,
    output_path: Path,
    title_prefix: str,
    xlabel: str,
    ylabel: str,
    proposal_only: bool = False,
    identity: bool = False,
):
    ncols = len(gamma_values)
    fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 4.2), squeeze=False)
    for idx, gamma in enumerate(gamma_values):
        ax = axes[0, idx]
        df_g = df[np.isclose(df["gamma_tag"], gamma)].copy()
        plot_lines(
            ax,
            df_g,
            x_col,
            y_col,
            methods,
            yerr_col=yerr_col,
            title=f"{title_prefix} (gamma={gamma:g})",
            xlabel=xlabel,
            ylabel=ylabel if idx == 0 else "",
            proposal_only=proposal_only,
            identity=identity,
        )
        if idx > 0 and ax.get_legend() is not None:
            ax.get_legend().remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def parse_ablation_dirname(name: str, prefix: str):
    pattern = re.compile(
        rf"^{re.escape(prefix)}_(?P<phi>exp_sat|rat_sat)_kappa(?P<kappa>[^_]+)_gamma(?P<gamma>.+)$"
    )
    match = pattern.match(name)
    if not match:
        return None
    return {
        "phi_tag": match.group("phi"),
        "kappa": float(match.group("kappa").replace("p", ".")),
        "gamma": float(match.group("gamma").replace("p", ".")),
    }


def load_kappa_ablation_summaries(outputs_root: Path, prefix: str) -> pd.DataFrame:
    rows = []
    for path in outputs_root.iterdir():
        if not path.is_dir():
            continue
        meta = parse_ablation_dirname(path.name, prefix)
        if meta is None:
            continue
        summary_path = path / "summary_agg.csv"
        if not summary_path.exists():
            continue
        df = pd.read_csv(summary_path)
        for key, value in meta.items():
            df[key] = value
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _heatmap_panel(ax, mat: np.ndarray, row_labels: list[float], col_labels: list[float], title: str, cmap: str) -> None:
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)), [f"{g:g}" for g in col_labels], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), [f"{k:g}" for k in row_labels])
    ax.set_xlabel("gamma")
    ax.set_ylabel("kappa")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_kappa_ablation_heatmaps(
    df: pd.DataFrame,
    *,
    method: str,
    phi_tag: str,
    metric_specs: list[tuple[str, str, str]],
    output_path: Path,
    title: str,
) -> None:
    if df.empty:
        return
    df = df[df["method"] == method].copy()
    if df.empty:
        return
    gammas = sorted(df["gamma"].unique())
    kappas = sorted(df["kappa"].unique())
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(5.0 * len(metric_specs), 4.4), squeeze=False)
    for idx, (metric, panel_title, cmap) in enumerate(metric_specs):
        mat = np.full((len(kappas), len(gammas)), np.nan)
        for i, kappa in enumerate(kappas):
            for j, gamma in enumerate(gammas):
                subset = df[np.isclose(df["kappa"], kappa) & np.isclose(df["gamma"], gamma)]
                if not subset.empty and metric in subset.columns:
                    mat[i, j] = float(subset.iloc[0][metric])
        _heatmap_panel(axes[0, idx], mat, kappas, gammas, panel_title, cmap)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_simulation_kappa_appendix(outputs_root: Path, appendix_dir: Path) -> None:
    df = load_kappa_ablation_summaries(outputs_root, "exp1_simulation_kappa_ablation")
    if df.empty:
        return
    df = df[(df["scenario"] == "A") & np.isclose(df["eps_true"], 0.2)]
    for phi_tag, method in PHI_LABEL_TO_METHOD.items():
        df_phi = df[df["phi_tag"] == phi_tag]
        plot_kappa_ablation_heatmaps(
            df_phi,
            method=method,
            phi_tag=phi_tag,
            metric_specs=[
                ("auprc_mean", "AUPRC", "viridis"),
                ("brier_mean", "Brier", "magma_r"),
                ("eps_post_mean_mean", "Posterior mean epsilon", "plasma"),
            ],
            output_path=appendix_dir / f"fig_kappa_ablation_simulation_{phi_tag}.png",
            title=f"Simulation kappa ablation ({method}, scenario A, epsilon=0.2)",
        )


def plot_regression_kappa_appendix(outputs_root: Path, appendix_dir: Path) -> None:
    prefixes = [
        ("exp1_regression_eps_sweep_kappa_ablation", "eps_sweep"),
        ("exp1_regression_kappa_ablation", "conditions"),
    ]
    for prefix, mode in prefixes:
        df = load_kappa_ablation_summaries(outputs_root, prefix)
        if df.empty:
            continue
        for phi_tag, method in PHI_LABEL_TO_METHOD.items():
            df_phi = df[df["phi_tag"] == phi_tag]
            if mode == "eps_sweep":
                df_phi = df_phi[np.isclose(df_phi["eps_true"], 0.2)]
                title = f"Regression kappa ablation ({method}, eps sweep, epsilon=0.2)"
                output_name = f"fig_kappa_ablation_regression_eps_{phi_tag}.png"
            else:
                if "condition" not in df_phi.columns:
                    continue
                df_phi = df_phi[df_phi["condition"] == "mu6"]
                title = f"Regression kappa ablation ({method}, condition=mu6)"
                output_name = f"fig_kappa_ablation_regression_conditions_{phi_tag}.png"
            plot_kappa_ablation_heatmaps(
                df_phi,
                method=method,
                phi_tag=phi_tag,
                metric_specs=[
                    ("rmse_test_mean", "RMSE", "viridis_r"),
                    ("auprc_mean", "AUPRC", "viridis"),
                    ("eps_post_mean_mean", "Posterior mean epsilon", "plasma"),
                ],
                output_path=appendix_dir / output_name,
                title=title,
            )


def _extract_method_samples(raw: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    key_map = {
        "DPD": "samples_dpd",
        "ExpSatDPD": "samples_exp_sat_dpd",
        "RatSatDPD": "samples_rat_sat_dpd",
        "NLL": "samples_nll",
        "GammaDiv": "samples_gamma",
        "StudentT": "samples_student_t",
    }
    out = {}
    for method, key in key_map.items():
        if key in raw:
            arr = np.asarray(raw[key])
            if arr.ndim == 2 and arr.shape[0] > 0:
                out[method] = arr
    return out


def _proposal_epsilon_samples(raw: np.lib.npyio.NpzFile, method: str, samples: np.ndarray) -> np.ndarray | None:
    if method not in PROPOSAL_METHODS:
        return None
    suffix_map = {
        "DPD": "dpd",
        "ExpSatDPD": "exp_sat_dpd",
        "RatSatDPD": "rat_sat_dpd",
    }
    eps_key = f"eps_samples_{suffix_map[method]}"
    if eps_key in raw:
        return np.asarray(raw[eps_key], dtype=float)
    if samples.shape[1] >= 5:
        rho = samples[:, -1]
        return np.maximum(0.0, 1.0 - 1.0 / (1.0 + np.exp(-rho)))
    return None


def load_simulation_joint_case(args, gamma: float):
    gamma_dir = None
    for g, path in discover_gamma_dirs(str(Path(args.outputs_root)), "exp1_simulation_gamma_sweep_gamma"):
        if math.isclose(float(g), float(gamma), rel_tol=0.0, abs_tol=1e-12):
            gamma_dir = path
            break
    if gamma_dir is None:
        raise FileNotFoundError(f"Simulation gamma directory not found for gamma={gamma}")
    sample_path = gamma_dir / args.simulation_scenario / f"eps{args.posterior_eps:.2f}" / f"rep{args.rep:03d}" / "samples.npz"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing samples file: {sample_path}")
    raw = np.load(sample_path, allow_pickle=True)
    method_samples = _extract_method_samples(raw)
    methods = {}
    for method, arr in method_samples.items():
        methods[method] = {
            "x": np.asarray(arr[:, 0], dtype=float),
            "y": np.asarray(arr[:, 1], dtype=float),
            "eps": _proposal_epsilon_samples(raw, method, arr),
        }
    return {
        "gamma": float(gamma),
        "methods": methods,
        "truth": {"x": 0.0, "y": 0.0, "eps": float(args.posterior_eps)},
        "labels": {"x": r"$\mu_1$", "y": r"$\mu_2$"},
    }


def load_regression_joint_case(args, gamma: float, condition: str | None = None):
    condition_name = condition or args.regression_condition
    gamma_dir = find_gamma_dir_from_prefixes(
        Path(args.outputs_root),
        ["exp1_regression_final_response_huber_gamma", "exp1_regression_final_gamma"],
        gamma,
    )
    sample_path = gamma_dir / condition_name / f"rep{args.rep:03d}" / "samples.npz"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing samples file: {sample_path}")
    raw = np.load(sample_path, allow_pickle=True)
    raw_keys = set(raw.files)
    missing_proposals = {"samples_exp_sat_dpd", "samples_rat_sat_dpd"} - raw_keys
    fallback_raw = None
    if missing_proposals:
        eps_prefixes = [
            f"exp1_regression_eps_sweep_{condition_name}_gamma",
            "exp1_regression_eps_sweep_response_homogeneous_mu6_gamma",
            "exp1_regression_eps_sweep_mu6_gamma",
        ]
        try:
            eps_dir = find_gamma_dir_from_prefixes(Path(args.outputs_root), eps_prefixes, gamma)
            eps_sample = eps_dir / f"eps{args.posterior_eps:.2f}" / f"rep{args.rep:03d}" / "samples.npz"
            if eps_sample.exists():
                fallback_raw = np.load(eps_sample, allow_pickle=True)
        except FileNotFoundError:
            fallback_raw = None

    x_train = np.asarray(raw["x_train"])
    y_train = np.asarray(raw["y_train"])
    _, _, _, scaler = standardize_regression_data(x_train, y_train, x_train)
    beta_true = np.asarray(raw["beta_true"], dtype=float)
    method_samples = _extract_method_samples(raw)
    if fallback_raw is not None:
        fallback_samples = _extract_method_samples(fallback_raw)
        for method in ("ExpSatDPD", "RatSatDPD"):
            if method not in method_samples and method in fallback_samples:
                method_samples[method] = fallback_samples[method]

    methods = {}
    for method, arr in method_samples.items():
        transformed = recover_posterior_params_original_scale(np.asarray(arr, dtype=float), scaler)
        beta = np.asarray(transformed["beta"], dtype=float)
        if beta.shape[1] < 3:
            continue
        methods[method] = {
            "x": beta[:, 1],
            "y": beta[:, 2],
            "eps": _proposal_epsilon_samples(raw, method, arr),
            "intercept": beta[:, 0],
            "raw_samples": beta,
        }
    return {
        "gamma": float(gamma),
        "methods": methods,
        "truth": {
            "x": float(beta_true[1]),
            "y": float(beta_true[2]),
            "intercept": float(beta_true[0]),
            "eps": float(np.mean(raw["mask_outlier"])),
        },
        "labels": {"x": r"$\beta_1$", "y": r"$\beta_2$"},
        "condition": condition_name,
    }


def _draw_2d_kde_contour(ax, x: np.ndarray, y: np.ndarray, color: str) -> None:
    if len(x) < 20:
        return
    idx = np.arange(len(x))
    if len(idx) > 5000:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, size=5000, replace=False)
    pts = np.vstack([x[idx], y[idx]])
    kde = gaussian_kde(pts)
    xmin, xmax = np.percentile(x, [1, 99])
    ymin, ymax = np.percentile(y, [1, 99])
    xpad = max((xmax - xmin) * 0.15, 1e-3)
    ypad = max((ymax - ymin) * 0.15, 1e-3)
    xx, yy = np.meshgrid(
        np.linspace(xmin - xpad, xmax + xpad, 120),
        np.linspace(ymin - ypad, ymax + ypad, 120),
    )
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    levels = np.quantile(zz, [0.70, 0.85, 0.94])
    levels = np.unique(levels)
    if len(levels) == 0:
        return
    ax.contour(xx, yy, zz, levels=levels, colors=[color], linewidths=1.4, alpha=0.95)


def _draw_1d_kde(ax, values: np.ndarray, color: str, label: str) -> None:
    values = np.asarray(values, dtype=float)
    if values.size < 10:
        return
    kde = gaussian_kde(values)
    lo, hi = np.percentile(values, [0.5, 99.5])
    pad = max((hi - lo) * 0.15, 1e-3)
    xs = np.linspace(lo - pad, hi + pad, 300)
    ys = kde(xs)
    ax.plot(xs, ys, color=color, linewidth=1.9, label=label)


def plot_joint_posterior_comparison(cases, *, title: str, output_path: Path) -> None:
    if not cases:
        return
    ncols = len(cases)
    fig, axes = plt.subplots(2, ncols, figsize=(4.3 * ncols, 8.2), squeeze=False)

    method_handles = {}
    for col, case in enumerate(cases):
        gamma = case["gamma"]
        ax_joint = axes[0, col]
        ax_eps = axes[1, col]
        for method in METHOD_ORDER:
            if method not in case["methods"]:
                continue
            info = case["methods"][method]
            color = METHOD_STYLES[method]["color"]
            _draw_2d_kde_contour(ax_joint, info["x"], info["y"], color)
            if method not in method_handles:
                method_handles[method] = plt.Line2D([0], [0], color=color, lw=1.8, label=method)
        ax_joint.axvline(case["truth"]["x"], color="black", linestyle="--", linewidth=1.1)
        ax_joint.axhline(case["truth"]["y"], color="black", linestyle="--", linewidth=1.1)
        ax_joint.set_title(rf"$\gamma={gamma:g}$")
        ax_joint.set_xlabel(case["labels"]["x"])
        if col == 0:
            ax_joint.set_ylabel(case["labels"]["y"])
        ax_joint.grid(True, alpha=0.2)

        for method in PROPOSAL_METHODS:
            if method not in case["methods"]:
                continue
            eps_vals = case["methods"][method]["eps"]
            if eps_vals is None:
                continue
            _draw_1d_kde(ax_eps, eps_vals, METHOD_STYLES[method]["color"], method)
        ax_eps.axvline(case["truth"]["eps"], color="black", linestyle="--", linewidth=1.1, label="True epsilon" if col == 0 else None)
        ax_eps.set_xlabel(r"$\epsilon$")
        if col == 0:
            ax_eps.set_ylabel("density")
        ax_eps.grid(True, alpha=0.2)
        if col == ncols - 1:
            ax_eps.legend(frameon=True)

    handles = list(method_handles.values())
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--", lw=1.2, label="True value"))
    fig.legend(handles=handles, loc="upper center", ncol=min(7, len(handles)), frameon=True)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def _load_simulation_method_samples(args, gamma: float) -> dict[str, np.ndarray]:
    gamma_dir = find_gamma_dir(Path(args.outputs_root), "exp1_simulation_gamma_sweep_gamma", gamma)
    sample_path = gamma_dir / args.simulation_scenario / f"eps{args.posterior_eps:.2f}" / f"rep{args.rep:03d}" / "samples.npz"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing simulation samples file: {sample_path}")
    raw = np.load(sample_path, allow_pickle=True)
    return _extract_method_samples(raw)


def _load_regression_method_samples(args, gamma: float, condition: str) -> tuple[dict[str, np.ndarray], dict]:
    gamma_dir = find_gamma_dir_from_prefixes(
        Path(args.outputs_root),
        ["exp1_regression_final_response_huber_gamma", "exp1_regression_final_gamma"],
        gamma,
    )
    sample_path = gamma_dir / condition / f"rep{args.rep:03d}" / "samples.npz"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing regression samples file: {sample_path}")
    raw = np.load(sample_path, allow_pickle=True)
    x_train = np.asarray(raw["x_train"])
    y_train = np.asarray(raw["y_train"])
    _, _, _, scaler = standardize_regression_data(x_train, y_train, x_train)
    return _extract_method_samples(raw), scaler


def plot_appendix_simulation_scale_posteriors(args, appendix_dir: Path) -> None:
    gammas = parse_float_list(args.posterior_main_gammas) or [0.1, 0.5, 1.0]
    cases = []
    for g in gammas:
        try:
            cases.append((g, _load_simulation_method_samples(args, g)))
        except FileNotFoundError:
            continue
    if not cases:
        return
    fig, axes = plt.subplots(2, len(cases), figsize=(5.0 * len(cases), 7.0), squeeze=False)
    for col, (gamma, method_samples) in enumerate(cases):
        ax_sx = axes[0, col]
        ax_sy = axes[1, col]
        for method in METHOD_ORDER:
            if method not in method_samples:
                continue
            arr = np.asarray(method_samples[method], dtype=float)
            if arr.shape[1] < 4:
                continue
            sig_x = np.exp(arr[:, 2])
            sig_y = np.exp(arr[:, 3])
            _draw_1d_kde(ax_sx, sig_x, METHOD_STYLES[method]["color"], method)
            _draw_1d_kde(ax_sy, sig_y, METHOD_STYLES[method]["color"], method)
        ax_sx.set_title(rf"$\gamma={float(gamma):g}$, $\sigma_x$")
        ax_sy.set_title(rf"$\gamma={float(gamma):g}$, $\sigma_y$")
        ax_sx.set_xlabel(r"$\sigma_x$")
        ax_sy.set_xlabel(r"$\sigma_y$")
        if col == 0:
            ax_sx.set_ylabel("density")
            ax_sy.set_ylabel("density")
        ax_sx.grid(True, alpha=0.2)
        ax_sy.grid(True, alpha=0.2)
    handles = [plt.Line2D([0], [0], color=METHOD_STYLES[m]["color"], lw=2.0, label=m) for m in METHOD_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=min(6, len(handles)), frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(
        appendix_dir / f"fig_appendix_simulation_sigma_posteriors_{args.simulation_scenario}_eps{format_eps_tag(args.posterior_eps)}.png",
        bbox_inches="tight",
        dpi=220,
    )
    plt.close(fig)


def plot_appendix_regression_intercept_sigma_posteriors(args, appendix_dir: Path, condition: str) -> None:
    gammas = parse_float_list(args.posterior_main_gammas) or [0.1, 0.5, 1.0]
    cases = []
    for g in gammas:
        try:
            method_samples, scaler = _load_regression_method_samples(args, g, condition)
            cases.append((g, method_samples, scaler))
        except FileNotFoundError:
            continue
    if not cases:
        return
    fig, axes = plt.subplots(2, len(cases), figsize=(5.0 * len(cases), 7.0), squeeze=False)
    for col, (gamma, method_samples, scaler) in enumerate(cases):
        ax_b0 = axes[0, col]
        ax_sig = axes[1, col]
        for method in METHOD_ORDER:
            if method not in method_samples:
                continue
            arr = np.asarray(method_samples[method], dtype=float)
            transformed = recover_posterior_params_original_scale(arr, scaler)
            beta = np.asarray(transformed["beta"], dtype=float)
            sigma = np.asarray(transformed["sigma"], dtype=float)
            if beta.shape[1] < 1:
                continue
            _draw_1d_kde(ax_b0, beta[:, 0], METHOD_STYLES[method]["color"], method)
            _draw_1d_kde(ax_sig, sigma, METHOD_STYLES[method]["color"], method)
        ax_b0.set_title(rf"{condition}, $\gamma={float(gamma):g}$, intercept")
        ax_sig.set_title(rf"{condition}, $\gamma={float(gamma):g}$, $\sigma$")
        ax_b0.set_xlabel("intercept")
        ax_sig.set_xlabel(r"$\sigma$")
        if col == 0:
            ax_b0.set_ylabel("density")
            ax_sig.set_ylabel("density")
        ax_b0.grid(True, alpha=0.2)
        ax_sig.grid(True, alpha=0.2)
    handles = [plt.Line2D([0], [0], color=METHOD_STYLES[m]["color"], lw=2.0, label=m) for m in METHOD_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=min(6, len(handles)), frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(
        appendix_dir / f"fig_appendix_regression_intercept_sigma_posteriors_{condition}.png",
        bbox_inches="tight",
        dpi=220,
    )
    plt.close(fig)


def load_regression_outlier_case(outputs_root: Path, gamma: float, condition: str, eps: float, rep: int):
    sample_path = None
    # 1) Condition-specific eps-sweep directory (if present).
    try:
        gamma_dir = find_gamma_dir_from_prefixes(
            outputs_root,
            [
                f"exp1_regression_eps_sweep_{condition}_gamma",
            ],
            gamma,
        )
        candidate = gamma_dir / f"eps{eps:.2f}" / f"rep{rep:03d}" / "samples.npz"
        if candidate.exists():
            sample_path = candidate
    except FileNotFoundError:
        sample_path = None
    # 2) Shared homogeneous eps-sweep directory (only for homogeneous condition).
    if sample_path is None and condition == "hom_mu6":
        try:
            gamma_dir = find_gamma_dir_from_prefixes(
                outputs_root,
                [
                    "exp1_regression_eps_sweep_response_homogeneous_mu6_gamma",
                    "exp1_regression_eps_sweep_mu6_gamma",
                ],
                gamma,
            )
            candidate = gamma_dir / f"eps{eps:.2f}" / f"rep{rep:03d}" / "samples.npz"
            if candidate.exists():
                sample_path = candidate
        except FileNotFoundError:
            sample_path = None
    # 3) Conditions output fallback.
    if sample_path is None:
        gamma_dir = find_gamma_dir_from_prefixes(
            outputs_root,
            ["exp1_regression_final_response_huber_gamma", "exp1_regression_final_gamma"],
            gamma,
        )
        sample_path = gamma_dir / condition / f"rep{rep:03d}" / "samples.npz"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing regression samples: {sample_path}")
    raw = np.load(sample_path, allow_pickle=True)
    x_train = np.asarray(raw["x_train"], dtype=float)
    y_train = np.asarray(raw["y_train"], dtype=float)
    p_out = np.asarray(raw["p_out_dpd"], dtype=float)
    mask_outlier = np.asarray(raw["mask_outlier"], dtype=bool)
    _, _, _, scaler = standardize_regression_data(x_train, y_train, x_train)
    samples_dpd = np.asarray(raw["samples_dpd"], dtype=float)
    return x_train, y_train, p_out, mask_outlier, samples_dpd, scaler


def load_simulation_outlier_case(outputs_root: Path, gamma: float, scenario: str, eps: float, rep: int):
    gamma_dir = find_gamma_dir(outputs_root, "exp1_simulation_gamma_sweep_gamma", gamma)
    sample_path = gamma_dir / scenario / f"eps{eps:.2f}" / f"rep{rep:03d}" / "samples.npz"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing simulation samples: {sample_path}")
    raw = np.load(sample_path, allow_pickle=True)
    x = np.asarray(raw["x"], dtype=float)
    p_key = "p_out_dpd" if "p_out_dpd" in raw.files else "p_out"
    p_out = np.asarray(raw[p_key], dtype=float)
    y_true = np.asarray(raw["y_true"], dtype=bool)
    samples_dpd = np.asarray(raw["samples_dpd"], dtype=float)
    return x, p_out, y_true, samples_dpd


def _overlay_regression_surface_uncertainty(
    ax,
    x_train: np.ndarray,
    samples_dpd: np.ndarray,
    scaler: dict,
) -> None:
    transformed = recover_posterior_params_original_scale(samples_dpd, scaler)
    beta_samples = np.asarray(transformed["beta"], dtype=float)
    sigma_samples = np.asarray(transformed["sigma"], dtype=float)
    if beta_samples.shape[0] == 0 or beta_samples.shape[1] < 3:
        return
    max_draws = min(1200, beta_samples.shape[0])
    beta_draws = beta_samples[:max_draws]
    sigma_draws = sigma_samples[:max_draws]

    qx = np.percentile(x_train[:, 0], [1, 99])
    qy = np.percentile(x_train[:, 1], [1, 99])
    x_pad = max((qx[1] - qx[0]) * 0.35, 0.35)
    y_pad = max((qy[1] - qy[0]) * 0.35, 0.35)
    gx = np.linspace(qx[0] - x_pad, qx[1] + x_pad, 36)
    gy = np.linspace(qy[0] - y_pad, qy[1] + y_pad, 36)
    xx, yy = np.meshgrid(gx, gy)
    flat_x = xx.ravel()
    flat_y = yy.ravel()

    mean_beta = beta_draws.mean(axis=0)
    z_mean = mean_beta[0] + mean_beta[1] * flat_x + mean_beta[2] * flat_y
    y_samps = (
        beta_draws[:, 0][:, None]
        + beta_draws[:, 1][:, None] * flat_x[None, :]
        + beta_draws[:, 2][:, None] * flat_y[None, :]
    )
    z_std = np.sqrt(np.var(y_samps, axis=0) + np.mean(sigma_draws**2))

    zz = z_mean.reshape(xx.shape)
    zz_up = (z_mean + z_std).reshape(xx.shape)
    zz_lo = (z_mean - z_std).reshape(xx.shape)
    ax.plot_surface(xx, yy, zz, color="dimgray", alpha=0.34, linewidth=0, antialiased=True)
    ax.plot_surface(xx, yy, zz_up, color="gray", alpha=0.16, linewidth=0, antialiased=True)
    ax.plot_surface(xx, yy, zz_lo, color="gray", alpha=0.16, linewidth=0, antialiased=True)
    # Add wireframes so the regression trend remains visible from any view angle.
    ax.plot_wireframe(xx, yy, zz, rstride=4, cstride=4, color="#5a5a5a", linewidth=0.65, alpha=0.85)
    ax.plot_wireframe(xx, yy, zz_up, rstride=6, cstride=6, color="#808080", linewidth=0.45, alpha=0.55)
    ax.plot_wireframe(xx, yy, zz_lo, rstride=6, cstride=6, color="#808080", linewidth=0.45, alpha=0.55)


def plot_regression_outlier_3d(
    ax,
    x_train: np.ndarray,
    y_train: np.ndarray,
    p_out: np.ndarray,
    mask_outlier: np.ndarray,
    samples_dpd: np.ndarray,
    scaler: dict,
    title: str,
) -> None:
    sc = ax.scatter(
        x_train[:, 0],
        x_train[:, 1],
        y_train,
        c=p_out,
        cmap="coolwarm",
        s=18,
        alpha=0.88,
        vmin=0.0,
        vmax=1.0,
    )
    if mask_outlier.any():
        ax.scatter(
            x_train[mask_outlier, 0],
            x_train[mask_outlier, 1],
            y_train[mask_outlier],
            facecolors="none",
            edgecolors="black",
            s=34,
            linewidths=0.9,
        )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.set_title(title)
    return sc


def _overlay_simulation_density_contour(ax, x: np.ndarray, samples_dpd: np.ndarray) -> None:
    if samples_dpd.shape[0] == 0 or samples_dpd.shape[1] < 4:
        return
    max_draws = min(1200, samples_dpd.shape[0])
    draws = samples_dpd[:max_draws]
    mu = draws[:, :2]
    scales = np.exp(draws[:, 2:4])

    qx = np.percentile(x[:, 0], [1, 99])
    qy = np.percentile(x[:, 1], [1, 99])
    pad_x = max((qx[1] - qx[0]) * 0.18, 1e-3)
    pad_y = max((qy[1] - qy[0]) * 0.18, 1e-3)
    gx = np.linspace(qx[0] - pad_x, qx[1] + pad_x, 140)
    gy = np.linspace(qy[0] - pad_y, qy[1] + pad_y, 140)
    xx, yy = np.meshgrid(gx, gy)

    dx = (xx[None, :, :] - mu[:, 0][:, None, None]) / scales[:, 0][:, None, None]
    dy = (yy[None, :, :] - mu[:, 1][:, None, None]) / scales[:, 1][:, None, None]
    norm = 1.0 / (2.0 * np.pi * scales[:, 0][:, None, None] * scales[:, 1][:, None, None])
    dens = norm * np.exp(-0.5 * (dx**2 + dy**2))
    dens_mean = dens.mean(axis=0)
    levels = np.quantile(dens_mean, [0.70, 0.84, 0.93, 0.975])
    levels = np.unique(levels)
    if len(levels) == 0:
        return
    ax.contour(
        xx,
        yy,
        dens_mean,
        levels=levels,
        colors="#6e6e6e",
        linewidths=1.3,
        linestyles="--",
        alpha=0.95,
    )


def plot_simulation_outlier_2d(
    ax,
    x: np.ndarray,
    p_out: np.ndarray,
    y_true: np.ndarray,
    samples_dpd: np.ndarray,
    title: str,
):
    sc = ax.scatter(
        x[:, 0],
        x[:, 1],
        c=p_out,
        cmap="coolwarm",
        s=22,
        alpha=0.88,
        vmin=0.0,
        vmax=1.0,
    )
    if y_true.any():
        ax.scatter(
            x[y_true, 0],
            x[y_true, 1],
            facecolors="none",
            edgecolors="black",
            s=36,
            linewidths=0.9,
        )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    return sc


def generate_outlier_visualizations(args, outputs_root: Path, main_dir: Path, appendix_dir: Path) -> None:
    gamma_list = parse_float_list(args.outlier_vis_gammas) or [0.1, 0.5, 1.0]
    eps = float(args.posterior_eps)
    eps_tag = format_eps_tag(eps)

    # Main text: regression 3D (single gamma), simulation 2D (single gamma)
    try:
        x_train, y_train, p_out_reg, mask_reg, samples_reg, scaler = load_regression_outlier_case(
            outputs_root, args.regression_main_gamma, args.regression_condition, eps, args.rep
        )
        fig = plt.figure(figsize=(7.2, 5.7))
        ax = fig.add_subplot(111, projection="3d")
        sc = plot_regression_outlier_3d(
            ax,
            x_train,
            y_train,
            p_out_reg,
            mask_reg,
            samples_reg,
            scaler,
            title=rf"Regression outlier probabilities ({args.regression_condition}, $\gamma={args.regression_main_gamma:g}$, $\epsilon={eps:.2f}$)",
        )
        fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.08, label="DPD outlier probability")
        fig.tight_layout()
        fig.savefig(
            main_dir / f"fig_main_regression_outlier3d_{args.regression_condition}_eps{eps_tag}_gamma{format_gamma_tag(args.regression_main_gamma)}.png",
            bbox_inches="tight",
            dpi=220,
        )
        plt.close(fig)
    except FileNotFoundError:
        pass

    try:
        x_sim, p_out_sim, y_true_sim, samples_sim = load_simulation_outlier_case(
            outputs_root, args.simulation_main_gamma, args.simulation_scenario, eps, args.rep
        )
        fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.6))
        sc = plot_simulation_outlier_2d(
            ax,
            x_sim,
            p_out_sim,
            y_true_sim,
            samples_sim,
            title=rf"Simulation outlier probabilities ({args.simulation_scenario}, $\gamma={args.simulation_main_gamma:g}$, $\epsilon={eps:.2f}$)",
        )
        fig.colorbar(sc, ax=ax, label="DPD outlier probability")
        fig.tight_layout()
        fig.savefig(
            main_dir / f"fig_main_simulation_outlier2d_{args.simulation_scenario}_eps{eps_tag}_gamma{format_gamma_tag(args.simulation_main_gamma)}.png",
            bbox_inches="tight",
            dpi=220,
        )
        plt.close(fig)
    except FileNotFoundError:
        pass

    # Appendix: gamma sweep panels
    reg_cases = []
    for gamma in gamma_list:
        try:
            reg_cases.append((gamma, *load_regression_outlier_case(outputs_root, gamma, args.regression_condition, eps, args.rep)))
        except FileNotFoundError:
            continue
    if reg_cases:
        fig = plt.figure(figsize=(5.2 * len(reg_cases), 5.8))
        axs = [fig.add_subplot(1, len(reg_cases), i + 1, projection="3d") for i in range(len(reg_cases))]
        last_sc = None
        for ax, (gamma, x_train, y_train, p_out_reg, mask_reg, samples_reg, scaler) in zip(axs, reg_cases):
            last_sc = plot_regression_outlier_3d(
                ax,
                x_train,
                y_train,
                p_out_reg,
                mask_reg,
                samples_reg,
                scaler,
                title=rf"$\gamma={gamma:g}$",
            )
        fig.colorbar(last_sc, ax=axs, shrink=0.75, pad=0.04, label="DPD outlier probability")
        fig.suptitle(rf"Regression outlier probabilities ({args.regression_condition}, $\epsilon={eps:.2f}$)")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(
            appendix_dir / f"fig_appendix_regression_outlier3d_{args.regression_condition}_eps{eps_tag}_gammas.png",
            bbox_inches="tight",
            dpi=220,
        )
        plt.close(fig)

    sim_cases = []
    for gamma in gamma_list:
        try:
            sim_cases.append((gamma, *load_simulation_outlier_case(outputs_root, gamma, args.simulation_scenario, eps, args.rep)))
        except FileNotFoundError:
            continue
    if sim_cases:
        fig, axes = plt.subplots(1, len(sim_cases), figsize=(5.2 * len(sim_cases), 5.0), squeeze=False)
        last_sc = None
        for ax, (gamma, x_sim, p_out_sim, y_true_sim, samples_sim) in zip(axes[0], sim_cases):
            last_sc = plot_simulation_outlier_2d(
                ax,
                x_sim,
                p_out_sim,
                y_true_sim,
                samples_sim,
                title=rf"$\gamma={gamma:g}$",
            )
        fig.colorbar(last_sc, ax=axes[0], shrink=0.9, pad=0.02, label="DPD outlier probability")
        fig.suptitle(rf"Simulation outlier probabilities ({args.simulation_scenario}, $\epsilon={eps:.2f}$)")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(
            appendix_dir / f"fig_appendix_simulation_outlier2d_{args.simulation_scenario}_eps{eps_tag}_gammas.png",
            bbox_inches="tight",
            dpi=220,
        )
        plt.close(fig)


def _eps_to_k_stats(samples_dpd: np.ndarray, n_obs: int) -> dict[str, float]:
    rho = np.asarray(samples_dpd[:, -1], dtype=float)
    eps_samples = np.maximum(0.0, 1.0 - 1.0 / (1.0 + np.exp(-rho)))
    k_samples = np.clip(np.round(eps_samples * float(n_obs)).astype(int), 0, n_obs)
    k_low, k_med, k_high = np.percentile(k_samples, [2.5, 50.0, 97.5])
    k_low_i = int(np.clip(round(k_low), 0, n_obs))
    k_med_i = int(np.clip(round(k_med), 0, n_obs))
    k_high_i = int(np.clip(round(k_high), 0, n_obs))
    k_sorted = sorted([k_low_i, k_med_i, k_high_i])
    return {
        "eps_mean": float(eps_samples.mean()),
        "eps_low": float(np.percentile(eps_samples, 2.5)),
        "eps_high": float(np.percentile(eps_samples, 97.5)),
        "k_low": int(k_sorted[0]),
        "k_med": int(k_sorted[1]),
        "k_high": int(k_sorted[2]),
    }


def _compute_regression_loglik(x_train: np.ndarray, y_train: np.ndarray, samples_dpd: np.ndarray) -> np.ndarray:
    x_design, y_scaled, _, _ = standardize_regression_data(x_train, y_train, x_train)
    x_np = np.asarray(x_design, dtype=float)
    y_np = np.asarray(y_scaled, dtype=float)
    theta_mean = np.asarray(samples_dpd, dtype=float).mean(axis=0)
    d_in = x_np.shape[1]
    beta = theta_mean[:d_in]
    log_scale = float(theta_mean[d_in])
    sigma = max(float(np.exp(log_scale)), 1e-9)
    mu = x_np @ beta
    z = (y_np - mu) / sigma
    return -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - 0.5 * (z**2)


def _compute_simulation_loglik(x: np.ndarray, samples_dpd: np.ndarray) -> np.ndarray:
    theta_mean = np.asarray(samples_dpd, dtype=float).mean(axis=0)
    mu = theta_mean[:2]
    scales = np.maximum(np.exp(theta_mean[2:4]), 1e-9)
    z = (x - mu[None, :]) / scales[None, :]
    norm = np.log(2.0 * np.pi) + np.log(scales[0]) + np.log(scales[1])
    return -norm - 0.5 * np.sum(z**2, axis=1)


def _plot_rank_panel(ax, ll: np.ndarray, k_stats: dict[str, float], title: str) -> None:
    order = np.argsort(ll)
    ll_sorted = ll[order]
    ranks = np.arange(1, len(ll_sorted) + 1)
    x_left = float(np.min(ll_sorted)) - 0.03 * max(1e-9, float(np.ptp(ll_sorted)))
    k_low = int(k_stats["k_low"])
    k_med = int(k_stats["k_med"])
    k_high = int(k_stats["k_high"])

    # Segment-wise fill for clearer interpretation of epsilon-derived k range.
    seg1 = ranks <= max(k_low, 1)
    seg2 = (ranks > max(k_low, 1)) & (ranks <= max(k_med, 1))
    seg3 = (ranks > max(k_med, 1)) & (ranks <= max(k_high, 1))
    seg4 = ranks > max(k_high, 1)
    if np.any(seg1):
        ax.fill_betweenx(ranks[seg1], x_left, ll_sorted[seg1], color="#b30000", alpha=0.24, linewidth=0.0)
    if np.any(seg2):
        ax.fill_betweenx(ranks[seg2], x_left, ll_sorted[seg2], color="#d7301f", alpha=0.20, linewidth=0.0)
    if np.any(seg3):
        ax.fill_betweenx(ranks[seg3], x_left, ll_sorted[seg3], color="#ef6548", alpha=0.16, linewidth=0.0)
    if np.any(seg4):
        ax.fill_betweenx(ranks[seg4], x_left, ll_sorted[seg4], color="#4c78a8", alpha=0.20, linewidth=0.0)
    ax.plot(ll_sorted, ranks, color="black", linewidth=1.5)
    ax.axhline(k_low, color="#d62728", linestyle=":", linewidth=1.0, label="k 2.5%")
    ax.axhline(k_med, color="#d62728", linestyle="-", linewidth=1.2, label="k median")
    ax.axhline(k_high, color="#d62728", linestyle="--", linewidth=1.0, label="k 97.5%")
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Log-likelihood", fontsize=17)
    ax.set_ylabel("")
    ax.set_ylim(len(ranks) + 2, 0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=13, frameon=True)
    ax.tick_params(axis="both", labelsize=15)
    txt = (
        rf"$\hat{{\epsilon}}={k_stats['eps_mean']:.3f}$"
        + "\n"
        + rf"$\epsilon_{{95\%\mathrm{{CrI}}}}=[{k_stats['eps_low']:.3f}, {k_stats['eps_high']:.3f}]$"
        + "\n"
        + rf"$k_{{50}}={k_stats['k_med']}$, $k_{{95}}={k_stats['k_high']}$"
    )
    ax.text(0.03, 0.06, txt, transform=ax.transAxes, va="bottom", ha="left", fontsize=13, bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"})


def _plot_selected_points_panel(
    ax,
    x: np.ndarray,
    order: np.ndarray,
    k_stats: dict[str, float],
    truth_mask: np.ndarray | None,
    title: str,
) -> None:
    k_med = int(k_stats["k_med"])
    k_high = int(k_stats["k_high"])
    idx_med = order[:k_med] if k_med > 0 else np.array([], dtype=int)
    idx_high = order[:k_high] if k_high > 0 else np.array([], dtype=int)
    idx_band = np.setdiff1d(idx_high, idx_med, assume_unique=False)

    ax.scatter(x[:, 0], x[:, 1], c="lightgray", s=14, alpha=0.55, label="all data")
    if len(idx_band) > 0:
        ax.scatter(x[idx_band, 0], x[idx_band, 1], c="#ffb366", s=22, alpha=0.85, label="k95 band")
    if len(idx_med) > 0:
        ax.scatter(x[idx_med, 0], x[idx_med, 1], c="#d62728", s=26, alpha=0.9, label="top-k50")
    if truth_mask is not None and np.any(truth_mask):
        ax.scatter(
            x[truth_mask, 0],
            x[truth_mask, 1],
            facecolors="none",
            edgecolors="black",
            s=36,
            linewidths=0.9,
            label="true outlier",
        )
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", fontsize=8, frameon=True)


def _plot_selected_points_panel_3d(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    order: np.ndarray,
    k_stats: dict[str, float],
    truth_mask: np.ndarray | None,
    title: str,
    swap_xy: bool = False,
) -> None:
    x_plot = x[:, 1] if swap_xy else x[:, 0]
    y_plot = x[:, 0] if swap_xy else x[:, 1]
    xlabel = "x2" if swap_xy else "x1"
    ylabel = "x1" if swap_xy else "x2"

    k_med = int(k_stats["k_med"])
    k_high = int(k_stats["k_high"])
    idx_med = order[:k_med] if k_med > 0 else np.array([], dtype=int)
    idx_high = order[:k_high] if k_high > 0 else np.array([], dtype=int)
    idx_band = np.setdiff1d(idx_high, idx_med, assume_unique=False)
    idx_bg_all = np.setdiff1d(np.arange(len(x)), idx_high, assume_unique=False)
    # Downsample background points so highlighted sets remain visually dominant.
    if len(idx_bg_all) > 0:
        rng = np.random.default_rng(0)
        keep_bg = max(int(0.22 * len(idx_bg_all)), min(180, len(idx_bg_all)))
        idx_bg = rng.choice(idx_bg_all, size=keep_bg, replace=False) if keep_bg < len(idx_bg_all) else idx_bg_all
    else:
        idx_bg = np.array([], dtype=int)

    if len(idx_bg) > 0:
        ax.scatter(
            x_plot[idx_bg],
            y_plot[idx_bg],
            y[idx_bg],
            c="#9b9b9b",
            s=9,
            alpha=0.32,
            edgecolors="#7f7f7f",
            linewidths=0.2,
            depthshade=False,
        )
    if len(idx_band) > 0:
        ax.scatter(
            x_plot[idx_band],
            y_plot[idx_band],
            y[idx_band],
            c="#e67e22",
            s=22,
            alpha=0.96,
            edgecolors="#a95a00",
            linewidths=0.4,
            depthshade=False,
        )
    if len(idx_med) > 0:
        ax.scatter(
            x_plot[idx_med],
            y_plot[idx_med],
            y[idx_med],
            c="#b30000",
            s=26,
            alpha=0.98,
            edgecolors="#660000",
            linewidths=0.45,
            depthshade=False,
        )
    if truth_mask is not None and np.any(truth_mask):
        ax.scatter(
            x_plot[truth_mask],
            y_plot[truth_mask],
            y[truth_mask],
            facecolors="none",
            edgecolors="black",
            s=34,
            linewidths=1.0,
            depthshade=False,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("y")
    ax.view_init(elev=22, azim=-62)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#9b9b9b", markeredgecolor="#7f7f7f", markersize=6, alpha=0.8, label="background data"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e67e22", markeredgecolor="#a95a00", markersize=7, label="k95 band"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#b30000", markeredgecolor="#660000", markersize=7, label="top-k50"),
    ]
    if truth_mask is not None and np.any(truth_mask):
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color="black", markerfacecolor="none", markersize=7, label="true outlier")
        )
    ax.legend(handles=legend_handles, loc="upper center", ncol=2, fontsize=9, frameon=True)


def _plot_likelihood_selection_figure(
    ll: np.ndarray,
    x_for_panel: np.ndarray,
    k_stats: dict[str, float],
    truth_mask: np.ndarray | None,
    y_for_panel: np.ndarray | None = None,
    *,
    output_path: Path,
    title: str,
) -> None:
    order = np.argsort(ll)
    fig = plt.figure(figsize=(12.2, 4.8))
    ax_rank = fig.add_subplot(1, 2, 1)
    if y_for_panel is None:
        ax_points = fig.add_subplot(1, 2, 2)
    else:
        ax_points = fig.add_subplot(1, 2, 2, projection="3d")
    _plot_rank_panel(ax_rank, ll, k_stats, title="Likelihood ranking with inferred k")
    if y_for_panel is None:
        _plot_selected_points_panel(ax_points, x_for_panel, order, k_stats, truth_mask, title="Points selected by ranking threshold")
    else:
        _plot_selected_points_panel_3d(ax_points, x_for_panel, y_for_panel, order, k_stats, truth_mask, title="Points selected by ranking threshold")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def generate_likelihood_ranking_visualizations(args, outputs_root: Path, main_dir: Path, appendix_dir: Path) -> None:
    eps = float(args.posterior_eps)
    eps_tag = format_eps_tag(eps)
    gamma_list = parse_float_list(args.outlier_vis_gammas) or [0.1, 0.5, 1.0]

    # Main text: representative gamma for each task
    try:
        x_train, y_train, _, mask_reg, samples_reg, _ = load_regression_outlier_case(
            outputs_root, args.regression_main_gamma, args.regression_condition, eps, args.rep
        )
        ll_reg = _compute_regression_loglik(x_train, y_train, samples_reg)
        k_stats_reg = _eps_to_k_stats(samples_reg, len(x_train))
        _plot_likelihood_selection_figure(
            ll_reg,
            x_train,
            k_stats_reg,
            mask_reg,
            y_for_panel=y_train,
            output_path=main_dir / f"fig_main_regression_likelihood_selection_{args.regression_condition}_eps{eps_tag}_gamma{format_gamma_tag(args.regression_main_gamma)}.png",
            title=rf"Regression likelihood ranking and selected points ({args.regression_condition}, $\gamma={args.regression_main_gamma:g}$, $\epsilon={eps:.2f}$)",
        )
    except FileNotFoundError:
        pass

    try:
        x_sim, _, y_true_sim, samples_sim = load_simulation_outlier_case(
            outputs_root, args.simulation_main_gamma, args.simulation_scenario, eps, args.rep
        )
        ll_sim = _compute_simulation_loglik(x_sim, samples_sim)
        k_stats_sim = _eps_to_k_stats(samples_sim, len(x_sim))
        _plot_likelihood_selection_figure(
            ll_sim,
            x_sim,
            k_stats_sim,
            y_true_sim,
            output_path=main_dir / f"fig_main_simulation_likelihood_selection_{args.simulation_scenario}_eps{eps_tag}_gamma{format_gamma_tag(args.simulation_main_gamma)}.png",
            title=rf"Simulation likelihood ranking and selected points ({args.simulation_scenario}, $\gamma={args.simulation_main_gamma:g}$, $\epsilon={eps:.2f}$)",
        )
    except FileNotFoundError:
        pass

    # Appendix: gamma comparisons
    reg_cases = []
    for gamma in gamma_list:
        try:
            x_train, y_train, _, mask_reg, samples_reg, _ = load_regression_outlier_case(
                outputs_root, gamma, args.regression_condition, eps, args.rep
            )
            reg_cases.append((gamma, x_train, y_train, _compute_regression_loglik(x_train, y_train, samples_reg), _eps_to_k_stats(samples_reg, len(x_train)), mask_reg))
        except FileNotFoundError:
            continue
    if reg_cases:
        ncols = len(reg_cases)
        fig = plt.figure(figsize=(5.2 * ncols, 8.2))
        for col, (gamma, x_train, y_train, ll_reg, k_stats_reg, mask_reg) in enumerate(reg_cases):
            ax_rank = fig.add_subplot(2, ncols, col + 1)
            ax_sel = fig.add_subplot(2, ncols, ncols + col + 1, projection="3d")
            _plot_rank_panel(ax_rank, ll_reg, k_stats_reg, title=rf"$\gamma={gamma:g}$")
            _plot_selected_points_panel_3d(
                ax_sel,
                x_train,
                y_train,
                np.argsort(ll_reg),
                k_stats_reg,
                mask_reg,
                title=rf"$\gamma={gamma:g}$",
            )
        fig.suptitle(rf"Regression likelihood-ranking selection ({args.regression_condition}, $\epsilon={eps:.2f}$)")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(
            appendix_dir / f"fig_appendix_regression_likelihood_selection_{args.regression_condition}_eps{eps_tag}_gammas.png",
            bbox_inches="tight",
            dpi=220,
        )
        plt.close(fig)

    sim_cases = []
    for gamma in gamma_list:
        try:
            x_sim, _, y_true_sim, samples_sim = load_simulation_outlier_case(
                outputs_root, gamma, args.simulation_scenario, eps, args.rep
            )
            sim_cases.append((gamma, x_sim, _compute_simulation_loglik(x_sim, samples_sim), _eps_to_k_stats(samples_sim, len(x_sim)), y_true_sim))
        except FileNotFoundError:
            continue
    if sim_cases:
        ncols = len(sim_cases)
        fig, axes = plt.subplots(2, ncols, figsize=(5.2 * ncols, 8.0), squeeze=False)
        for col, (gamma, x_sim, ll_sim, k_stats_sim, y_true_sim) in enumerate(sim_cases):
            _plot_rank_panel(axes[0, col], ll_sim, k_stats_sim, title=rf"$\gamma={gamma:g}$")
            _plot_selected_points_panel(axes[1, col], x_sim, np.argsort(ll_sim), k_stats_sim, y_true_sim, title=rf"$\gamma={gamma:g}$")
        fig.suptitle(rf"Simulation likelihood-ranking selection ({args.simulation_scenario}, $\epsilon={eps:.2f}$)")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(
            appendix_dir / f"fig_appendix_simulation_likelihood_selection_{args.simulation_scenario}_eps{eps_tag}_gammas.png",
            bbox_inches="tight",
            dpi=220,
        )
        plt.close(fig)


def _plot_posterior_row(ax_param, ax_eps, case, show_ylabels: bool) -> None:
    for method in METHOD_ORDER:
        if method not in case["methods"]:
            continue
        info = case["methods"][method]
        _draw_2d_kde_contour(ax_param, info["x"], info["y"], METHOD_STYLES[method]["color"])
    ax_param.axvline(case["truth"]["x"], color="black", linestyle="--", linewidth=1.0)
    ax_param.axhline(case["truth"]["y"], color="black", linestyle="--", linewidth=1.0)
    ax_param.set_xlabel(case["labels"]["x"])
    if show_ylabels:
        ax_param.set_ylabel(case["labels"]["y"])
    ax_param.xaxis.label.set_size(15)
    ax_param.yaxis.label.set_size(15)
    ax_param.tick_params(axis="both", labelsize=14)
    ax_param.grid(True, alpha=0.2)

    for method in PROPOSAL_METHODS:
        if method not in case["methods"]:
            continue
        eps_vals = case["methods"][method]["eps"]
        if eps_vals is None:
            continue
        _draw_1d_kde(ax_eps, eps_vals, METHOD_STYLES[method]["color"], method)
    ax_eps.axvline(case["truth"]["eps"], color="black", linestyle="--", linewidth=1.0)
    ax_eps.set_xlabel(r"$\epsilon$")
    if show_ylabels:
        ax_eps.set_ylabel("density")
    ax_eps.xaxis.label.set_size(15)
    ax_eps.yaxis.label.set_size(15)
    ax_eps.tick_params(axis="both", labelsize=14)
    ax_eps.grid(True, alpha=0.2)


def create_main_fig1_posterior(args, outputs_root: Path, main_dir: Path) -> None:
    gammas = parse_float_list(args.posterior_main_gammas) or [0.1, 0.5, 1.0]
    sim_cases = []
    for g in gammas:
        try:
            sim_cases.append(load_simulation_joint_case(args, g))
        except FileNotFoundError:
            continue
    if not sim_cases:
        return
    n = len(sim_cases)

    # Fig1a: simulation posterior only
    fig, axes = plt.subplots(2, n, figsize=(4.8 * n, 7.4), squeeze=False)
    for j in range(n):
        gamma = sim_cases[j]["gamma"]
        _plot_posterior_row(axes[0, j], axes[1, j], sim_cases[j], show_ylabels=(j == 0))
        axes[0, j].set_title(rf"Sim $\gamma={gamma:g}$: param", fontsize=16)
        axes[1, j].set_title(rf"Sim $\gamma={gamma:g}$: $\epsilon$", fontsize=16)
    legend_handles = [
        plt.Line2D([0], [0], color=METHOD_STYLES[m]["color"], lw=2.0, label=m) for m in METHOD_ORDER
    ]
    legend_handles.append(plt.Line2D([0], [0], color="black", lw=1.2, linestyle="--", label="True value"))
    fig.legend(handles=legend_handles, loc="upper center", ncol=len(legend_handles), frameon=True, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(main_dir / "fig1a_simulation_posterior.png", bbox_inches="tight", dpi=220)
    plt.close(fig)

    # Combined Fig1: simulation + regression posterior (2x6, legacy main-text layout).
    reg_cases = []
    for g in gammas:
        try:
            reg_cases.append(load_regression_joint_case(args, g))
        except FileNotFoundError:
            continue
    n_combined = min(len(sim_cases), len(reg_cases))
    if n_combined > 0:
        sim_cases_c = sim_cases[:n_combined]
        reg_cases_c = reg_cases[:n_combined]
        fig, axes = plt.subplots(2, 2 * n_combined, figsize=(4.8 * n_combined * 2, 7.4), squeeze=False)
        for j in range(n_combined):
            gamma = sim_cases_c[j]["gamma"]
            _plot_posterior_row(axes[0, j], axes[1, j], sim_cases_c[j], show_ylabels=(j == 0))
            axes[0, j].set_title(rf"Sim $\gamma={gamma:g}$: param", fontsize=16)
            axes[1, j].set_title(rf"Sim $\gamma={gamma:g}$: $\epsilon$", fontsize=16)

            col = n_combined + j
            _plot_posterior_row(axes[0, col], axes[1, col], reg_cases_c[j], show_ylabels=False)
            axes[0, col].set_title(rf"Reg $\gamma={gamma:g}$: param", fontsize=16)
            axes[1, col].set_title(rf"Reg $\gamma={gamma:g}$: $\epsilon$", fontsize=16)
        legend_handles = [
            plt.Line2D([0], [0], color=METHOD_STYLES[m]["color"], lw=2.0, label=m) for m in METHOD_ORDER
        ]
        legend_handles.append(plt.Line2D([0], [0], color="black", lw=1.2, linestyle="--", label="True value"))
        fig.legend(handles=legend_handles, loc="upper center", ncol=len(legend_handles), frameon=True, fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        fig.savefig(main_dir / "fig1_posterior_regression_simulation.png", bbox_inches="tight", dpi=220)
        plt.close(fig)

    # Fig1b: regression homogeneous + heterogeneous posterior
    # Row1: (beta1, beta2), Row2: intercept, Row3: epsilon.
    hom_cases, het_cases = [], []
    for g in gammas:
        try:
            hom_cases.append(load_regression_joint_case(args, g, condition="hom_mu6"))
        except FileNotFoundError:
            pass
        try:
            het_cases.append(load_regression_joint_case(args, g, condition="het_mu6"))
        except FileNotFoundError:
            pass
    n2 = min(len(hom_cases), len(het_cases))
    if n2 == 0:
        return
    fig, axes = plt.subplots(3, 2 * n2, figsize=(4.8 * n2 * 2, 11.2), squeeze=False)
    for j in range(n2):
        case_h = hom_cases[j]
        # hom: param (2D)
        ax_param_h = axes[0, j]
        for method in METHOD_ORDER:
            if method not in case_h["methods"]:
                continue
            info = case_h["methods"][method]
            _draw_2d_kde_contour(ax_param_h, info["x"], info["y"], METHOD_STYLES[method]["color"])
        ax_param_h.axvline(case_h["truth"]["x"], color="black", linestyle="--", linewidth=1.0)
        ax_param_h.axhline(case_h["truth"]["y"], color="black", linestyle="--", linewidth=1.0)
        ax_param_h.set_xlabel(case_h["labels"]["x"])
        if j == 0:
            ax_param_h.set_ylabel(case_h["labels"]["y"])
        ax_param_h.xaxis.label.set_size(15)
        ax_param_h.yaxis.label.set_size(15)
        ax_param_h.tick_params(axis="both", labelsize=14)
        ax_param_h.grid(True, alpha=0.2)
        # hom: intercept
        ax_int_h = axes[1, j]
        intercept_vals_h = []
        for method in METHOD_ORDER:
            if method not in case_h["methods"]:
                continue
            vals = np.asarray(case_h["methods"][method]["intercept"], dtype=float)
            intercept_vals_h.append(vals)
            _draw_1d_kde(ax_int_h, vals, METHOD_STYLES[method]["color"], method)
        ax_int_h.axvline(case_h["truth"]["intercept"], color="black", linestyle="--", linewidth=1.0)
        if intercept_vals_h:
            concat = np.concatenate(intercept_vals_h)
            lo = min(float(np.percentile(concat, 0.5)), float(case_h["truth"]["intercept"]))
            hi = max(float(np.percentile(concat, 99.5)), float(case_h["truth"]["intercept"]))
            pad = max((hi - lo) * 0.14, 0.15)
            ax_int_h.set_xlim(lo - pad, hi + pad)
        ax_int_h.set_xlabel("intercept")
        if j == 0:
            ax_int_h.set_ylabel("density")
        ax_int_h.xaxis.label.set_size(15)
        ax_int_h.yaxis.label.set_size(15)
        ax_int_h.tick_params(axis="both", labelsize=14)
        ax_int_h.grid(True, alpha=0.2)
        # hom: epsilon
        ax_eps_h = axes[2, j]
        for method in PROPOSAL_METHODS:
            if method not in case_h["methods"]:
                continue
            eps_vals = case_h["methods"][method]["eps"]
            if eps_vals is None:
                continue
            _draw_1d_kde(ax_eps_h, eps_vals, METHOD_STYLES[method]["color"], method)
        ax_eps_h.axvline(case_h["truth"]["eps"], color="black", linestyle="--", linewidth=1.0)
        ax_eps_h.set_xlabel(r"$\epsilon$")
        if j == 0:
            ax_eps_h.set_ylabel("density")
        ax_eps_h.xaxis.label.set_size(15)
        ax_eps_h.yaxis.label.set_size(15)
        ax_eps_h.tick_params(axis="both", labelsize=14)
        ax_eps_h.grid(True, alpha=0.2)
        axes[0, j].set_title(rf"Reg-hom $\gamma={case_h['gamma']:g}$: param", fontsize=14)
        axes[1, j].set_title(rf"Reg-hom $\gamma={case_h['gamma']:g}$: intercept", fontsize=14)
        axes[2, j].set_title(rf"Reg-hom $\gamma={case_h['gamma']:g}$: $\epsilon$", fontsize=14)

        case_t = het_cases[j]
        col = n2 + j
        # het: param (2D)
        ax_param_t = axes[0, col]
        for method in METHOD_ORDER:
            if method not in case_t["methods"]:
                continue
            info = case_t["methods"][method]
            _draw_2d_kde_contour(ax_param_t, info["x"], info["y"], METHOD_STYLES[method]["color"])
        ax_param_t.axvline(case_t["truth"]["x"], color="black", linestyle="--", linewidth=1.0)
        ax_param_t.axhline(case_t["truth"]["y"], color="black", linestyle="--", linewidth=1.0)
        ax_param_t.set_xlabel(case_t["labels"]["x"])
        ax_param_t.xaxis.label.set_size(15)
        ax_param_t.tick_params(axis="both", labelsize=14)
        ax_param_t.grid(True, alpha=0.2)
        # het: intercept
        ax_int_t = axes[1, col]
        intercept_vals_t = []
        for method in METHOD_ORDER:
            if method not in case_t["methods"]:
                continue
            vals = np.asarray(case_t["methods"][method]["intercept"], dtype=float)
            intercept_vals_t.append(vals)
            _draw_1d_kde(ax_int_t, vals, METHOD_STYLES[method]["color"], method)
        ax_int_t.axvline(case_t["truth"]["intercept"], color="black", linestyle="--", linewidth=1.0)
        if intercept_vals_t:
            concat = np.concatenate(intercept_vals_t)
            lo = min(float(np.percentile(concat, 0.5)), float(case_t["truth"]["intercept"]))
            hi = max(float(np.percentile(concat, 99.5)), float(case_t["truth"]["intercept"]))
            pad = max((hi - lo) * 0.14, 0.15)
            ax_int_t.set_xlim(lo - pad, hi + pad)
        ax_int_t.set_xlabel("intercept")
        ax_int_t.xaxis.label.set_size(15)
        ax_int_t.tick_params(axis="both", labelsize=14)
        ax_int_t.grid(True, alpha=0.2)
        # het: epsilon
        ax_eps_t = axes[2, col]
        for method in PROPOSAL_METHODS:
            if method not in case_t["methods"]:
                continue
            eps_vals = case_t["methods"][method]["eps"]
            if eps_vals is None:
                continue
            _draw_1d_kde(ax_eps_t, eps_vals, METHOD_STYLES[method]["color"], method)
        ax_eps_t.axvline(case_t["truth"]["eps"], color="black", linestyle="--", linewidth=1.0)
        ax_eps_t.set_xlabel(r"$\epsilon$")
        ax_eps_t.xaxis.label.set_size(15)
        ax_eps_t.tick_params(axis="both", labelsize=14)
        ax_eps_t.grid(True, alpha=0.2)
        axes[0, col].set_title(rf"Reg-het $\gamma={case_t['gamma']:g}$: param", fontsize=14)
        axes[1, col].set_title(rf"Reg-het $\gamma={case_t['gamma']:g}$: intercept", fontsize=14)
        axes[2, col].set_title(rf"Reg-het $\gamma={case_t['gamma']:g}$: $\epsilon$", fontsize=14)
    legend_handles = [
        plt.Line2D([0], [0], color=METHOD_STYLES[m]["color"], lw=2.0, label=m) for m in METHOD_ORDER
    ]
    legend_handles.append(plt.Line2D([0], [0], color="black", lw=1.2, linestyle="--", label="True value"))
    fig.legend(handles=legend_handles, loc="upper center", ncol=len(legend_handles), frameon=True, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(main_dir / "fig1b_regression_posterior_hom_het.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def create_main_fig2_accuracy(args, outputs_root: Path, main_dir: Path, df_sim: pd.DataFrame, df_reg_eps: pd.DataFrame, df_reg_final: pd.DataFrame) -> None:
    df_sim_g = df_sim[
        np.isclose(df_sim["gamma_tag"], float(args.simulation_main_gamma))
        & (df_sim["scenario"] == args.simulation_scenario)
    ].copy()
    df_reg_g = df_reg_eps[np.isclose(df_reg_eps["gamma_tag"], float(args.regression_main_gamma))].copy()
    if df_sim_g.empty or df_reg_g.empty:
        return
    fig = plt.figure(figsize=(16.0, 9.0))
    gs = fig.add_gridspec(2, 3)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2], projection="3d")
    eps = float(args.posterior_eps)

    methods_sim = available_methods(df_sim_g)
    methods_reg = available_methods(df_reg_g)
    # Row 1 = simulation
    plot_lines(
        ax00,
        df_sim_g,
        "eps_true",
        "mu_bias_norm_mean",
        methods_sim,
        yerr_col="mu_bias_norm_std",
        title=r"Simulation parameter error ($\|\hat{\mu}-\mu^\star\|_2$)",
        xlabel="True epsilon",
        ylabel=r"Parameter error ($L_2$)",
    )
    plot_lines(
        ax01,
        df_sim_g,
        "eps_true",
        "eps_post_mean_mean",
        methods_sim,
        yerr_col="eps_post_mean_std",
        title="Simulation epsilon estimation",
        xlabel="True epsilon",
        ylabel="Posterior mean epsilon",
        proposal_only=True,
        identity=True,
    )
    try:
        x_sim, p_out_sim, y_true_sim, _ = load_simulation_outlier_case(
            outputs_root, float(args.simulation_main_gamma), args.simulation_scenario, eps, args.rep
        )
        sc2 = plot_simulation_outlier_2d(
            ax02,
            x_sim,
            p_out_sim,
            y_true_sim,
            np.empty((0, 0)),
            title=rf"Simulation data/outlier prob ($\gamma={float(args.simulation_main_gamma):g}$)",
        )
        fig.colorbar(sc2, ax=ax02)
    except FileNotFoundError:
        pass

    # Row 2 = regression
    plot_lines(
        ax10,
        df_reg_g,
        "eps_true",
        "rmse_test_mean",
        methods_reg,
        yerr_col="rmse_test_std",
        title="Regression predictive error (RMSE)",
        xlabel="True epsilon",
        ylabel="Predictive error (RMSE)",
    )
    plot_lines(
        ax11,
        df_reg_g,
        "eps_true",
        "eps_post_mean_mean",
        methods_reg,
        yerr_col="eps_post_mean_std",
        title="Regression epsilon estimation",
        xlabel="True epsilon",
        ylabel="Posterior mean epsilon",
        proposal_only=True,
        identity=True,
    )
    try:
        x_train, y_train, p_out_reg, mask_reg, _, _ = load_regression_outlier_case(
            outputs_root, float(args.regression_main_gamma), "hom_mu6", eps, args.rep
        )
        sc = plot_regression_outlier_3d(
            ax12,
            x_train,
            y_train,
            p_out_reg,
            mask_reg,
            np.empty((0, 0)),
            {},
            title=rf"Regression data/outlier prob ($\gamma={float(args.regression_main_gamma):g}$)",
        )
        fig.colorbar(sc, ax=ax12, shrink=0.75, pad=0.08)
    except FileNotFoundError:
        pass

    for ax in [ax00, ax01, ax10, ax11, ax12]:
        ax.tick_params(axis="both", labelsize=15)
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)
        title = ax.title
        if title is not None:
            title.set_fontsize(17)
    ax02.tick_params(axis="both", labelsize=14)
    ax02.xaxis.label.set_size(15)
    ax02.yaxis.label.set_size(15)
    ax02.title.set_fontsize(16)
    if hasattr(ax02, "zaxis"):
        ax02.zaxis.label.set_size(15)
        ax02.zaxis.set_tick_params(labelsize=14)
    for ax in [ax00, ax01, ax10, ax11]:
        leg = ax.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_fontsize(12)
    fig.tight_layout()
    fig.savefig(main_dir / "fig2_accuracy_regression_simulation.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def create_main_fig3_likelihood(args, outputs_root: Path, main_dir: Path) -> None:
    eps = float(args.posterior_eps)
    fig = plt.figure(figsize=(28.0, 5.2))
    ax1 = fig.add_subplot(1, 6, 1)
    ax2 = fig.add_subplot(1, 6, 2)
    ax3 = fig.add_subplot(1, 6, 3)
    ax4 = fig.add_subplot(1, 6, 4, projection="3d")
    ax5 = fig.add_subplot(1, 6, 5)
    ax6 = fig.add_subplot(1, 6, 6, projection="3d")
    try:
        x_sim, _, y_true_sim, samples_sim = load_simulation_outlier_case(
            outputs_root, float(args.simulation_main_gamma), args.simulation_scenario, eps, args.rep
        )
        ll_sim = _compute_simulation_loglik(x_sim, samples_sim)
        ks_sim = _eps_to_k_stats(samples_sim, len(x_sim))
        _plot_rank_panel(ax1, ll_sim, ks_sim, title=rf"Log-likelihood ranking (Simulation; $\gamma={float(args.simulation_main_gamma):g}$)")
        _plot_selected_points_panel(ax2, x_sim, np.argsort(ll_sim), ks_sim, y_true_sim, title="Simulation selected points")
    except FileNotFoundError:
        pass
    try:
        x_reg, y_reg, _, m_reg, s_reg, _ = load_regression_outlier_case(
            outputs_root, float(args.regression_main_gamma), "hom_mu6", eps, args.rep
        )
        ll_reg = _compute_regression_loglik(x_reg, y_reg, s_reg)
        ks_reg = _eps_to_k_stats(s_reg, len(x_reg))
        _plot_rank_panel(ax3, ll_reg, ks_reg, title=rf"Log-likelihood ranking (Reg-hom; $\gamma={float(args.regression_main_gamma):g}$)")
        _plot_selected_points_panel_3d(ax4, x_reg, y_reg, np.argsort(ll_reg), ks_reg, m_reg, title="Reg-hom selected points")
    except FileNotFoundError:
        pass
    try:
        xh, yh, _, mh, sh, _ = load_regression_outlier_case(
            outputs_root, float(args.regression_main_gamma), "het_mu6", eps, args.rep
        )
        llh = _compute_regression_loglik(xh, yh, sh)
        ksh = _eps_to_k_stats(sh, len(xh))
        _plot_rank_panel(ax5, llh, ksh, title=rf"Log-likelihood ranking (Reg-het; $\gamma={float(args.regression_main_gamma):g}$)")
        _plot_selected_points_panel_3d(
            ax6, xh, yh, np.argsort(llh), ksh, mh, title="Reg-het selected points", swap_xy=True
        )
    except FileNotFoundError:
        pass
    for ax in [ax2, ax4, ax6]:
        ax.tick_params(axis="both", labelsize=15)
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)
        ax.title.set_fontsize(16)
    leg2 = ax2.get_legend()
    if leg2 is not None:
        for txt in leg2.get_texts():
            txt.set_fontsize(12)
    leg4 = ax4.get_legend()
    if leg4 is not None:
        for txt in leg4.get_texts():
            txt.set_fontsize(12)
    leg6 = ax6.get_legend()
    if leg6 is not None:
        for txt in leg6.get_texts():
            txt.set_fontsize(12)
    ax4.zaxis.set_tick_params(labelsize=14)
    ax4.zaxis.label.set_size(15)
    ax6.zaxis.set_tick_params(labelsize=14)
    ax6.zaxis.label.set_size(15)
    fig.tight_layout()
    fig.savefig(main_dir / "fig3_likelihood_selection_hom_het.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def _mu_posterior_axis_limits(case: dict, methods: list[str]) -> tuple[float, float, float, float]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for method in methods:
        if method not in case["methods"]:
            continue
        xs.append(np.asarray(case["methods"][method]["x"], dtype=float))
        ys.append(np.asarray(case["methods"][method]["y"], dtype=float))
    if not xs:
        return -1.0, 1.0, -1.0, 1.0
    all_x = np.concatenate(xs)
    all_y = np.concatenate(ys)
    truth_x = float(case["truth"]["x"])
    truth_y = float(case["truth"]["y"])
    xmin = min(float(np.percentile(all_x, 1)), truth_x)
    xmax = max(float(np.percentile(all_x, 99)), truth_x)
    ymin = min(float(np.percentile(all_y, 1)), truth_y)
    ymax = max(float(np.percentile(all_y, 99)), truth_y)
    xpad = max((xmax - xmin) * 0.12, 0.25)
    ypad = max((ymax - ymin) * 0.12, 0.25)
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad


def _draw_simulation_mu_posterior_overlay(
    ax,
    case: dict,
    methods: list[str],
    *,
    title: str,
    show_ylabel: bool,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    for method in methods:
        if method not in case["methods"]:
            continue
        info = case["methods"][method]
        _draw_2d_kde_contour(ax, info["x"], info["y"], METHOD_STYLES[method]["color"])
    ax.axvline(case["truth"]["x"], color="black", linestyle="--", linewidth=1.1)
    ax.axhline(case["truth"]["y"], color="black", linestyle="--", linewidth=1.1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r"$\mu_1$")
    if show_ylabel:
        ax.set_ylabel(r"$\mu_2$")
    ax.set_title(title, fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal", adjustable="box")


def generate_intro_simulation_figures(args, outputs_root: Path, intro_dir: Path) -> None:
    """Export one 1x3 intro figure for Gaussian density-estimation simulation."""
    gamma = float(args.simulation_main_gamma)
    eps = float(args.posterior_eps)
    eps_tag = format_eps_tag(eps)
    gamma_tag = format_gamma_tag(gamma)
    scenario = args.simulation_scenario
    tag = f"{scenario}_gamma{gamma_tag}_eps{eps_tag}"
    compare_methods = ["DPD", "NLL"]

    try:
        case = load_simulation_joint_case(args, gamma)
    except FileNotFoundError as exc:
        print(f"Skipping intro simulation figure: {exc}")
        return
    if not all(m in case["methods"] for m in compare_methods):
        print("Skipping intro simulation figure: missing DPD or NLL samples.")
        return
    if "DPD" not in case["methods"] or case["methods"]["DPD"]["eps"] is None:
        print("Skipping intro simulation figure: missing DPD epsilon samples.")
        return
    eps_vals = np.asarray(case["methods"]["DPD"]["eps"], dtype=float)
    if eps_vals.size == 0:
        print("Skipping intro simulation figure: empty DPD epsilon samples.")
        return
    try:
        x_sim, p_out_sim, y_true_sim, _ = load_simulation_outlier_case(
            outputs_root, gamma, scenario, eps, args.rep
        )
    except FileNotFoundError as exc:
        print(f"Skipping intro simulation figure: {exc}")
        return

    truth_eps = float(case["truth"]["eps"])
    xlim_lo, xlim_hi, ylim_lo, ylim_hi = _mu_posterior_axis_limits(case, compare_methods)
    xlim = (xlim_lo, xlim_hi)
    ylim = (ylim_lo, ylim_hi)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), squeeze=False)
    ax_mu, ax_eps, ax_out = axes[0]

    _draw_simulation_mu_posterior_overlay(
        ax_mu,
        case,
        compare_methods,
        title="Model-parameter posterior",
        show_ylabel=True,
        xlim=xlim,
        ylim=ylim,
    )
    _draw_1d_kde(ax_eps, eps_vals, METHOD_STYLES["DPD"]["color"], "DPD")
    ax_eps.axvline(
        truth_eps, color="black", linestyle="--", linewidth=1.2, label=rf"True $\epsilon^\star$"
    )
    lo = min(float(np.percentile(eps_vals, 0.5)), truth_eps)
    hi = max(float(np.percentile(eps_vals, 99.5)), truth_eps)
    pad = max((hi - lo) * 0.14, 0.02)
    ax_eps.set_xlim(lo - pad, hi + pad)
    ax_eps.set_xlabel(r"$\epsilon$")
    ax_eps.set_ylabel("density")
    ax_eps.set_title(r"Contamination-rate posterior", fontsize=15)
    ax_eps.tick_params(axis="both", labelsize=13)
    ax_eps.xaxis.label.set_size(14)
    ax_eps.yaxis.label.set_size(14)
    ax_eps.grid(True, alpha=0.2)

    sc = plot_simulation_outlier_2d(
        ax_out,
        x_sim,
        p_out_sim,
        y_true_sim,
        np.empty((0, 0)),
        title="Probabilistic outlier detection",
    )
    ax_out.set_xlabel(r"$x_1$")
    ax_out.set_ylabel(r"$x_2$")
    ax_out.tick_params(axis="both", labelsize=13)
    ax_out.xaxis.label.set_size(14)
    ax_out.yaxis.label.set_size(14)
    ax_out.title.set_fontsize(15)
    cbar = fig.colorbar(sc, ax=ax_out, shrink=0.82, pad=0.02)
    cbar.set_label("outlier probability", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    intro_legend_fs = 13
    ax_mu.legend(
        handles=[
            plt.Line2D([0], [0], color=METHOD_STYLES["DPD"]["color"], lw=2.2, label="DPD"),
            plt.Line2D([0], [0], color=METHOD_STYLES["NLL"]["color"], lw=2.2, label="NLL"),
            plt.Line2D([0], [0], color="black", lw=1.2, linestyle="--", label=r"True $\mu^\star$"),
        ],
        loc="lower right",
        frameon=True,
        fontsize=intro_legend_fs,
    )
    ax_eps.legend(loc="upper right", frameon=True, fontsize=intro_legend_fs)
    ax_out.legend(
        handles=[
            plt.Line2D(
                [0], [0], marker="o", color="black", markerfacecolor="none",
                markersize=9, linewidth=0, label="True outlier",
            ),
        ],
        loc="upper left",
        frameon=True,
        fontsize=intro_legend_fs,
    )
    fig.tight_layout()
    fig.savefig(intro_dir / f"fig_intro_simulation_{tag}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def export_main_text_final_set(args, outputs_root: Path, main_dir: Path, df_sim: pd.DataFrame, df_reg_eps: pd.DataFrame, df_reg_final: pd.DataFrame) -> None:
    """Export fixed 3-figure main-text set + caption draft."""
    create_main_fig1_posterior(args, outputs_root, main_dir)
    create_main_fig2_accuracy(args, outputs_root, main_dir, df_sim, df_reg_eps, df_reg_final)
    create_main_fig3_likelihood(args, outputs_root, main_dir)

    caption_path = main_dir / "main_figure_captions_draft.md"
    caption_text = f"""# Main Text Figure Captions (Draft)

## Fig.1a Posterior (Simulation)
Posterior visualization for simulation parameters and epsilon across three gamma settings.

## Fig.1b Posterior (Regression; homogeneous + heterogeneous)
Posterior visualization for regression parameters and epsilon across homogeneous and heterogeneous contamination settings.

## Fig.2 Accuracy Views (Regression + Simulation, 2x3)
Accuracy-related summary panels including predictive/parameter error, epsilon estimation, and representative data with outlier probabilities.

## Fig.3 Likelihood Ranking and Selection (1x6)
Likelihood ranking and epsilon-posterior 95% interval are mapped to concrete outlier selections in data space, shown for simulation, regression-homogeneous, and regression-heterogeneous settings.
"""
    caption_path.write_text(caption_text, encoding="utf-8")


def generate_posterior_grids(args, output_dir: Path, sim_available: list[float], reg_available: list[float]) -> None:
    main_dir = output_dir / "main_text"
    appendix_dir = output_dir / "appendix"
    common_available = [g for g in sim_available if any(math.isclose(g, r, rel_tol=0.0, abs_tol=1e-12) for r in reg_available)]
    if not common_available:
        return
    main_gammas = select_gammas(parse_float_list(args.posterior_main_gammas), common_available, fallback=[0.1, 0.5, 1.0])
    appendix_gammas = select_gammas(parse_float_list(args.posterior_appendix_gammas), common_available, fallback=common_available)
    sim_main_cases = [load_simulation_joint_case(args, g) for g in main_gammas]
    reg_main_cases = [load_regression_joint_case(args, g) for g in main_gammas]
    sim_appendix_cases = [load_simulation_joint_case(args, g) for g in appendix_gammas]
    reg_appendix_cases = [load_regression_joint_case(args, g) for g in appendix_gammas]
    eps_tag = f"{args.posterior_eps:.2f}".replace(".", "p")
    plot_joint_posterior_comparison(
        sim_main_cases,
        title=rf"Simulation posterior summaries ({args.simulation_scenario}, $\epsilon={args.posterior_eps:.2f}$)",
        output_path=main_dir / f"fig_simulation_posterior_main_{args.simulation_scenario}_eps{eps_tag}.png",
    )
    plot_joint_posterior_comparison(
        reg_main_cases,
        title=rf"Regression posterior summaries ({args.regression_condition}, $\epsilon={args.posterior_eps:.2f}$)",
        output_path=main_dir / f"fig_regression_posterior_main_{args.regression_condition}_eps{eps_tag}.png",
    )
    plot_joint_posterior_comparison(
        sim_appendix_cases,
        title=rf"Simulation appendix posterior grid ({args.simulation_scenario}, $\epsilon={args.posterior_eps:.2f}$)",
        output_path=appendix_dir / f"fig_simulation_posterior_appendix_{args.simulation_scenario}_eps{eps_tag}.png",
    )
    plot_joint_posterior_comparison(
        reg_appendix_cases,
        title=rf"Regression appendix posterior grid ({args.regression_condition}, $\epsilon={args.posterior_eps:.2f}$)",
        output_path=appendix_dir / f"fig_regression_posterior_appendix_{args.regression_condition}_eps{eps_tag}.png",
    )


def main():
    parser = argparse.ArgumentParser(description="Generate main-text and appendix figures for exp1 paper results.")
    parser.add_argument("--outputs_root", default="outputs")
    parser.add_argument("--output_dir", default="outputs/paper_exp1_suite")
    parser.add_argument("--simulation_main_gamma", type=float, default=0.5)
    parser.add_argument("--regression_main_gamma", type=float, default=1.0)
    parser.add_argument("--simulation_scenario", default="A", choices=["A", "B"])
    parser.add_argument("--regression_condition", default="hom_mu6")
    parser.add_argument("--posterior_eps", type=float, default=0.2)
    parser.add_argument("--posterior_main_gammas", default="0.1 0.5 1.0")
    parser.add_argument("--posterior_appendix_gammas", default=None)
    parser.add_argument("--outlier_vis_gammas", default="0.1 0.5 1.0")
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--simulation_n", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--num_warmup", type=int, default=2000)
    parser.add_argument("--num_samples", type=int, default=4000)
    parser.add_argument("--eta_mode", default="covdet")
    parser.add_argument("--target_accept", type=float, default=0.8)
    parser.add_argument("--init_step", type=float, default=1.0)
    parser.add_argument("--no_standardize_simulation", action="store_true")
    parser.add_argument("--out_dist", default="gaussian")
    parser.add_argument("--out_mean_x", type=float, default=5.0)
    parser.add_argument("--out_mean_y", type=float, default=5.0)
    parser.add_argument("--out_scale_x", type=float, default=1.0)
    parser.add_argument("--out_scale_y", type=float, default=1.0)
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    output_dir = Path(args.output_dir)
    main_dir = output_dir / "main_text"
    appendix_dir = output_dir / "appendix"
    intro_dir = output_dir / "intro"
    ensure_dir(main_dir)
    ensure_dir(appendix_dir)
    ensure_dir(intro_dir)

    df_sim = load_gamma_summaries(outputs_root, "exp1_simulation_gamma_sweep_gamma")
    df_reg_final = load_gamma_summaries(outputs_root, "exp1_regression_final_response_huber_gamma")
    if df_reg_final.empty:
        df_reg_final = load_gamma_summaries(outputs_root, "exp1_regression_final_gamma")
    df_reg_eps = load_gamma_summaries(outputs_root, "exp1_regression_eps_sweep_response_homogeneous_mu6_gamma")
    if df_reg_eps.empty:
        df_reg_eps = load_gamma_summaries(outputs_root, "exp1_regression_eps_sweep_mu6_gamma")

    if not df_sim.empty:
        plot_main_simulation_figure(
            df_sim,
            gamma=args.simulation_main_gamma,
            scenario=args.simulation_scenario,
            output_path=main_dir / f"fig_main_simulation_summary_{args.simulation_scenario}_gamma{format_gamma_tag(args.simulation_main_gamma)}.png",
        )
        sim_gammas = sorted(df_sim["gamma_tag"].unique())
        df_sim_scenario = df_sim[df_sim["scenario"] == args.simulation_scenario]
        sim_methods = available_methods(df_sim_scenario)
        plot_gamma_grid(
            df_sim_scenario,
            sim_gammas,
            "eps_true",
            "auprc_mean",
            "auprc_std",
            sim_methods,
            output_path=appendix_dir / f"fig_appendix_simulation_auprc_{args.simulation_scenario}.png",
            title_prefix=f"Simulation AUPRC ({args.simulation_scenario})",
            xlabel="True contamination epsilon",
            ylabel="AUPRC",
        )
        plot_gamma_grid(
            df_sim_scenario,
            sim_gammas,
            "eps_true",
            "eps_post_mean_mean",
            "eps_post_mean_std",
            sim_methods,
            output_path=appendix_dir / f"fig_appendix_simulation_eps_estimation_{args.simulation_scenario}.png",
            title_prefix=f"Simulation epsilon estimation ({args.simulation_scenario})",
            xlabel="True contamination epsilon",
            ylabel="Posterior mean epsilon",
            proposal_only=True,
            identity=True,
        )
        plot_gamma_grid(
            df_sim_scenario,
            sim_gammas,
            "eps_true",
            "eps_ci95_cover_mean",
            None,
            sim_methods,
            output_path=appendix_dir / f"fig_appendix_simulation_eps_coverage_{args.simulation_scenario}.png",
            title_prefix=f"Simulation epsilon coverage ({args.simulation_scenario})",
            xlabel="True contamination epsilon",
            ylabel="Coverage probability",
            proposal_only=True,
        )

    if not df_reg_eps.empty:
        plot_main_regression_performance_figure(
            df_reg_eps,
            gamma=args.regression_main_gamma,
            output_path=main_dir / f"fig_main_regression_performance_gamma{format_gamma_tag(args.regression_main_gamma)}.png",
        )
        plot_main_regression_epsilon_figure(
            df_reg_eps,
            gamma=args.regression_main_gamma,
            output_path=main_dir / f"fig_main_regression_eps_gamma{format_gamma_tag(args.regression_main_gamma)}.png",
        )
        reg_eps_gammas = sorted(df_reg_eps["gamma_tag"].unique())
        reg_eps_methods = available_methods(df_reg_eps)
        plot_gamma_grid(
            df_reg_eps,
            reg_eps_gammas,
            "eps_true",
            "rmse_test_mean",
            "rmse_test_std",
            reg_eps_methods,
            output_path=appendix_dir / "fig_appendix_regression_eps_rmse.png",
            title_prefix="Regression eps sweep RMSE",
            xlabel="True contamination epsilon",
            ylabel="Test RMSE",
        )
        plot_gamma_grid(
            df_reg_eps,
            reg_eps_gammas,
            "eps_true",
            "auprc_mean",
            "auprc_std",
            reg_eps_methods,
            output_path=appendix_dir / "fig_appendix_regression_eps_auprc.png",
            title_prefix="Regression eps sweep AUPRC",
            xlabel="True contamination epsilon",
            ylabel="AUPRC",
        )
        plot_gamma_grid(
            df_reg_eps,
            reg_eps_gammas,
            "eps_true",
            "eps_post_mean_mean",
            "eps_post_mean_std",
            reg_eps_methods,
            output_path=appendix_dir / "fig_appendix_regression_eps_estimation.png",
            title_prefix="Regression eps sweep epsilon estimation",
            xlabel="True contamination epsilon",
            ylabel="Posterior mean epsilon",
            proposal_only=True,
            identity=True,
        )
        plot_gamma_grid(
            df_reg_eps,
            reg_eps_gammas,
            "eps_true",
            "eps_ci95_cover_mean",
            None,
            reg_eps_methods,
            output_path=appendix_dir / "fig_appendix_regression_eps_coverage.png",
            title_prefix="Regression eps sweep epsilon coverage",
            xlabel="True contamination epsilon",
            ylabel="Coverage probability",
            proposal_only=True,
        )

    if not df_reg_final.empty:
        reg_final_gammas = sorted(df_reg_final["gamma_tag"].unique())
        reg_final_methods = available_methods(df_reg_final)
        plot_gamma_grid(
            df_reg_final,
            reg_final_gammas,
            "condition",
            "rmse_test_mean",
            "rmse_test_std",
            reg_final_methods,
            output_path=appendix_dir / "fig_appendix_regression_conditions_rmse.png",
            title_prefix="Regression conditions RMSE",
            xlabel="Condition",
            ylabel="Test RMSE",
        )
        plot_gamma_grid(
            df_reg_final,
            reg_final_gammas,
            "condition",
            "beta_l2_error_mean",
            "beta_l2_error_std",
            reg_final_methods,
            output_path=appendix_dir / "fig_appendix_regression_conditions_beta_l2.png",
            title_prefix="Regression conditions beta L2",
            xlabel="Condition",
            ylabel="Beta L2 error",
        )

    sim_available = [gamma for gamma, _ in discover_gamma_dirs(str(outputs_root), "exp1_simulation_gamma_sweep_gamma")]
    reg_available = [gamma for gamma, _ in discover_gamma_dirs(str(outputs_root), "exp1_regression_final_response_huber_gamma")]
    if not reg_available:
        reg_available = [gamma for gamma, _ in discover_gamma_dirs(str(outputs_root), "exp1_regression_final_gamma")]
    if sim_available and reg_available:
        generate_posterior_grids(args, output_dir, sim_available, reg_available)

    generate_outlier_visualizations(args, outputs_root, main_dir, appendix_dir)
    generate_likelihood_ranking_visualizations(args, outputs_root, main_dir, appendix_dir)
    generate_intro_simulation_figures(args, outputs_root, intro_dir)
    export_main_text_final_set(args, outputs_root, main_dir, df_sim, df_reg_eps, df_reg_final)
    plot_appendix_simulation_scale_posteriors(args, appendix_dir)
    plot_appendix_regression_intercept_sigma_posteriors(args, appendix_dir, condition="hom_mu6")
    plot_appendix_regression_intercept_sigma_posteriors(args, appendix_dir, condition="het_mu6")
    plot_simulation_kappa_appendix(outputs_root, appendix_dir)
    plot_regression_kappa_appendix(outputs_root, appendix_dir)

    print(f"Saved paper figure suite to {output_dir}")


if __name__ == "__main__":
    main()
