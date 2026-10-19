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
from hb_experiments.figure_export import save_figure

from experiments.paper_plot_paths import (
    discover_gamma_dirs,
    parse_float_list,
    select_gammas,
)
from hb_experiments.coordinates import (
    recover_posterior_params_original_scale,
    standardize_regression_data,
)


METHOD_ORDER = ["DPD", "ExpSatDPD", "RatSatDPD", "NLL", "GammaDiv", "StudentT"]
PROPOSAL_METHODS = ["DPD", "ExpSatDPD", "RatSatDPD"]
MAIN_METHODS = ["DPD", "NLL", "GammaDiv", "StudentT"]
METHOD_DISPLAY = {
    "DPD": "H-B (Ours)",
    "ExpSatDPD": "H-B ExpSat",
    "RatSatDPD": "H-B RatSat",
    "NLL": "Standard Bayes",
    "GammaDiv": r"$\gamma$-divergence",
    "StudentT": "Student-t",
}
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
LEGEND_FONTSIZE = 15
TITLE_FONTSIZE = 17
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 13


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
            label=METHOD_DISPLAY.get(method, method),
        )
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, fontsize=LEGEND_FONTSIZE)


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
    save_figure(fig, output_path, bbox_inches="tight", dpi=200)
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
    save_figure(fig, output_path, bbox_inches="tight", dpi=200)
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
    save_figure(fig, output_path, bbox_inches="tight", dpi=200)
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
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=True, fontsize=LEGEND_FONTSIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output_path, bbox_inches="tight", dpi=200)
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
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xticks(np.arange(len(col_labels)), [f"{g:g}" for g in col_labels], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), [f"{k:g}" for k in row_labels])
    ax.set_xlabel("gamma", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("kappa", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
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
    fig.suptitle(title, fontsize=TITLE_FONTSIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, output_path, bbox_inches="tight", dpi=200)
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


def _assert_matching_regression_dataset(primary, fallback, primary_path, fallback_path):
    """Reject posterior overlays unless both files describe the same training data."""
    required = ("x_train", "y_train", "mask_outlier", "beta_true")
    for key in required:
        if key not in primary or key not in fallback:
            raise ValueError(
                f"Cannot verify a paired posterior comparison: missing {key!r} in "
                f"{primary_path} or {fallback_path}. Store the shared training data "
                "and truth with each posterior sample file."
            )
        a = np.asarray(primary[key])
        b = np.asarray(fallback[key])
        if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
            raise ValueError(
                f"Cannot overlay regression posteriors from different datasets "
                f"({key} differs): {primary_path} versus {fallback_path}. "
                "Fit all compared methods to the same observations and use that "
                "dataset's own response/covariate transformation."
            )
    if "sigma_true" in primary or "sigma_true" in fallback:
        if "sigma_true" not in primary or "sigma_true" not in fallback:
            raise ValueError("Cannot verify matching regression truth: sigma_true is missing.")
        if not np.array_equal(np.asarray(primary["sigma_true"]), np.asarray(fallback["sigma_true"]), equal_nan=True):
            raise ValueError("Cannot overlay regression posteriors: sigma_true differs.")


def load_regression_joint_case(args, gamma: float):
    outputs_root = Path(args.outputs_root)
    sample_path = None
    for g, path in discover_gamma_dirs(str(outputs_root), "exp1_regression_eps_sweep_mu6_gamma"):
        if math.isclose(float(g), float(gamma), rel_tol=0.0, abs_tol=1e-12):
            candidate = path / f"eps{args.posterior_eps:.2f}" / f"rep{args.rep:03d}" / "samples.npz"
            if candidate.exists():
                sample_path = candidate
            break
    for g, path in ([] if sample_path is not None else discover_gamma_dirs(str(outputs_root), "exp1_regression_final_gamma")):
        if math.isclose(float(g), float(gamma), rel_tol=0.0, abs_tol=1e-12):
            candidate = path / args.regression_condition / f"rep{args.rep:03d}" / "samples.npz"
            if candidate.exists():
                sample_path = candidate
            break
    # Fall back to eps-sweep samples when the requested gamma is unavailable under final/.
    if sample_path is None:
        eps_dir = find_gamma_dir(
            outputs_root, f"exp1_regression_eps_sweep_{args.regression_condition}_gamma", gamma
        )
        sample_path = eps_dir / f"eps{args.posterior_eps:.2f}" / f"rep{args.rep:03d}" / "samples.npz"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing samples file: {sample_path}")
    raw = np.load(sample_path, allow_pickle=True)
    raw_keys = set(raw.files)
    missing_proposals = {"samples_exp_sat_dpd", "samples_rat_sat_dpd"} - raw_keys
    fallback_raw = None
    if missing_proposals:
        try:
            eps_dir = find_gamma_dir(
                outputs_root, f"exp1_regression_eps_sweep_{args.regression_condition}_gamma", gamma
            )
            eps_sample = eps_dir / f"eps{args.posterior_eps:.2f}" / f"rep{args.rep:03d}" / "samples.npz"
            if eps_sample.exists() and eps_sample.resolve() != sample_path.resolve():
                fallback_raw = np.load(eps_sample, allow_pickle=True)
        except FileNotFoundError:
            fallback_raw = None

    x_train = np.asarray(raw["x_train"])
    y_train = np.asarray(raw["y_train"])
    _, _, _, scaler = standardize_regression_data(x_train, y_train, x_train)
    beta_true = np.asarray(raw["beta_true"], dtype=float)
    method_samples = _extract_method_samples(raw)
    method_sources = {method: raw for method in method_samples}
    if fallback_raw is not None:
        fallback_samples = _extract_method_samples(fallback_raw)
        augmenting = any(method not in method_samples and method in fallback_samples
                         for method in ("ExpSatDPD", "RatSatDPD"))
        if augmenting:
            _assert_matching_regression_dataset(raw, fallback_raw, sample_path, eps_sample)
        for method in ("ExpSatDPD", "RatSatDPD"):
            if method not in method_samples and method in fallback_samples:
                method_samples[method] = fallback_samples[method]
                method_sources[method] = fallback_raw

    methods = {}
    for method, arr in method_samples.items():
        transformed = recover_posterior_params_original_scale(np.asarray(arr, dtype=float), scaler)
        beta = np.asarray(transformed["beta"], dtype=float)
        if beta.shape[1] < 3:
            continue
        methods[method] = {
            "x": beta[:, 1],
            "y": beta[:, 2],
            "eps": _proposal_epsilon_samples(method_sources[method], method, arr),
        }
    return {
        "gamma": float(gamma),
        "methods": methods,
        "truth": {"x": float(beta_true[1]), "y": float(beta_true[2]), "eps": float(args.posterior_eps)},
        "labels": {"x": r"$\beta_1$", "y": r"$\beta_2$"},
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
    """Appendix posterior gamma-sensitivity plot using the main method set only."""
    if not cases:
        return
    ncols = len(cases)
    fig, axes = plt.subplots(2, ncols, figsize=(4.3 * ncols, 8.2), squeeze=False)
    method_handles = {}
    for col, case in enumerate(cases):
        gamma = case["gamma"]
        ax_joint = axes[0, col]
        ax_eps = axes[1, col]
        for method in MAIN_METHODS:
            if method not in case["methods"]:
                continue
            info = case["methods"][method]
            color = METHOD_STYLES[method]["color"]
            _draw_2d_kde_contour(ax_joint, info["x"], info["y"], color)
            method_handles.setdefault(
                method,
                plt.Line2D([0], [0], color=color, lw=1.8, label=METHOD_DISPLAY.get(method, method)),
            )
        ax_joint.axvline(case["truth"]["x"], color="black", linestyle="--", linewidth=1.1)
        ax_joint.axhline(case["truth"]["y"], color="black", linestyle="--", linewidth=1.1)
        ax_joint.set_title(rf"$\gamma={gamma:g}$", fontsize=TITLE_FONTSIZE)
        ax_joint.set_xlabel(case["labels"]["x"], fontsize=LABEL_FONTSIZE)
        if col == 0:
            ax_joint.set_ylabel(case["labels"]["y"], fontsize=LABEL_FONTSIZE)
        ax_joint.grid(True, alpha=0.2)

        if "DPD" in case["methods"] and case["methods"]["DPD"]["eps"] is not None:
            _draw_1d_kde(
                ax_eps,
                case["methods"]["DPD"]["eps"],
                METHOD_STYLES["DPD"]["color"],
                METHOD_DISPLAY["DPD"],
            )
        ax_eps.axvline(case["truth"]["eps"], color="black", linestyle="--", linewidth=1.1)
        ax_eps.set_xlabel(r"$\epsilon$", fontsize=LABEL_FONTSIZE)
        if col == 0:
            ax_eps.set_ylabel("Posterior density", fontsize=LABEL_FONTSIZE)
        ax_eps.grid(True, alpha=0.2)

    handles = list(method_handles.values())
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--", lw=1.2, label="True value"))
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 5), frameon=True, fontsize=LEGEND_FONTSIZE)
    fig.suptitle(title, fontsize=TITLE_FONTSIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output_path, bbox_inches="tight", dpi=220)
    plt.close(fig)

def load_regression_outlier_case(outputs_root: Path, gamma: float, condition: str, eps: float, rep: int):
    gamma_dir = find_gamma_dir(outputs_root, f"exp1_regression_eps_sweep_{condition}_gamma", gamma)
    sample_path = gamma_dir / f"eps{eps:.2f}" / f"rep{rep:03d}" / "samples.npz"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing regression eps-sweep samples: {sample_path}")
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
    ax.set_xlabel("x1", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("x2", fontsize=LABEL_FONTSIZE)
    ax.set_zlabel("y", fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
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
    ax.set_xlabel("x1", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("x2", fontsize=LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
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
        save_figure(fig, 
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
        save_figure(fig, 
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
        fig.suptitle(rf"Regression outlier probabilities ({args.regression_condition}, $\epsilon={eps:.2f}$)", fontsize=TITLE_FONTSIZE)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        save_figure(fig, 
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
        fig.suptitle(rf"Simulation outlier probabilities ({args.simulation_scenario}, $\epsilon={eps:.2f}$)", fontsize=TITLE_FONTSIZE)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        save_figure(fig, 
            appendix_dir / f"fig_appendix_simulation_outlier2d_{args.simulation_scenario}_eps{eps_tag}_gammas.png",
            bbox_inches="tight",
            dpi=220,
        )
        plt.close(fig)


def _eps_to_k_stats(samples_dpd: np.ndarray, n_obs: int) -> dict[str, float]:
    """Summarise posterior epsilon and its induced audit-budget distribution."""
    rho = np.asarray(samples_dpd[:, -1], dtype=float)
    eps_samples = np.maximum(0.0, 1.0 - 1.0 / (1.0 + np.exp(-rho)))
    k_samples = np.clip(np.round(eps_samples * float(n_obs)).astype(int), 0, n_obs)
    k_low, k_high = np.percentile(k_samples, [2.5, 97.5])
    return {
        "eps_mean": float(eps_samples.mean()),
        "eps_low": float(np.percentile(eps_samples, 2.5)),
        "eps_high": float(np.percentile(eps_samples, 97.5)),
        "k_low": int(np.clip(round(float(k_low)), 0, n_obs)),
        "k_central": int(np.clip(round(float(k_samples.mean())), 0, n_obs)),
        "k_high": int(np.clip(round(float(k_high)), 0, n_obs)),
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
    """Compatibility ranking with posterior-derived audit-budget uncertainty."""
    order = np.argsort(ll)
    ll_sorted = np.asarray(ll, dtype=float)[order]
    ranks = np.arange(1, len(ll_sorted) + 1)
    k_low = int(k_stats["k_low"])
    k_central = int(k_stats["k_central"])
    k_high = int(k_stats["k_high"])

    if k_high > k_low:
        ax.axvspan(k_low, k_high, alpha=0.14, label="95% audit-budget interval")
    ax.axvline(k_central, linestyle="-", linewidth=1.5, label="Central audit budget")
    ax.plot(ranks, ll_sorted, color="black", linewidth=1.5)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Audit rank", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Log model density", fontsize=LABEL_FONTSIZE)
    ax.grid(True, alpha=0.22)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(loc="lower right", fontsize=LEGEND_FONTSIZE, frameon=True)
    # Numeric summary sits mid/upper-right, above the lower-right legend.
    txt = (
        rf"$\mathbb{{E}}[\epsilon\mid y]={k_stats['eps_mean']:.3f}$"
        + "\n"
        + rf"$K_{{\rm central}}={k_central}$"
        + "\n"
        + rf"$95\%~\mathrm{{CrI~for~}}K=[{k_low},{k_high}]$"
    )
    ax.text(
        0.98, 0.58, txt, transform=ax.transAxes, va="center", ha="right", fontsize=LABEL_FONTSIZE,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
    )

def _plot_selected_points_panel(
    ax,
    x: np.ndarray,
    order: np.ndarray,
    k_stats: dict[str, float],
    truth_mask: np.ndarray | None,
    title: str,
) -> None:
    k_central = int(k_stats["k_central"])
    k_high = int(k_stats["k_high"])
    idx_med = order[:k_central] if k_central > 0 else np.array([], dtype=int)
    idx_high = order[:k_high] if k_high > 0 else np.array([], dtype=int)
    idx_band = np.setdiff1d(idx_high, idx_med, assume_unique=False)

    ax.scatter(x[:, 0], x[:, 1], c="lightgray", s=14, alpha=0.55, label="all data")
    if len(idx_band) > 0:
        ax.scatter(x[idx_band, 0], x[idx_band, 1], c="#ffb366", s=22, alpha=0.85, label="upper-budget extension")
    if len(idx_med) > 0:
        ax.scatter(x[idx_med, 0], x[idx_med, 1], c="#d62728", s=26, alpha=0.9, label="central audit set")
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
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("x1", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("x2", fontsize=LABEL_FONTSIZE)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", fontsize=LEGEND_FONTSIZE, frameon=True)


def _plot_selected_points_panel_3d(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    order: np.ndarray,
    k_stats: dict[str, float],
    truth_mask: np.ndarray | None,
    title: str,
) -> None:
    k_central = int(k_stats["k_central"])
    k_high = int(k_stats["k_high"])
    idx_med = order[:k_central] if k_central > 0 else np.array([], dtype=int)
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
            x[idx_bg, 0],
            x[idx_bg, 1],
            y[idx_bg],
            c="#c8c8c8",
            s=7,
            alpha=0.18,
            depthshade=False,
        )
    if len(idx_band) > 0:
        ax.scatter(
            x[idx_band, 0],
            x[idx_band, 1],
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
            x[idx_med, 0],
            x[idx_med, 1],
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
            x[truth_mask, 0],
            x[truth_mask, 1],
            y[truth_mask],
            facecolors="none",
            edgecolors="black",
            s=34,
            linewidths=1.0,
            depthshade=False,
        )
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("x1", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("x2", fontsize=LABEL_FONTSIZE)
    ax.set_zlabel("y")
    ax.view_init(elev=22, azim=-62)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#c8c8c8", markersize=6, alpha=0.5, label="background data"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e67e22", markeredgecolor="#a95a00", markersize=7, label="upper-budget extension"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#b30000", markeredgecolor="#660000", markersize=7, label="central audit set"),
    ]
    if truth_mask is not None and np.any(truth_mask):
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color="black", markerfacecolor="none", markersize=7, label="true outlier")
        )
    ax.legend(handles=legend_handles, loc="upper center", ncol=2, fontsize=LEGEND_FONTSIZE, frameon=True)


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
    fig.suptitle(title, fontsize=TITLE_FONTSIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, output_path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def generate_likelihood_ranking_visualizations(args, outputs_root: Path, main_dir: Path, appendix_dir: Path) -> None:
    """Appendix gamma sensitivity for posterior audit budgets and FoD."""
    eps = float(args.posterior_eps)
    eps_tag = format_eps_tag(eps)
    gamma_list = parse_float_list(args.outlier_vis_gammas) or [0.1, 0.5, 1.0]

    reg_cases = []
    for gamma in gamma_list:
        try:
            x_train, y_train, fod, mask_reg, samples_reg, scaler = load_regression_outlier_case(
                outputs_root, gamma, args.regression_condition, eps, args.rep
            )
            reg_cases.append((gamma, x_train, y_train, fod, mask_reg, samples_reg, scaler))
        except FileNotFoundError:
            continue
    if reg_cases:
        ncols = len(reg_cases)
        fig, axes = plt.subplots(2, ncols, figsize=(5.0 * ncols, 7.8), squeeze=False)
        last_sc = None
        for col, (gamma, x_train, y_train, fod, mask_reg, samples_reg, scaler) in enumerate(reg_cases):
            ll = _compute_regression_loglik(x_train, y_train, samples_reg)
            k_stats = _eps_to_k_stats(samples_reg, len(x_train))
            _plot_rank_panel(axes[0, col], ll, k_stats, title=rf"$\gamma={gamma:g}$")
            last_sc = _plot_regression_fod_panel(
                axes[1, col], x_train, y_train, fod, mask_reg, samples_reg, scaler, title=rf"$\gamma={gamma:g}$"
            )
        if last_sc is not None:
            fig.colorbar(last_sc, ax=axes[1, :].tolist(), shrink=0.9, pad=0.02, label="FoD")
        fig.suptitle(rf"Regression audit-budget and FoD sensitivity ($\epsilon_0={eps:.2f}$)", fontsize=TITLE_FONTSIZE)
        fig.tight_layout(rect=(0, 0, 0.98, 0.95))
        save_figure(fig, 
            appendix_dir / f"fig_appendix_regression_audit_fod_{args.regression_condition}_eps{eps_tag}_gammas.png",
            bbox_inches="tight", dpi=220,
        )
        plt.close(fig)

    sim_cases = []
    for gamma in gamma_list:
        try:
            x_sim, fod, y_true_sim, samples_sim = load_simulation_outlier_case(
                outputs_root, gamma, args.simulation_scenario, eps, args.rep
            )
            sim_cases.append((gamma, x_sim, fod, y_true_sim, samples_sim))
        except FileNotFoundError:
            continue
    if sim_cases:
        ncols = len(sim_cases)
        fig, axes = plt.subplots(2, ncols, figsize=(5.0 * ncols, 7.8), squeeze=False)
        last_sc = None
        for col, (gamma, x_sim, fod, y_true_sim, samples_sim) in enumerate(sim_cases):
            ll = _compute_simulation_loglik(x_sim, samples_sim)
            k_stats = _eps_to_k_stats(samples_sim, len(x_sim))
            _plot_rank_panel(axes[0, col], ll, k_stats, title=rf"$\gamma={gamma:g}$")
            last_sc = _plot_simulation_fod_panel(
                axes[1, col], x_sim, fod, y_true_sim, title=rf"$\gamma={gamma:g}$"
            )
        if last_sc is not None:
            fig.colorbar(last_sc, ax=axes[1, :].tolist(), shrink=0.9, pad=0.02, label="FoD")
        fig.suptitle(rf"Simulation audit-budget and FoD sensitivity ($\epsilon_0={eps:.2f}$)", fontsize=TITLE_FONTSIZE)
        fig.tight_layout(rect=(0, 0, 0.98, 0.95))
        save_figure(fig, 
            appendix_dir / f"fig_appendix_simulation_audit_fod_{args.simulation_scenario}_eps{eps_tag}_gammas.png",
            bbox_inches="tight", dpi=220,
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
    ax_param.xaxis.label.set_size(LABEL_FONTSIZE)
    ax_param.yaxis.label.set_size(LABEL_FONTSIZE)
    ax_param.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax_param.grid(True, alpha=0.2)

    for method in PROPOSAL_METHODS:
        if method not in case["methods"]:
            continue
        eps_vals = case["methods"][method]["eps"]
        if eps_vals is None:
            continue
        _draw_1d_kde(ax_eps, eps_vals, METHOD_STYLES[method]["color"], method)
    ax_eps.axvline(case["truth"]["eps"], color="black", linestyle="--", linewidth=1.0)
    ax_eps.set_xlabel(r"$\epsilon$", fontsize=LABEL_FONTSIZE)
    if show_ylabels:
        ax_eps.set_ylabel("density", fontsize=LABEL_FONTSIZE)
    ax_eps.xaxis.label.set_size(LABEL_FONTSIZE)
    ax_eps.yaxis.label.set_size(LABEL_FONTSIZE)
    ax_eps.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax_eps.grid(True, alpha=0.2)


def _plot_simulation_fod_panel(ax, x: np.ndarray, fod: np.ndarray, truth_mask: np.ndarray, title: str):
    sc = ax.scatter(x[:, 0], x[:, 1], c=fod, s=18, alpha=0.85, vmin=0.0, vmax=1.0)
    if np.any(truth_mask):
        ax.scatter(
            x[truth_mask, 0], x[truth_mask, 1], facecolors="none", edgecolors="black",
            s=38, linewidths=0.9, label="True outlier",
        )
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(r"$x_1$", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(r"$x_2$", fontsize=LABEL_FONTSIZE)
    ax.grid(True, alpha=0.2)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    if np.any(truth_mask):
        ax.legend(loc="lower right", frameon=True, fontsize=LEGEND_FONTSIZE)
    return sc


def _regression_fitted_residual(
    x_train: np.ndarray,
    y_train: np.ndarray,
    samples_dpd: np.ndarray,
    scaler: dict,
) -> tuple[np.ndarray, np.ndarray]:
    transformed = recover_posterior_params_original_scale(np.asarray(samples_dpd, dtype=float), scaler)
    beta = np.asarray(transformed["beta"], dtype=float).mean(axis=0)
    fitted = beta[0] + np.asarray(x_train, dtype=float) @ beta[1:]
    residual = np.asarray(y_train, dtype=float) - fitted
    return fitted, residual


def _plot_regression_fod_panel(
    ax,
    x_train: np.ndarray,
    y_train: np.ndarray,
    fod: np.ndarray,
    truth_mask: np.ndarray,
    samples_dpd: np.ndarray,
    scaler: dict,
    title: str,
):
    fitted, residual = _regression_fitted_residual(x_train, y_train, samples_dpd, scaler)
    sc = ax.scatter(fitted, residual, c=fod, s=18, alpha=0.85, vmin=0.0, vmax=1.0)
    if np.any(truth_mask):
        ax.scatter(
            fitted[truth_mask], residual[truth_mask], facecolors="none", edgecolors="black",
            s=38, linewidths=0.9, label="True outlier",
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Posterior fitted value", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Residual", fontsize=LABEL_FONTSIZE)
    ax.grid(True, alpha=0.2)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    if np.any(truth_mask):
        ax.legend(loc="lower right", frameon=True, fontsize=LEGEND_FONTSIZE)
    return sc


def create_main_fig1_posterior(args, outputs_root: Path, main_dir: Path) -> None:
    """Main Fig. 1 (2x2): representative joint posterior recovery.

    Simulation uses ``simulation_main_gamma`` (default 0.5); regression uses
    ``regression_main_gamma`` (default 1.0). Displays H-B (Ours) vs Standard Bayes
    for model parameters and H-B only for the contamination proportion.
    """
    try:
        sim_case = load_simulation_joint_case(args, float(args.simulation_main_gamma))
        reg_case = load_regression_joint_case(args, float(args.regression_main_gamma))
    except FileNotFoundError:
        return

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), squeeze=False)
    legend_handles = []
    for row, (case, row_name) in enumerate([(sim_case, "Simulation"), (reg_case, "Regression")]):
        ax_param, ax_eps = axes[row]
        for method in ["DPD", "NLL"]:
            if method not in case["methods"]:
                continue
            info = case["methods"][method]
            color = METHOD_STYLES[method]["color"]
            _draw_2d_kde_contour(ax_param, info["x"], info["y"], color)
            if row == 0:
                legend_handles.append(
                    plt.Line2D([0], [0], color=color, lw=1.8, label=METHOD_DISPLAY.get(method, method))
                )
        ax_param.axvline(case["truth"]["x"], color="black", linestyle="--", linewidth=1.1)
        ax_param.axhline(case["truth"]["y"], color="black", linestyle="--", linewidth=1.1)
        ax_param.set_title(f"{row_name}: model parameter", fontsize=TITLE_FONTSIZE)
        ax_param.set_xlabel(case["labels"]["x"], fontsize=LABEL_FONTSIZE)
        ax_param.set_ylabel(case["labels"]["y"], fontsize=LABEL_FONTSIZE)
        ax_param.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax_param.grid(True, alpha=0.2)

        if "DPD" in case["methods"] and case["methods"]["DPD"]["eps"] is not None:
            _draw_1d_kde(
                ax_eps, case["methods"]["DPD"]["eps"], METHOD_STYLES["DPD"]["color"], METHOD_DISPLAY["DPD"]
            )
        ax_eps.axvline(case["truth"]["eps"], color="black", linestyle="--", linewidth=1.1, label="True value")
        ax_eps.set_title(f"{row_name}: contamination proportion", fontsize=TITLE_FONTSIZE)
        ax_eps.set_xlabel(r"$\epsilon$", fontsize=LABEL_FONTSIZE)
        ax_eps.set_ylabel("Posterior density", fontsize=LABEL_FONTSIZE)
        ax_eps.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax_eps.grid(True, alpha=0.2)

    legend_handles.append(plt.Line2D([0], [0], color="black", linestyle="--", lw=1.2, label="True value"))
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", ncol=3, frameon=True, fontsize=LEGEND_FONTSIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, main_dir / "fig1_posterior_regression_simulation.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def create_main_fig2_accuracy(args, outputs_root: Path, main_dir: Path, df_sim: pd.DataFrame, df_reg_eps: pd.DataFrame) -> None:
    """Main Fig. 2 (2x2): target-model performance and epsilon recovery.

    Layout:
      top-left     Simulation target-parameter error
      top-right    Simulation recovery of epsilon_0 (H-B only)
      bottom-left  Regression clean-test RMSE
      bottom-right Regression recovery of epsilon_0 (H-B only)

    Outlier-ranking / AUPRC comparisons are appendix-only.
    """
    df_sim_g = df_sim[
        np.isclose(df_sim["gamma_tag"], float(args.simulation_main_gamma))
        & (df_sim["scenario"] == args.simulation_scenario)
    ].copy()
    df_reg_g = df_reg_eps[np.isclose(df_reg_eps["gamma_tag"], float(args.regression_main_gamma))].copy()
    if df_sim_g.empty or df_reg_g.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), squeeze=False)
    main_sim = [m for m in MAIN_METHODS if m in available_methods(df_sim_g)]
    main_reg = [m for m in MAIN_METHODS if m in available_methods(df_reg_g)]

    plot_lines(
        axes[0, 0], df_sim_g, "eps_true", "mu_bias_norm_mean", main_sim,
        yerr_col="mu_bias_norm_std", title="Simulation: target-parameter error",
        xlabel=r"True contamination $\epsilon_0$", ylabel=r"$\|\hat\mu-\mu_0\|_2$",
    )
    plot_lines(
        axes[0, 1], df_sim_g, "eps_true", "eps_post_mean_mean", ["DPD"],
        yerr_col="eps_post_mean_std", title=r"Simulation: recovery of $\epsilon_0$",
        xlabel=r"True contamination $\epsilon_0$", ylabel=r"Posterior mean of $\epsilon$", identity=True,
    )
    plot_lines(
        axes[1, 0], df_reg_g, "eps_true", "rmse_test_mean", main_reg,
        yerr_col="rmse_test_std", title="Regression: clean-test RMSE",
        xlabel=r"True contamination $\epsilon_0$", ylabel="Test RMSE",
    )
    plot_lines(
        axes[1, 1], df_reg_g, "eps_true", "eps_post_mean_mean", ["DPD"],
        yerr_col="eps_post_mean_std", title=r"Regression: recovery of $\epsilon_0$",
        xlabel=r"True contamination $\epsilon_0$", ylabel=r"Posterior mean of $\epsilon$", identity=True,
    )

    method_handles, method_labels = axes[0, 0].get_legend_handles_labels()
    true_handles, true_labels = axes[0, 1].get_legend_handles_labels()
    for ax in axes.ravel():
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    legend_handles = list(method_handles)
    legend_labels = list(method_labels)
    for handle, label in zip(true_handles, true_labels):
        if label == "True" and label not in legend_labels:
            legend_handles.append(handle)
            legend_labels.append(label)
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=max(1, len(legend_labels)), frameon=True, fontsize=LEGEND_FONTSIZE)
    for ax in axes.ravel():
        ax.title.set_fontsize(TITLE_FONTSIZE)
        ax.xaxis.label.set_size(LABEL_FONTSIZE)
        ax.yaxis.label.set_size(LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, main_dir / "fig2_accuracy_regression_simulation.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def create_main_fig3_likelihood(args, outputs_root: Path, main_dir: Path) -> None:
    """Main Fig. 3 (2x2): posterior-guided auditing (how many / which observations).

    Left: inferred audit budget from the epsilon posterior along a compatibility ranking.
    Right: FoD visualization (simulation scatter; regression fitted-value/residual).
    Simulation uses γ=0.5 and regression uses γ=1.0 by default.
    """
    eps = float(args.posterior_eps)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6), squeeze=False)
    try:
        x_sim, fod_sim, y_true_sim, samples_sim = load_simulation_outlier_case(
            outputs_root, float(args.simulation_main_gamma), args.simulation_scenario, eps, args.rep
        )
        ll_sim = _compute_simulation_loglik(x_sim, samples_sim)
        ks_sim = _eps_to_k_stats(samples_sim, len(x_sim))
        _plot_rank_panel(axes[0, 0], ll_sim, ks_sim, title="Simulation: inferred audit budget")
        sc_sim = _plot_simulation_fod_panel(
            axes[0, 1], x_sim, fod_sim, y_true_sim, title="Simulation: FoD across observations"
        )
        fig.colorbar(sc_sim, ax=axes[0, 1], label="FoD")
    except FileNotFoundError:
        pass

    try:
        x_reg, y_reg, fod_reg, mask_reg, samples_reg, scaler_reg = load_regression_outlier_case(
            outputs_root, float(args.regression_main_gamma), args.regression_condition, eps, args.rep
        )
        ll_reg = _compute_regression_loglik(x_reg, y_reg, samples_reg)
        ks_reg = _eps_to_k_stats(samples_reg, len(x_reg))
        _plot_rank_panel(axes[1, 0], ll_reg, ks_reg, title="Regression: inferred audit budget")
        sc_reg = _plot_regression_fod_panel(
            axes[1, 1], x_reg, y_reg, fod_reg, mask_reg, samples_reg, scaler_reg,
            title="Regression: FoD in fitted-residual space",
        )
        fig.colorbar(sc_reg, ax=axes[1, 1], label="FoD")
    except FileNotFoundError:
        pass

    fig.tight_layout()
    save_figure(fig, main_dir / "fig3_likelihood_selection_combined.png", bbox_inches="tight", dpi=220)
    plt.close(fig)

def create_appendix_phi_sensitivity(
    args,
    appendix_dir: Path,
    df_sim: pd.DataFrame,
    df_reg_eps: pd.DataFrame,
) -> None:
    """Appendix: sensitivity to the default φ specification (DPD vs ExpSat vs RatSat)."""
    df_sim_g = df_sim[
        np.isclose(df_sim["gamma_tag"], float(args.simulation_main_gamma))
        & (df_sim["scenario"] == args.simulation_scenario)
    ].copy()
    df_reg_g = df_reg_eps[np.isclose(df_reg_eps["gamma_tag"], float(args.regression_main_gamma))].copy()
    if df_sim_g.empty or df_reg_g.empty:
        return

    sim_methods = [m for m in PROPOSAL_METHODS if m in available_methods(df_sim_g)]
    reg_methods = [m for m in PROPOSAL_METHODS if m in available_methods(df_reg_g)]
    if len(sim_methods) <= 1 and len(reg_methods) <= 1:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), squeeze=False)
    plot_lines(
        axes[0, 0], df_sim_g, "eps_true", "mu_bias_norm_mean", sim_methods,
        yerr_col="mu_bias_norm_std", title=r"Simulation: parameter error",
        xlabel=r"True contamination $\epsilon_0$", ylabel=r"$\|\hat\mu-\mu_0\|_2$",
    )
    plot_lines(
        axes[0, 1], df_sim_g, "eps_true", "eps_post_mean_mean", sim_methods,
        yerr_col="eps_post_mean_std", title=r"Simulation: $\epsilon$ recovery",
        xlabel=r"True contamination $\epsilon_0$", ylabel=r"Posterior mean of $\epsilon$", identity=True,
    )
    plot_lines(
        axes[0, 2], df_sim_g, "eps_true", "auprc_mean", sim_methods,
        yerr_col="auprc_std", title="Simulation: ranking-only AUPRC",
        xlabel=r"True contamination $\epsilon_0$", ylabel="AUPRC",
    )
    plot_lines(
        axes[1, 0], df_reg_g, "eps_true", "rmse_test_mean", reg_methods,
        yerr_col="rmse_test_std", title="Regression: clean-test RMSE",
        xlabel=r"True contamination $\epsilon_0$", ylabel="Test RMSE",
    )
    plot_lines(
        axes[1, 1], df_reg_g, "eps_true", "eps_post_mean_mean", reg_methods,
        yerr_col="eps_post_mean_std", title=r"Regression: $\epsilon$ recovery",
        xlabel=r"True contamination $\epsilon_0$", ylabel=r"Posterior mean of $\epsilon$", identity=True,
    )
    plot_lines(
        axes[1, 2], df_reg_g, "eps_true", "auprc_mean", reg_methods,
        yerr_col="auprc_std", title="Regression: ranking-only AUPRC",
        xlabel=r"True contamination $\epsilon_0$", ylabel="AUPRC",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax in axes.ravel():
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        ax.title.set_fontsize(TITLE_FONTSIZE)
        ax.xaxis.label.set_size(LABEL_FONTSIZE)
        ax.yaxis.label.set_size(LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=True, fontsize=LEGEND_FONTSIZE)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, appendix_dir / "fig_appendix_phi_sensitivity.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def create_appendix_ranking_only_comparison(
    args,
    appendix_dir: Path,
    df_sim: pd.DataFrame,
    df_reg_eps: pd.DataFrame,
) -> None:
    """Appendix: baseline ranking-only AUPRC at representative gammas.

    Competing methods do not infer contamination / audit budget, so this figure
    compares observation ranking quality alone (not the full auditing mechanism).
    """
    df_sim_g = df_sim[
        np.isclose(df_sim["gamma_tag"], float(args.simulation_main_gamma))
        & (df_sim["scenario"] == args.simulation_scenario)
    ].copy()
    df_reg_g = df_reg_eps[np.isclose(df_reg_eps["gamma_tag"], float(args.regression_main_gamma))].copy()
    if df_sim_g.empty and df_reg_g.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), squeeze=False)
    if not df_sim_g.empty:
        methods = [m for m in MAIN_METHODS if m in available_methods(df_sim_g)]
        plot_lines(
            axes[0, 0], df_sim_g, "eps_true", "auprc_mean", methods,
            yerr_col="auprc_std",
            title=rf"Simulation ranking-only AUPRC ($\gamma={args.simulation_main_gamma:g}$)",
            xlabel=r"True contamination $\epsilon_0$", ylabel="AUPRC",
        )
    if not df_reg_g.empty:
        methods = [m for m in MAIN_METHODS if m in available_methods(df_reg_g)]
        plot_lines(
            axes[0, 1], df_reg_g, "eps_true", "auprc_mean", methods,
            yerr_col="auprc_std",
            title=rf"Regression ranking-only AUPRC ($\gamma={args.regression_main_gamma:g}$)",
            xlabel=r"True contamination $\epsilon_0$", ylabel="AUPRC",
        )
    handles, labels = [], []
    for ax in axes.ravel():
        h, lab = ax.get_legend_handles_labels()
        if h and not handles:
            handles, labels = h, lab
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        ax.title.set_fontsize(TITLE_FONTSIZE)
        ax.xaxis.label.set_size(LABEL_FONTSIZE)
        ax.yaxis.label.set_size(LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=True, fontsize=LEGEND_FONTSIZE)
    fig.suptitle(
        "Ranking-only comparison (baselines do not infer contamination / audit budget)",
        fontsize=TITLE_FONTSIZE,
        y=1.02,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, appendix_dir / "fig_appendix_ranking_only_comparison.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def _topk_precision_recall_f1(y_true: np.ndarray, scores: np.ndarray, k: int) -> dict[str, float]:
    """Precision / recall / F1 for the top-k observations ranked by FoD scores."""
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=float)
    n = len(y_true)
    n_pos = int(y_true.sum())
    k = int(np.clip(int(k), 0, n))
    if n_pos == 0 or k <= 0:
        return {"precision_at_k": np.nan, "recall_at_k": np.nan, "f1_at_k": np.nan}
    order = np.argsort(scores)[::-1][:k]
    selected = np.zeros(n, dtype=bool)
    selected[order] = True
    tp = int(np.sum(selected & y_true))
    fp = int(np.sum(selected & ~y_true))
    fn = int(np.sum(~selected & y_true))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision_at_k": float(precision),
        "recall_at_k": float(recall),
        "f1_at_k": float(f1),
    }


def _auditing_metrics_from_samples(raw: np.lib.npyio.NpzFile) -> dict[str, float] | None:
    """H-B only: FoD AUPRC and top-K metrics at K_central = round(mean(K_samples))."""
    from sklearn.metrics import average_precision_score

    if "samples_dpd" not in raw.files:
        return None
    samples = np.asarray(raw["samples_dpd"], dtype=float)
    if "p_out_dpd" in raw.files:
        fod = np.asarray(raw["p_out_dpd"], dtype=float)
    elif "p_out" in raw.files:
        fod = np.asarray(raw["p_out"], dtype=float)
    else:
        return None
    if "y_true" in raw.files:
        y_true = np.asarray(raw["y_true"]).astype(bool)
    elif "mask_outlier" in raw.files:
        y_true = np.asarray(raw["mask_outlier"]).astype(bool)
    else:
        return None

    k_stats = _eps_to_k_stats(samples, len(y_true))
    k_central = int(k_stats["k_central"])
    if int(y_true.sum()) == 0:
        return {
            "fod_auprc": np.nan,
            "k_central": float(k_central),
            "precision_at_k": np.nan,
            "recall_at_k": np.nan,
            "f1_at_k": np.nan,
            "source": "samples",
        }
    topk = _topk_precision_recall_f1(y_true, fod, k_central)
    return {
        "fod_auprc": float(average_precision_score(y_true.astype(int), fod)),
        "k_central": float(k_central),
        **topk,
        "source": "samples",
    }


def _auditing_metrics_from_metrics_json(metrics_path: Path) -> dict[str, float] | None:
    """Fallback when samples.npz is unavailable: recover table fields from metrics.json.

    Uses saved FoD AUPRC / recall_at_k_est. ``K_central`` prefers the stored ``k_est``
    (the budget used for that recall); otherwise ``round(eps_post_mean * n)``.
    Precision/F1 are recovered from recall, k_oracle, and K_central.
    """
    import json

    meta = json.loads(metrics_path.read_text(encoding="utf-8"))
    dpd = meta.get("dpd")
    if not isinstance(dpd, dict):
        return None
    n = int(meta.get("n", 0))
    eps_true = float(meta.get("eps_true", dpd.get("eps_true", np.nan)))
    auprc = dpd.get("auprc", np.nan)
    recall = dpd.get("recall_k_est", np.nan)
    if "k_est" in dpd and dpd["k_est"] is not None:
        k_central = int(dpd["k_est"])
    elif n > 0 and dpd.get("eps_post_mean") is not None:
        k_central = int(np.clip(np.round(float(dpd["eps_post_mean"]) * n), 0, n))
    else:
        return None
    if "k_oracle" in dpd and dpd["k_oracle"] is not None:
        k_oracle = int(dpd["k_oracle"])
    elif n > 0 and np.isfinite(eps_true):
        k_oracle = int(np.clip(np.round(eps_true * n), 0, n))
    else:
        k_oracle = 0

    if k_oracle <= 0 or not np.isfinite(float(recall)):
        precision = np.nan
        f1 = np.nan
        recall_out = np.nan if k_oracle <= 0 else float(recall)
    else:
        tp = float(recall) * float(k_oracle)
        precision = float(np.clip(tp / max(k_central, 1), 0.0, 1.0))
        recall_out = float(recall)
        f1 = 2.0 * precision * recall_out / max(precision + recall_out, 1e-12)

    return {
        "fod_auprc": float(auprc) if auprc is not None else np.nan,
        "k_central": float(k_central),
        "precision_at_k": float(precision) if precision == precision else np.nan,
        "recall_at_k": float(recall_out) if recall_out == recall_out else np.nan,
        "f1_at_k": float(f1) if f1 == f1 else np.nan,
        "source": "metrics_json",
    }


def _discover_eps_rep_dirs(case_root: Path) -> list[tuple[float, int, Path]]:
    """Return (eps_true, rep, rep_dir) under a gamma-specific experiment directory."""
    rows = []
    if not case_root.exists():
        return rows
    for eps_dir in sorted(case_root.glob("eps*")):
        m = re.match(r"eps([0-9]+(?:\.[0-9]+)?)$", eps_dir.name)
        if not m:
            continue
        eps_true = float(m.group(1))
        for rep_dir in sorted(eps_dir.glob("rep*")):
            m_rep = re.match(r"rep(\d+)$", rep_dir.name)
            if not m_rep:
                continue
            rows.append((eps_true, int(m_rep.group(1)), rep_dir))
    return rows


def export_hb_auditing_table(args, outputs_root: Path, main_dir: Path) -> Path | None:
    """Export a main-text CSV evaluating the H-B self-calibrated auditing mechanism.

    For each contamination level (simulation and regression), report FoD AUPRC and
    precision / recall / F1 at ``K_central = round(mean(K_samples))`` when posterior
    samples are available; otherwise fall back to saved ``metrics.json`` (all reps).
    Baseline ranking-only AUPRC comparisons remain appendix figures.
    """
    settings = []
    try:
        settings.append(
            (
                "simulation",
                float(args.simulation_main_gamma),
                find_gamma_dir(outputs_root, "exp1_simulation_gamma_sweep_gamma", float(args.simulation_main_gamma))
                / args.simulation_scenario,
            )
        )
    except FileNotFoundError:
        pass
    try:
        settings.append(
            (
                "regression",
                float(args.regression_main_gamma),
                find_gamma_dir(
                    outputs_root,
                    f"exp1_regression_eps_sweep_{args.regression_condition}_gamma",
                    float(args.regression_main_gamma),
                ),
            )
        )
    except FileNotFoundError:
        pass

    per_rep_rows = []
    for setting, gamma, case_root in settings:
        for eps_true, rep, rep_dir in _discover_eps_rep_dirs(case_root):
            metrics = None
            sample_path = rep_dir / "samples.npz"
            if sample_path.exists():
                raw = np.load(sample_path, allow_pickle=True)
                metrics = _auditing_metrics_from_samples(raw)
            if metrics is None:
                metrics_path = rep_dir / "metrics.json"
                if metrics_path.exists():
                    metrics = _auditing_metrics_from_metrics_json(metrics_path)
            if metrics is None:
                continue
            per_rep_rows.append(
                {
                    "setting": setting,
                    "gamma": gamma,
                    "eps_true": eps_true,
                    "rep": rep,
                    **metrics,
                }
            )

    if not per_rep_rows:
        return None

    df_rep = pd.DataFrame(per_rep_rows)
    metric_cols = ["fod_auprc", "k_central", "precision_at_k", "recall_at_k", "f1_at_k"]
    agg_kwargs = {"n_reps": ("rep", "count")}
    for c in metric_cols:
        agg_kwargs[f"{c}_mean"] = (c, "mean")
        agg_kwargs[f"{c}_std"] = (c, "std")
    agg = (
        df_rep.groupby(["setting", "gamma", "eps_true"], as_index=False)
        .agg(**agg_kwargs)
        .sort_values(["setting", "eps_true"])
    )
    out_path = main_dir / "table_main_hb_auditing.csv"
    agg.to_csv(out_path, index=False)
    df_rep.sort_values(["setting", "eps_true", "rep"]).to_csv(
        main_dir / "table_main_hb_auditing_per_rep.csv", index=False
    )
    return out_path


def export_main_text_final_set(args, outputs_root: Path, main_dir: Path, df_sim: pd.DataFrame, df_reg_eps: pd.DataFrame) -> None:
    """Export main-text Figs. 1–3, the H-B auditing CSV, and caption drafts."""
    create_main_fig1_posterior(args, outputs_root, main_dir)
    create_main_fig2_accuracy(args, outputs_root, main_dir, df_sim, df_reg_eps)
    create_main_fig3_likelihood(args, outputs_root, main_dir)
    export_hb_auditing_table(args, outputs_root, main_dir)

    caption_path = main_dir / "main_figure_captions_draft.md"
    caption_text = f"""# Main Text Figure Captions (Draft)

Representative settings: simulation $\\gamma={args.simulation_main_gamma:g}$, regression $\\gamma={args.regression_main_gamma:g}$.
Main-text method label is H-B (Ours) (default $\\phi$ = DPD). Full $\\gamma$-sensitivity, ranking-only baselines, $\\phi$-sensitivity, and $\\kappa$-ablation appear in the Appendix.

## Fig.1 Joint posterior recovery (2x2)
Left: model-parameter posterior (H-B vs Standard Bayes). Right: contamination-proportion posterior (H-B). Dashed lines mark true values.

## Fig.2 Target performance and epsilon recovery (2x2)
Top-left: simulation target-parameter error. Top-right: simulation recovery of $\\epsilon_0$ (H-B only). Bottom-left: regression clean-test RMSE. Bottom-right: regression recovery of $\\epsilon_0$ (H-B only). Ranking-only AUPRC is Appendix-only.

## Fig.3 Posterior-guided auditing: how many and which observations (2x2)
Left: inferred audit budget from the epsilon posterior (central $K$ and 95\\% posterior interval for $K$, matching the shaded interval) along a compatibility ranking. Right: FoD scores (simulation scatter; regression fitted-value/residual). The accompanying CSV evaluates H-B FoD AUPRC and precision/recall/F1 at $K_{{\\rm central}}=\\mathrm{{round}}(\\mathbb{{E}}[K])$.
"""
    caption_path.write_text(caption_text, encoding="utf-8")

def generate_posterior_grids(args, output_dir: Path, sim_available: list[float], reg_available: list[float]) -> None:
    """Appendix-only gamma sensitivity for posterior shapes."""
    appendix_dir = output_dir / "appendix"
    common_available = [
        g for g in sim_available
        if any(math.isclose(g, r, rel_tol=0.0, abs_tol=1e-12) for r in reg_available)
    ]
    if not common_available:
        return
    appendix_gammas = select_gammas(
        parse_float_list(args.posterior_appendix_gammas), common_available, fallback=common_available
    )
    sim_cases = [load_simulation_joint_case(args, g) for g in appendix_gammas]
    reg_cases = [load_regression_joint_case(args, g) for g in appendix_gammas]
    eps_tag = f"{args.posterior_eps:.2f}".replace(".", "p")
    plot_joint_posterior_comparison(
        sim_cases,
        title=rf"Simulation posterior sensitivity ({args.simulation_scenario}, $\epsilon_0={args.posterior_eps:.2f}$)",
        output_path=appendix_dir / f"fig_appendix_simulation_posterior_gamma_{args.simulation_scenario}_eps{eps_tag}.png",
    )
    plot_joint_posterior_comparison(
        reg_cases,
        title=rf"Regression posterior sensitivity ({args.regression_condition}, $\epsilon_0={args.posterior_eps:.2f}$)",
        output_path=appendix_dir / f"fig_appendix_regression_posterior_gamma_{args.regression_condition}_eps{eps_tag}.png",
    )

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate section-6 paper figures from saved exp1 outputs. "
            "Main text uses simulation γ=0.5 and regression γ=1.0 by default; "
            "full γ-sensitivity and ranking-only baselines remain in the appendix."
        )
    )
    parser.add_argument("--outputs_root", default="outputs")
    parser.add_argument("--output_dir", default="outputs/paper_exp1_suite_section6_revised")
    parser.add_argument("--simulation_main_gamma", type=float, default=0.5)
    parser.add_argument("--regression_main_gamma", type=float, default=1.0)
    parser.add_argument("--simulation_scenario", default="A", choices=["A", "B"])
    parser.add_argument("--regression_condition", default="mu6")
    parser.add_argument("--posterior_eps", type=float, default=0.2)
    # Retained for backward-compatible CLI calls; main Fig. 1 now uses only the representative gamma.
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
    ensure_dir(main_dir)
    ensure_dir(appendix_dir)

    df_sim = load_gamma_summaries(outputs_root, "exp1_simulation_gamma_sweep_gamma")
    df_reg_final = load_gamma_summaries(outputs_root, "exp1_regression_final_gamma")
    df_reg_eps = load_gamma_summaries(outputs_root, "exp1_regression_eps_sweep_mu6_gamma")

    # Appendix gamma-sensitivity summaries. Main-text figures are exported separately below.
    if not df_sim.empty:
        sim_gammas = sorted(df_sim["gamma_tag"].unique())
        df_sim_scenario = df_sim[df_sim["scenario"] == args.simulation_scenario]
        sim_main_methods = [m for m in MAIN_METHODS if m in available_methods(df_sim_scenario)]
        plot_gamma_grid(df_sim_scenario, sim_gammas, "eps_true", "mu_bias_norm_mean", "mu_bias_norm_std",
                        sim_main_methods, output_path=appendix_dir / f"fig_appendix_simulation_parameter_error_{args.simulation_scenario}.png",
                        title_prefix=f"Simulation parameter error ({args.simulation_scenario})",
                        xlabel=r"True contamination $\epsilon_0$", ylabel=r"$\|\hat\mu-\mu_0\|_2$")
        plot_gamma_grid(df_sim_scenario, sim_gammas, "eps_true", "auprc_mean", "auprc_std",
                        sim_main_methods, output_path=appendix_dir / f"fig_appendix_simulation_ranking_only_auprc_{args.simulation_scenario}.png",
                        title_prefix=f"Simulation ranking-only AUPRC ({args.simulation_scenario})",
                        xlabel=r"True contamination $\epsilon_0$", ylabel="AUPRC")
        plot_gamma_grid(df_sim_scenario, sim_gammas, "eps_true", "eps_post_mean_mean", "eps_post_mean_std",
                        ["DPD"], output_path=appendix_dir / f"fig_appendix_simulation_eps_estimation_{args.simulation_scenario}.png",
                        title_prefix=f"Simulation epsilon recovery ({args.simulation_scenario})",
                        xlabel=r"True contamination $\epsilon_0$", ylabel=r"Posterior mean of $\epsilon$", identity=True)
        plot_gamma_grid(df_sim_scenario, sim_gammas, "eps_true", "eps_ci95_cover_mean", None,
                        ["DPD"], output_path=appendix_dir / f"fig_appendix_simulation_eps_coverage_{args.simulation_scenario}.png",
                        title_prefix=f"Simulation epsilon coverage ({args.simulation_scenario})",
                        xlabel=r"True contamination $\epsilon_0$", ylabel="Coverage probability")

    if not df_reg_eps.empty:
        reg_eps_gammas = sorted(df_reg_eps["gamma_tag"].unique())
        reg_main_methods = [m for m in MAIN_METHODS if m in available_methods(df_reg_eps)]
        plot_gamma_grid(df_reg_eps, reg_eps_gammas, "eps_true", "rmse_test_mean", "rmse_test_std",
                        reg_main_methods, output_path=appendix_dir / "fig_appendix_regression_eps_rmse.png",
                        title_prefix="Regression clean-test RMSE", xlabel=r"True contamination $\epsilon_0$", ylabel="Test RMSE")
        plot_gamma_grid(df_reg_eps, reg_eps_gammas, "eps_true", "auprc_mean", "auprc_std",
                        reg_main_methods, output_path=appendix_dir / "fig_appendix_regression_ranking_only_auprc.png",
                        title_prefix="Regression ranking-only AUPRC", xlabel=r"True contamination $\epsilon_0$", ylabel="AUPRC")
        plot_gamma_grid(df_reg_eps, reg_eps_gammas, "eps_true", "eps_post_mean_mean", "eps_post_mean_std",
                        ["DPD"], output_path=appendix_dir / "fig_appendix_regression_eps_estimation.png",
                        title_prefix="Regression epsilon recovery", xlabel=r"True contamination $\epsilon_0$",
                        ylabel=r"Posterior mean of $\epsilon$", identity=True)
        plot_gamma_grid(df_reg_eps, reg_eps_gammas, "eps_true", "eps_ci95_cover_mean", None,
                        ["DPD"], output_path=appendix_dir / "fig_appendix_regression_eps_coverage.png",
                        title_prefix="Regression epsilon coverage", xlabel=r"True contamination $\epsilon_0$",
                        ylabel="Coverage probability")

    if not df_reg_final.empty:
        reg_final_gammas = sorted(df_reg_final["gamma_tag"].unique())
        reg_final_methods = [m for m in MAIN_METHODS if m in available_methods(df_reg_final)]
        plot_gamma_grid(df_reg_final, reg_final_gammas, "condition", "rmse_test_mean", "rmse_test_std",
                        reg_final_methods, output_path=appendix_dir / "fig_appendix_regression_conditions_rmse.png",
                        title_prefix="Regression conditions RMSE", xlabel="Condition", ylabel="Test RMSE")
        plot_gamma_grid(df_reg_final, reg_final_gammas, "condition", "beta_l2_error_mean", "beta_l2_error_std",
                        reg_final_methods, output_path=appendix_dir / "fig_appendix_regression_conditions_beta_l2.png",
                        title_prefix="Regression conditions parameter error", xlabel="Condition",
                        ylabel=r"$\|\hat\beta-\beta_0\|_2$")

    if not df_sim.empty and not df_reg_eps.empty:
        create_appendix_phi_sensitivity(args, appendix_dir, df_sim, df_reg_eps)
        create_appendix_ranking_only_comparison(args, appendix_dir, df_sim, df_reg_eps)

    sim_available = [gamma for gamma, _ in discover_gamma_dirs(str(outputs_root), "exp1_simulation_gamma_sweep_gamma")]
    reg_available = [gamma for gamma, _ in discover_gamma_dirs(str(outputs_root), "exp1_regression_eps_sweep_mu6_gamma")]
    if sim_available and reg_available:
        generate_posterior_grids(args, output_dir, sim_available, reg_available)

    generate_likelihood_ranking_visualizations(args, outputs_root, main_dir, appendix_dir)
    export_main_text_final_set(args, outputs_root, main_dir, df_sim, df_reg_eps)
    plot_simulation_kappa_appendix(outputs_root, appendix_dir)
    plot_regression_kappa_appendix(outputs_root, appendix_dir)

    print(f"Saved paper figure suite to {output_dir}")


if __name__ == "__main__":
    main()
