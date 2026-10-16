import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List

if "MALLOC_TRIM_THRESHOLD_" not in os.environ:
    os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"
if "MKL_NUM_THREADS" not in os.environ:
    os.environ["MKL_NUM_THREADS"] = "1"

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hbayes.paper_sampling import CONTEXT
from hbayes.experiment_temperature import (TEMPERATURE_VERSION, covariance_temperature,
    response_temperature, positive_temperature, fit_metadata, nuts_summary, convergence_summary,
    configuration, begin_run, record_completion)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.exp1_regression.run_exp1_regression import (
    BASELINE_METHODS,
    METHOD_JSON_KEYS,
    NumpyEncoder,
    get_all_methods,
    get_primary_proposal_method,
    get_proposal_methods,
)
from experiments.exp1_regression.utils_exp1_regression import (
    compute_dpd_outlier_prob,
    evaluate_detection_metrics,
    evaluate_regression_metrics,
    fit_regression_methods,
    generate_representative_plots_from_raw,
    score_baseline,
    simulate_regression_dataset,
    standardize_regression_data,
    summarize_dpd_posterior,
    summarize_posterior_prediction,
)


@dataclass
class SweepConfig:
    n: int = 1000
    n_test: int = 1000
    x_dim: int = 2
    eps_grid: List[float] = None
    mu_out: float = 6.0
    contamination_mode: str = "intercept_shift"
    replicates: int = 30
    gamma: float = 0.5
    student_t_df: float = 4.0
    base_seed: int = 42
    output_dir: str = "outputs/exp1_regression_eps_sweep_mu6"
    num_warmup: int = 2000
    num_samples: int = 4000
    chains: int = 4
    eta_mode: str = "response_var"
    eta_gamma_div: float = 1.0
    eta_fixed: float = 1.0
    holder_phi: str | None = None
    holder_kappa: float | None = None
    target_accept: float = 0.9
    init_step: float = 0.5

    def __post_init__(self):
        if self.eps_grid is None:
            self.eps_grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]


def run_single_replicate(cfg: SweepConfig, eps_true: float, rep_id: int) -> Dict[str, Any]:
    eps_key = int(round(eps_true * 100))
    seed = cfg.base_seed + 1000 * rep_id + 10000 * eps_key

    recorded = CONTEXT.get("recorded_inputs")
    if recorded is not None:
        data = {k: recorded[k] for k in ["x_train", "y_train", "x_test", "y_test", "mask_outlier", "beta_true", "sigma_true"]}
        x_train, y_train, x_test = recorded["X"], recorded["y"], recorded["X_test"]
        scaler = {k: recorded[f"scaler_{k}"] for k in ["x_mean", "x_std", "y_mean", "y_std"]}
    else:
        data = simulate_regression_dataset(
            n=cfg.n,
            n_test=cfg.n_test,
            eps=eps_true,
            mu_out=cfg.mu_out,
            seed=seed,
            x_dim=cfg.x_dim,
            contamination_mode=cfg.contamination_mode,
        )
        x_train, y_train, x_test, scaler = standardize_regression_data(
            data["x_train"], data["y_train"], data["x_test"]
        )

    fit_res = fit_regression_methods(
        x_train,
        y_train,
        seed=seed,
        chains=cfg.chains,
        num_warmup=cfg.num_warmup,
        num_samples=cfg.num_samples,
        gamma=cfg.gamma,
        eta_mode=cfg.eta_mode,
        eta_gamma_div=cfg.eta_gamma_div,
        eta_fixed=cfg.eta_fixed,
        holder_phi=cfg.holder_phi,
        holder_kappa=cfg.holder_kappa,
        student_t_df=cfg.student_t_df,
        target_accept=cfg.target_accept,
        init_step=cfg.init_step,
    )

    proposal_methods = get_proposal_methods(cfg)
    all_methods = get_all_methods(cfg)
    primary_proposal = get_primary_proposal_method(cfg)
    results = {}
    k_oracle = int(np.sum(data["mask_outlier"]))
    proposal_extras = {}
    for method in proposal_methods:
        proposal_extras[method] = compute_dpd_outlier_prob(fit_res[method]["samples"], x_train, y_train, seed=seed)
    k_est = int(proposal_extras[primary_proposal]["k_mean"])

    score_arrays = {}
    for method in all_methods:
        pred = summarize_posterior_prediction(fit_res[method]["samples"], x_test, scaler)
        metrics = evaluate_regression_metrics(
            pred["y_pred"],
            data["y_test"],
            pred["beta_hat"],
            data["beta_true"],
            pred["sigma_hat"],
            data["sigma_true"],
        )
        if method in proposal_methods:
            metrics.update(summarize_dpd_posterior(fit_res[method]["samples"], eps_true=eps_true))
            metrics.update(
                evaluate_detection_metrics(
                    proposal_extras[method]["p_out"],
                    data["mask_outlier"],
                    k_oracle=k_oracle,
                    k_est=k_est,
                    is_prob=True,
                )
            )
            metrics["eta"] = fit_res[method]["eta"]
            metrics["cov_det"] = fit_res[method]["cov_det"]
            metrics["holder_phi"] = fit_res[method]["holder_phi"]
            metrics["holder_kappa"] = fit_res[method]["holder_kappa"]
        else:
            likelihood = "student_t" if method == "StudentT" else "gaussian"
            scores = score_baseline(
                fit_res[method]["samples"],
                x_train,
                y_train,
                likelihood=likelihood,
                student_t_df=cfg.student_t_df,
                seed=seed,
            )
            score_arrays[f"score_{METHOD_JSON_KEYS[method]}"] = np.asarray(scores)
            metrics.update(
                evaluate_detection_metrics(
                    scores,
                    data["mask_outlier"],
                    k_oracle=k_oracle,
                    k_est=k_est,
                    is_prob=False,
                )
            )
            if method == "StudentT":
                metrics["nu"] = cfg.student_t_df
        metrics.update(fit_metadata(fit_res[method]))
        results[method] = metrics

    results["_raw"] = {
        "temperature_version": np.asarray(TEMPERATURE_VERSION),
        "gamma": np.asarray(cfg.gamma),
        "eta_mode": np.asarray(cfg.eta_mode),
        "eta_gamma_div": np.asarray(cfg.eta_gamma_div),
        "x_train": data["x_train"],
        "y_train": data["y_train"],
        "x_test": data["x_test"],
        "y_test": data["y_test"],
        "mask_outlier": data["mask_outlier"],
        "beta_true": data["beta_true"],
        "sigma_true": np.asarray(data["sigma_true"]),
        "k_est_dpd": np.asarray(k_est),
        "samples_dpd": np.asarray(fit_res[primary_proposal]["samples"]),
        "samples_nll": np.asarray(fit_res["NLL"]["samples"]),
        "samples_gamma": np.asarray(fit_res["GammaDiv"]["samples"]),
        "samples_student_t": np.asarray(fit_res["StudentT"]["samples"]),
        "p_out_dpd": np.asarray(proposal_extras[primary_proposal]["p_out"]),
        "proposal_method_primary": np.asarray(primary_proposal),
    }
    for method in proposal_methods:
        results["_raw"][f"samples_{METHOD_JSON_KEYS[method]}"] = np.asarray(fit_res[method]["samples"])
        results["_raw"][f"p_out_{METHOD_JSON_KEYS[method]}"] = np.asarray(proposal_extras[method]["p_out"])
    results["_raw"].update(score_arrays)
    return results


def save_single_run_results(output_root, eps_true, rep, cfg, results, save_samples=False):
    run_dir = os.path.join(output_root, f"eps{eps_true:.2f}", f"rep{rep:03d}")
    os.makedirs(run_dir, exist_ok=True)
    out_json = {
        "temperature_version": TEMPERATURE_VERSION,
        "run_config": configuration(cfg, __file__),
        "eps_true": eps_true,
        "mu_out": cfg.mu_out,
        "contamination_mode": cfg.contamination_mode,
        "n": cfg.n,
        "n_test": cfg.n_test,
        "rep": rep,
        "proposal_methods": get_proposal_methods(cfg),
        "holder_phi": cfg.holder_phi,
        "holder_kappa": cfg.holder_kappa,
    }
    for method in get_all_methods(cfg):
        out_json[METHOD_JSON_KEYS[method]] = results[method]
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(out_json, f, indent=2, cls=NumpyEncoder)
    if save_samples and "_raw" in results:
        np.savez_compressed(os.path.join(run_dir, "samples.npz"), **results["_raw"])


def generate_summary_plots(df_agg, output_dir):
    eps_values = sorted(df_agg["eps_true"].unique())
    method_order = ["DPD", "ExpSatDPD", "RatSatDPD", "NLL", "GammaDiv", "StudentT"]
    methods = [m for m in method_order if m in df_agg["method"].unique()]
    styles = {
        "DPD": {"color": "#4c78a8", "marker": "o"},
        "ExpSatDPD": {"color": "#e45756", "marker": "v"},
        "RatSatDPD": {"color": "#72b7b2", "marker": "P"},
        "NLL": {"color": "#2ca02c", "marker": "s"},
        "GammaDiv": {"color": "#f58518", "marker": "^"},
        "StudentT": {"color": "#b279a2", "marker": "D"},
    }

    def _plot(metric, ylabel, filename):
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        for method in methods:
            df_m = df_agg[df_agg["method"] == method].sort_values("eps_true")
            style = styles[method]
            ax.errorbar(
                df_m["eps_true"],
                df_m[f"{metric}_mean"],
                yerr=df_m[f"{metric}_std"],
                fmt=f"-{style['marker']}",
                color=style["color"],
                capsize=3,
                label=method,
            )
        ax.set_xlabel("True contamination epsilon")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
        plt.close(fig)

    _plot("rmse_test", "Test RMSE vs epsilon", "fig_rmse_vs_eps_mu6.png")
    _plot("auprc", "AUPRC vs epsilon", "fig_auprc_vs_eps_mu6.png")

    proposal_methods = [
        m for m in methods if "eps_post_mean_mean" in df_agg.columns and not df_agg[df_agg["method"] == m]["eps_post_mean_mean"].isna().all()
    ]
    if proposal_methods:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot(eps_values, eps_values, "k--", label="True epsilon")
        for method in proposal_methods:
            df_method = df_agg[df_agg["method"] == method].sort_values("eps_true")
            style = styles.get(method, {"color": "gray", "marker": "x"})
            ax.errorbar(
                df_method["eps_true"],
                df_method["eps_post_mean_mean"],
                yerr=df_method["eps_post_mean_std"],
                fmt=f"-{style['marker']}",
                color=style["color"],
                capsize=4,
                label=f"{method} posterior mean",
            )
        ax.set_xlabel("True contamination epsilon")
        ax.set_ylabel("Estimated epsilon")
        ax.set_title("Proposal epsilon estimation (mu_out = 6)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(os.path.join(output_dir, "fig_eps_estimation_mu6.png"), bbox_inches="tight")
        plt.close(fig)

        if "eps_ci95_cover_mean" in df_agg.columns:
            fig_cov, ax_cov = plt.subplots(1, 1, figsize=(7, 5))
            for method in proposal_methods:
                df_method = df_agg[df_agg["method"] == method].sort_values("eps_true")
                style = styles.get(method, {"color": "gray", "marker": "x"})
                ax_cov.plot(
                    df_method["eps_true"],
                    df_method["eps_ci95_cover_mean"],
                    f"-{style['marker']}",
                    color=style["color"],
                    label=method,
                )
            ax_cov.axhline(0.95, color="k", linestyle="--", label="Target 0.95")
            ax_cov.set_xlabel("True contamination epsilon")
            ax_cov.set_ylabel("Coverage probability")
            ax_cov.set_ylim(0.0, 1.05)
            ax_cov.set_title("DPD epsilon CI coverage (mu_out = 6)")
            ax_cov.grid(True, alpha=0.3)
            ax_cov.legend()
            fig_cov.savefig(os.path.join(output_dir, "fig_eps_coverage_mu6.png"), bbox_inches="tight")
            plt.close(fig_cov)


def parse_eps_grid(value: str) -> List[float]:
    tokens = value.replace(",", " ").split()
    return [float(v) for v in tokens]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int)
    parser.add_argument("--n_test", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--mu_out", type=float)
    parser.add_argument("--contamination_mode", choices=["intercept_shift", "leverage"])
    parser.add_argument("--eps_grid", type=str, help="Space- or comma-separated epsilon values")
    parser.add_argument("--student_t_df", type=float)
    parser.add_argument("--chains", type=int)
    parser.add_argument("--num_warmup", type=int)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--eta_mode", type=str)
    parser.add_argument("--eta_gamma_div", type=float, default=1.0)
    parser.add_argument("--eta_fixed", type=float)
    parser.add_argument("--holder_phi", type=str)
    parser.add_argument("--holder_kappa", type=float)
    parser.add_argument("--target_accept", type=float)
    parser.add_argument("--init_step", type=float)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    cfg = SweepConfig()
    cfg.eta_gamma_div = positive_temperature(args.eta_gamma_div)
    if args.n is not None:
        cfg.n = args.n
    if args.n_test is not None:
        cfg.n_test = args.n_test
    if args.gamma is not None:
        cfg.gamma = args.gamma
    if args.mu_out is not None:
        cfg.mu_out = args.mu_out
    if args.contamination_mode is not None:
        cfg.contamination_mode = args.contamination_mode
    if args.eps_grid is not None:
        cfg.eps_grid = parse_eps_grid(args.eps_grid)
    if args.student_t_df is not None:
        cfg.student_t_df = args.student_t_df
    if args.chains is not None:
        cfg.chains = args.chains
    if args.num_warmup is not None:
        cfg.num_warmup = args.num_warmup
    if args.num_samples is not None:
        cfg.num_samples = args.num_samples
    if args.replicates is not None:
        cfg.replicates = args.replicates
    if args.eta_mode is not None:
        cfg.eta_mode = args.eta_mode
    if args.eta_fixed is not None:
        cfg.eta_fixed = args.eta_fixed
    if args.holder_phi is not None:
        cfg.holder_phi = args.holder_phi
    if args.holder_kappa is not None:
        cfg.holder_kappa = args.holder_kappa
    if args.target_accept is not None:
        cfg.target_accept = args.target_accept
    if args.init_step is not None:
        cfg.init_step = args.init_step
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir

    begin_run(cfg, __file__)
    os.makedirs(cfg.output_dir, exist_ok=True)
    all_rows = []
    failures = []
    start_time = time.time()

    for eps_true in cfg.eps_grid:
        print(f"=== Starting epsilon {eps_true:.2f} (mu_out={cfg.mu_out}) ===")
        for rep in range(cfg.replicates):
            try:
                res = run_single_replicate(cfg, eps_true, rep)
                if hasattr(jax, "clear_caches"):
                    jax.clear_caches()
                gc.collect()

                save_single_run_results(cfg.output_dir, eps_true, rep, cfg, res, save_samples=(rep == 0))
                if rep == 0:
                    generate_representative_plots_from_raw(
                        res["_raw"],
                        condition=f"mu{int(cfg.mu_out)}_eps{eps_true:.2f}",
                        output_dir=cfg.output_dir,
                        student_t_df=cfg.student_t_df,
                    )

                for method in get_all_methods(cfg):
                    row = {
                        "eps_true": eps_true,
                        "mu_out": cfg.mu_out,
                        "replicate": rep,
                        "method": method,
                    }
                    row.update(res[method])
                    all_rows.append(row)
            except Exception as e:
                failures.append({"eps_true": eps_true, "replicate": rep, "error": str(e)})
                print(f"    ! Error in rep {rep}: {e}")
                import traceback

                traceback.print_exc()

    record_completion(cfg, len(cfg.eps_grid) * cfg.replicates,
        len(all_rows) // len(get_all_methods(cfg)), failures)
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(cfg.output_dir, "summary_metrics.csv"), index=False)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    group_keys = ["eps_true", "mu_out", "method"]
    numeric_cols = [c for c in numeric_cols if c not in group_keys and c != "replicate"]
    df_agg = df.groupby(group_keys)[numeric_cols].agg(["mean", "std"]).reset_index()
    df_agg.columns = ["_".join(col).strip() if col[1] else col[0] for col in df_agg.columns.values]
    df_agg.to_csv(os.path.join(cfg.output_dir, "summary_agg.csv"), index=False)

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f}s")
    generate_summary_plots(df_agg, cfg.output_dir)


if __name__ == "__main__":
    main()
