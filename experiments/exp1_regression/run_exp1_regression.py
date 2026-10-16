import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List

# JAX memory/thread limits for cluster runs.
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
from hbayes.experiment_temperature import (TEMPERATURE_VERSION, covariance_temperature,
    response_temperature, positive_temperature, fit_metadata, nuts_summary, convergence_summary,
    configuration, begin_run, record_completion)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.exp1_regression.utils_exp1_regression import (
    compute_dpd_outlier_prob,
    evaluate_detection_metrics,
    evaluate_regression_metrics,
    fit_regression_methods,
    generate_representative_plots_from_raw,
    resolve_proposal_method_specs,
    score_baseline,
    simulate_regression_dataset,
    standardize_regression_data,
    summarize_dpd_posterior,
    summarize_posterior_prediction,
)


@dataclass
class RunConfig:
    n: int = 300
    n_test: int = 1000
    x_dim: int = 2
    conditions: List[str] = None
    replicates: int = 30
    gamma: float = 0.5
    student_t_df: float = 4.0
    base_seed: int = 42
    output_dir: str = "outputs/exp1_regression"
    num_warmup: int = 1000
    num_samples: int = 1000
    chains: int = 4
    eta_mode: str = "response_var"
    eta_gamma_div: float = 1.0
    eta_fixed: float = 1.0
    holder_phi: str | None = None
    holder_kappa: float | None = None
    target_accept: float = 0.8
    init_step: float = 1.0

    def __post_init__(self):
        if self.conditions is None:
            self.conditions = ["clean", "mu3", "mu6"]


CONDITION_SPECS = {
    "clean": {"eps": 0.0, "mu_out": 0.0, "contamination_mode": "leverage"},
    "mu3": {"eps": 0.2, "mu_out": 3.0, "contamination_mode": "leverage"},
    "mu6": {"eps": 0.2, "mu_out": 6.0, "contamination_mode": "leverage"},
}

BASELINE_METHODS = ["NLL", "GammaDiv", "StudentT"]
METHOD_JSON_KEYS = {
    "DPD": "dpd",
    "ExpSatDPD": "exp_sat_dpd",
    "RatSatDPD": "rat_sat_dpd",
    "NLL": "nll",
    "GammaDiv": "gamma",
    "StudentT": "student_t",
}


def get_proposal_methods(cfg: RunConfig) -> List[str]:
    return list(resolve_proposal_method_specs(cfg.holder_phi, cfg.holder_kappa).keys())


def get_all_methods(cfg: RunConfig) -> List[str]:
    return get_proposal_methods(cfg) + BASELINE_METHODS


def get_primary_proposal_method(cfg: RunConfig) -> str:
    proposal_methods = get_proposal_methods(cfg)
    if "DPD" in proposal_methods:
        return "DPD"
    return proposal_methods[0]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def run_single_replicate(cfg: RunConfig, condition: str, rep_id: int) -> Dict[str, Any]:
    spec = CONDITION_SPECS[condition]
    eps_true = spec["eps"]
    mu_out = spec["mu_out"]
    contamination_mode = spec.get("contamination_mode", "intercept_shift")
    seed = cfg.base_seed + 1000 * rep_id + 10000 * list(CONDITION_SPECS).index(condition)

    data = simulate_regression_dataset(
        n=cfg.n,
        n_test=cfg.n_test,
        eps=eps_true,
        mu_out=mu_out,
        seed=seed,
        x_dim=cfg.x_dim,
        contamination_mode=contamination_mode,
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
    all_methods = proposal_methods + BASELINE_METHODS
    primary_proposal = get_primary_proposal_method(cfg)
    results = {}
    k_oracle = int(np.sum(data["mask_outlier"]))
    proposal_extras = {}
    for method in proposal_methods:
        proposal_extras[method] = compute_dpd_outlier_prob(fit_res[method]["samples"], x_train, y_train, seed=seed)
    k_est = int(proposal_extras[primary_proposal]["k_mean"])

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
            proposal_extra = proposal_extras[method]
            metrics.update(summarize_dpd_posterior(fit_res[method]["samples"], eps_true=eps_true))
            metrics.update(
                evaluate_detection_metrics(
                    proposal_extra["p_out"],
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
    return results


def save_single_run_results(output_root, condition, rep, cfg, results, save_samples=False):
    spec = CONDITION_SPECS[condition]
    run_dir = os.path.join(output_root, condition, f"rep{rep:03d}")
    os.makedirs(run_dir, exist_ok=True)

    out_json = {
        "condition": condition,
        "n": cfg.n,
        "n_test": cfg.n_test,
        "eps_true": spec["eps"],
        "mu_out": spec["mu_out"],
        "contamination_mode": spec.get("contamination_mode", "intercept_shift"),
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
    condition_order = [c for c in ["clean", "mu3", "mu6"] if c in df_agg["condition"].unique()]
    method_order = ["DPD", "ExpSatDPD", "RatSatDPD", "NLL", "GammaDiv", "StudentT"]
    methods = [m for m in method_order if m in df_agg["method"].unique()]
    x_pos = np.arange(len(condition_order))
    styles = {
        "DPD": {"color": "#4c78a8", "marker": "o"},
        "ExpSatDPD": {"color": "#e45756", "marker": "v"},
        "RatSatDPD": {"color": "#72b7b2", "marker": "P"},
        "NLL": {"color": "#2ca02c", "marker": "s"},
        "GammaDiv": {"color": "#f58518", "marker": "^"},
        "StudentT": {"color": "#b279a2", "marker": "D"},
    }

    def _plot_metric(metric, ylabel, filename):
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        for method in methods:
            df_m = df_agg[df_agg["method"] == method].set_index("condition").reindex(condition_order).reset_index()
            style = styles[method]
            ax.errorbar(
                x_pos,
                df_m[f"{metric}_mean"],
                yerr=df_m[f"{metric}_std"],
                fmt=f"-{style['marker']}",
                color=style["color"],
                capsize=3,
                label=method,
            )
        ax.set_xticks(x_pos, condition_order)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
        plt.close(fig)

    _plot_metric("rmse_test", "Test RMSE", "fig_rmse_by_condition.png")
    _plot_metric("beta_l2_error", "Beta L2 Error", "fig_beta_l2_by_condition.png")

    proposal_methods = [
        m for m in methods if "eps_post_mean_mean" in df_agg.columns and not df_agg[df_agg["method"] == m]["eps_post_mean_mean"].isna().all()
    ]
    if proposal_methods:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        eps_true = [CONDITION_SPECS[c]["eps"] for c in condition_order]
        ax.plot(x_pos, eps_true, "k--", label="True eps")
        for method in proposal_methods:
            df_method = df_agg[df_agg["method"] == method].set_index("condition").reindex(condition_order).reset_index()
            style = styles.get(method, {"color": "gray", "marker": "x"})
            ax.errorbar(
                x_pos,
                df_method["eps_post_mean_mean"],
                yerr=df_method["eps_post_mean_std"],
                fmt=f"-{style['marker']}",
                color=style["color"],
                capsize=4,
                label=f"{method} posterior mean",
            )
        ax.set_xticks(x_pos, condition_order)
        ax.set_ylabel("Contamination ratio")
        ax.set_title("Proposal contamination estimation")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(os.path.join(output_dir, "fig_eps_estimation.png"), bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int)
    parser.add_argument("--n_test", type=int)
    parser.add_argument("--gamma", type=float)
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
    parser.add_argument("--condition", type=str, help="Run a single condition: clean, mu3, mu6")
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    cfg = RunConfig()
    cfg.eta_gamma_div = positive_temperature(args.eta_gamma_div)
    if args.n is not None:
        cfg.n = args.n
    if args.n_test is not None:
        cfg.n_test = args.n_test
    if args.gamma is not None:
        cfg.gamma = args.gamma
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
    if args.condition is not None:
        cfg.conditions = [args.condition]

    begin_run(cfg, __file__)
    os.makedirs(cfg.output_dir, exist_ok=True)
    all_results_flat = []
    start_time = time.time()

    for condition in cfg.conditions:
        if condition not in CONDITION_SPECS:
            raise ValueError(f"Unknown condition: {condition}")
        spec = CONDITION_SPECS[condition]
        print(f"=== Starting condition {condition} (eps={spec['eps']}, mu_out={spec['mu_out']}) ===")
        for rep in range(cfg.replicates):
            try:
                res = run_single_replicate(cfg, condition, rep)
                if hasattr(jax, "clear_caches"):
                    jax.clear_caches()
                gc.collect()

                save_single_run_results(
                    cfg.output_dir,
                    condition,
                    rep,
                    cfg,
                    res,
                    save_samples=(rep == 0),
                )
                if rep == 0:
                    generate_representative_plots_from_raw(
                        res["_raw"],
                        condition=condition,
                        output_dir=cfg.output_dir,
                        student_t_df=cfg.student_t_df,
                    )

                for method in get_all_methods(cfg):
                    row = {
                        "condition": condition,
                        "eps_true": spec["eps"],
                        "mu_out": spec["mu_out"],
                        "replicate": rep,
                        "method": method,
                    }
                    row.update(res[method])
                    all_results_flat.append(row)
            except Exception as e:
                print(f"    ! Error in rep {rep}: {e}")
                import traceback

                traceback.print_exc()

    df = pd.DataFrame(all_results_flat)
    csv_path = os.path.join(cfg.output_dir, "summary_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved summary metrics to {csv_path}")

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f}s")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    group_keys = ["condition", "eps_true", "mu_out", "method"]
    numeric_cols = [c for c in numeric_cols if c not in group_keys and c != "replicate"]
    df_agg = df.groupby(group_keys)[numeric_cols].agg(["mean", "std"]).reset_index()
    df_agg.columns = ["_".join(col).strip() if col[1] else col[0] for col in df_agg.columns.values]

    agg_path = os.path.join(cfg.output_dir, "summary_agg.csv")
    df_agg.to_csv(agg_path, index=False)
    print(f"Saved aggregated summary to {agg_path}")

    generate_summary_plots(df_agg, cfg.output_dir)


if __name__ == "__main__":
    main()
