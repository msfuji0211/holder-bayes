import argparse
import os
import sys
import gc

# --- JAX Memory/Thread Limit Setup ---
# Set environment variables to prevent LLVM OOM errors on CPU clusters.
# These must be set BEFORE importing jax or other libraries that use it.

# Force libc to release memory to OS more aggressively
if "MALLOC_TRIM_THRESHOLD_" not in os.environ:
    os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"

# Limit intra-op parallelism
if "XLA_FLAGS" not in os.environ:
    # Reduce thread count for compilation and execution
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

if "MKL_NUM_THREADS" not in os.environ:
    os.environ["MKL_NUM_THREADS"] = "1"
# -------------------------------------

import json
import time
import jax
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from hbayes.paper_sampling import CONTEXT
from hbayes.experiment_temperature import (TEMPERATURE_VERSION, covariance_temperature,
    response_temperature, positive_temperature, fit_metadata, nuts_summary, convergence_summary,
    configuration, begin_run, record_completion)
import matplotlib.pyplot as plt

    # Import utility functions from utils_exp1
from experiments.exp1_simulation.utils_exp1 import (
    simulate_dataset,
    fit_methods,
    compute_dpd_outlier_prob,
    score_baseline,
    evaluate_metrics,
    plot_reliability_diagram,
    resolve_proposal_method_specs,
)

# --- Experiment Runner ---

@dataclass
class RunConfig:
    n: int = 500
    eps_grid: List[float] = None
    replicates: int = 30
    gamma: float = 0.5
    student_t_df: float = 4.0
    base_seed: int = 42
    output_dir: str = "outputs/exp1_simulation"
    scenarios: List[str] = None
    num_warmup: int = 1000
    num_samples: int = 1000
    chains: int = 4
    standardize: bool = True
    eta_mode: str = "covdet"
    eta_gamma_div: float = 1.0
    holder_phi: str | None = None
    holder_kappa: float = None
    out_dist: str = None
    out_mean_x: float = None
    out_mean_y: float = None
    out_scale_x: float = None
    out_scale_y: float = None
    target_accept: float = 0.8
    init_step: float = 1.0

    def __post_init__(self):
        if self.eps_grid is None:
            self.eps_grid = [0.0, 0.1, 0.2, 0.3, 0.4]
        if self.scenarios is None:
            self.scenarios = ["A", "B"]


METHOD_JSON_KEYS = {
    "DPD": "dpd",
    "ExpSatDPD": "exp_sat_dpd",
    "RatSatDPD": "rat_sat_dpd",
    "NLL": "standard",
    "GammaDiv": "gamma",
    "StudentT": "student_t",
}


def get_proposal_methods(cfg: RunConfig) -> List[str]:
    return list(resolve_proposal_method_specs(cfg.holder_phi, cfg.holder_kappa).keys())


def get_primary_proposal_method(cfg: RunConfig) -> str:
    proposal_methods = get_proposal_methods(cfg)
    if "DPD" in proposal_methods:
        return "DPD"
    return proposal_methods[0]


def get_all_methods(cfg: RunConfig) -> List[str]:
    return get_proposal_methods(cfg) + ["NLL", "GammaDiv", "StudentT"]

def run_single_replicate(
    cfg: RunConfig, 
    scenario: str, 
    eps: float, 
    rep_id: int
) -> Dict[str, Any]:
    
    # Deterministic seed
    eps_int = int(round(eps * 10))
    seed = cfg.base_seed + 1000 * rep_id + 10 * eps_int
    if scenario == "B": seed += 50000 
    
    # 1. Simulate Data
    out_mean = None
    if cfg.out_mean_x is not None and cfg.out_mean_y is not None:
        out_mean = (cfg.out_mean_x, cfg.out_mean_y)
    
    out_scale = None
    if cfg.out_scale_x is not None and cfg.out_scale_y is not None:
        out_scale = (cfg.out_scale_x, cfg.out_scale_y)
        
    recorded = CONTEXT.get("recorded_density_inputs")
    if recorded is not None:
        x, y_true = recorded["x"], recorded["y_true"]
    else:
        x, y_true = simulate_dataset(cfg.n, eps, scenario, seed,
                                     out_dist=cfg.out_dist, out_mean=out_mean, out_scale=out_scale)
    n_outliers_true = np.sum(y_true)
    
    # 2. Fit Models
    # Returns: { "dpd_joint": ..., "standard": ..., "gamma": ... }
    fit_res = fit_methods(
        x, seed, 
        chains=cfg.chains, 
        num_warmup=cfg.num_warmup, 
        num_samples=cfg.num_samples,
        gamma=cfg.gamma,
        standardize=cfg.standardize,
        eta_mode=cfg.eta_mode,
        eta_gamma_div=cfg.eta_gamma_div,
        holder_phi=cfg.holder_phi,
        holder_kappa=cfg.holder_kappa,
        student_t_df=cfg.student_t_df,
        target_accept=cfg.target_accept,
        init_step=cfg.init_step
    )
    
    k_oracle = int(n_outliers_true)
    proposal_methods = get_proposal_methods(cfg)
    primary_proposal = get_primary_proposal_method(cfg)
    proposal_outputs = {
        method: compute_dpd_outlier_prob(fit_res[method]["samples"], x, seed=seed) for method in proposal_methods
    }
    k_est = int(proposal_outputs[primary_proposal]["k_mean"])

    results = {}
    for method in proposal_methods:
        proposal_out = proposal_outputs[method]
        p_contam = proposal_out["p_out"]
        eps_post_mean = proposal_out["eps_samples"].mean()
        eps_post_median = np.median(proposal_out["eps_samples"])
        eps_ci95 = np.percentile(proposal_out["eps_samples"], [2.5, 97.5])
        metrics_prop = {
            "eps_true": eps,
            "eps_post_mean": eps_post_mean,
            "eps_post_median": eps_post_median,
            "eps_post_ci95_low": float(eps_ci95[0]),
            "eps_post_ci95_high": float(eps_ci95[1]),
            "eps_ci95_cover": int((eps >= eps_ci95[0]) and (eps <= eps_ci95[1])),
            "k_oracle": k_oracle,
            "k_est": k_est,
        }
        eval_prop = evaluate_metrics(
            p_contam,
            y_true,
            k_oracle=k_oracle,
            k_est=k_est,
            is_prob=True,
        )
        metrics_prop.update(eval_prop)
        results[method] = metrics_prop

    # Plot Reliability Diagram for representative case (e.g. rep=0 and eps=0.2)
    # We check condition here, but we need output_dir to save.
    # Pass output_dir via cfg? cfg is in scope? 
    # run_single_replicate receives cfg.
    if rep_id == 0 and abs(eps - 0.2) < 1e-9:
        eval_primary = results[primary_proposal]
        if "calib_data" in eval_primary:
            fig_cal, ax_cal = plt.subplots(1, 1, figsize=(5, 5))
            plot_reliability_diagram(
                ax_cal,
                eval_primary["calib_data"],
                title=f"Reliability Diagram ({primary_proposal}, {scenario}, ε={eps:.1f})",
            )
            
            # Save to scenario root or specific eps folder?
            # Requirement 6.3: fig_reliability_dpd_{scenario}_eps0.20.png
            # Let's save it in the experiment root output dir
            out_path = os.path.join(cfg.output_dir, f"fig_reliability_dpd_{scenario}_eps{eps:.2f}.png")
            fig_cal.savefig(out_path, bbox_inches="tight")
            plt.close(fig_cal)
            print(f"    Saved reliability diagram to {out_path}")
            
    # Remove calib_data from metrics dict before returning to avoid cluttering JSON/CSV
    for method in proposal_methods:
        if "calib_data" in results[method]:
            del results[method]["calib_data"]

    score_arrays = {}
    # 4. Evaluate Baselines
    baseline_specs = [
        ("NLL", "standard", "gaussian"),
        ("GammaDiv", "gamma", "gaussian"),
        ("StudentT", "student_t", "student_t"),
    ]
    for name, key, likelihood in baseline_specs:
        samples = fit_res[key]["samples"]
        score = score_baseline(samples, x, likelihood=likelihood, student_t_df=cfg.student_t_df, seed=seed)
        
        # Evaluate (is_prob=False, k_est comes from DPD's estimate or just ignored?)
        # User requirement: "top-k recall (k is oracle or DPD k_est)"
        eval_base = evaluate_metrics(
            score, y_true, 
            k_oracle=k_oracle, k_est=k_est, # Re-use DPD's k_est
            is_prob=False
        )
        score_arrays[f"score_{METHOD_JSON_KEYS[name]}"] = np.asarray(score)
        results[name] = eval_base

    # Parameter estimation bias (mu)
    def calc_param_metrics(samples, prefix):
        pm_x = np.mean(samples[:, 0])
        pm_y = np.mean(samples[:, 1])
        bias_norm = np.sqrt(pm_x**2 + pm_y**2)
        return {f"{prefix}_bias_norm": bias_norm}
        
    for method in proposal_methods:
        results[method].update(calc_param_metrics(fit_res[method]["samples"], "mu"))
    results["NLL"].update(calc_param_metrics(fit_res["standard"]["samples"], "mu"))
    results["GammaDiv"].update(calc_param_metrics(fit_res["gamma"]["samples"], "mu"))
    results["StudentT"].update(calc_param_metrics(fit_res["student_t"]["samples"], "mu"))
    results["StudentT"]["nu"] = cfg.student_t_df

    # Keep representative raw posterior draws so samples.npz can be written for rep=0.
    results["_raw"] = {
        "temperature_version": np.asarray(TEMPERATURE_VERSION),
        "gamma": np.asarray(cfg.gamma),
        "eta_mode": np.asarray(cfg.eta_mode),
        "eta_gamma_div": np.asarray(cfg.eta_gamma_div),
        "x": np.asarray(x),
        "y_true": np.asarray(y_true),
        "samples_dpd": np.asarray(fit_res[primary_proposal]["samples"]),
        "samples_nll": np.asarray(fit_res["standard"]["samples"]),
        "samples_gamma": np.asarray(fit_res["gamma"]["samples"]),
        "samples_student_t": np.asarray(fit_res["student_t"]["samples"]),
        "eps_samples": np.asarray(proposal_outputs[primary_proposal]["eps_samples"]),
        "p_out": np.asarray(proposal_outputs[primary_proposal]["p_out"]),
        "k_est": int(k_est),
        "eta": float(fit_res[primary_proposal]["eta"]),
        "proposal_method_primary": np.asarray(primary_proposal),
    }
    for method in proposal_methods:
        results["_raw"][f"samples_{METHOD_JSON_KEYS[method]}"] = np.asarray(fit_res[method]["samples"])
        results["_raw"][f"eps_samples_{METHOD_JSON_KEYS[method]}"] = np.asarray(proposal_outputs[method]["eps_samples"])
        results["_raw"][f"p_out_{METHOD_JSON_KEYS[method]}"] = np.asarray(proposal_outputs[method]["p_out"])

    for method in get_all_methods(cfg):
        key = method if method in proposal_methods else METHOD_JSON_KEYS[method]
        results[method].update(fit_metadata(fit_res[key]))
    results["_raw"].update(score_arrays)
    return results

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_single_run_results(output_root, scenario, eps, rep, cfg, results, save_samples=False):
    """
    Save results for a single run to outputs/exp1/{scenario}/eps{eps:.2f}/rep{rep:03d}/.
    """
    run_dir = os.path.join(output_root, f"{scenario}", f"eps{eps:.2f}", f"rep{rep:03d}")
    os.makedirs(run_dir, exist_ok=True)
    
    # metrics.json (exclude _raw)
    out_json = {
        "temperature_version": TEMPERATURE_VERSION,
        "run_config": configuration(cfg, __file__),
        "scenario": scenario,
        "n": cfg.n,
        "eps_true": eps,
        "rep": rep,
        "proposal_methods": get_proposal_methods(cfg),
        "holder_phi": cfg.holder_phi,
        "holder_kappa": cfg.holder_kappa,
    }
    for method in get_all_methods(cfg):
        out_json[METHOD_JSON_KEYS[method]] = results[method]
    
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(out_json, f, indent=2, cls=NumpyEncoder)
        
    # (Optional) samples.npz
    if save_samples and "_raw" in results:
        raw = results["_raw"]
        np.savez_compressed(os.path.join(run_dir, "samples.npz"), **raw)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--student_t_df", type=float)
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--eta_mode", type=str)
    parser.add_argument("--eta_gamma_div", type=float, default=1.0)
    parser.add_argument("--holder_phi", type=str)
    parser.add_argument("--holder_kappa", type=float)
    parser.add_argument("--chains", type=int)
    parser.add_argument("--num_warmup", type=int)
    parser.add_argument("--num_samples", type=int)
    
    # Outlier config
    parser.add_argument("--outlier", dest="out_dist", type=str)
    parser.add_argument("--out_mean_x", type=float)
    parser.add_argument("--out_mean_y", type=float)
    parser.add_argument("--out_scale_x", type=float)
    parser.add_argument("--out_scale_y", type=float)
    
    # MCMC config
    parser.add_argument("--target_accept", type=float)
    parser.add_argument("--init_step", type=float)
    
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--scenarios", type=str, default="A")
    parser.add_argument("--eps_grid", type=str)

    # Other
    parser.add_argument("--contam", type=float, help="Single contamination level to run")
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()

    cfg = RunConfig()
    cfg.eta_gamma_div = positive_temperature(args.eta_gamma_div)
    
    if args.n is not None: cfg.n = args.n
    if args.replicates is not None: cfg.replicates = args.replicates
    cfg.scenarios = args.scenarios.replace(",", " ").split()
    if not cfg.scenarios or any(v not in {"A", "B"} for v in cfg.scenarios):
        parser.error("--scenarios must contain A and/or B")
    if args.eps_grid is not None:
        cfg.eps_grid = [float(v) for v in args.eps_grid.replace(",", " ").split()]
    if args.gamma is not None: cfg.gamma = args.gamma
    if args.student_t_df is not None: cfg.student_t_df = args.student_t_df
    # RunConfig default standardize is True. Parser default is False (store_true).
    # If user provides --standardize, args.standardize is True.
    # If user does not provide, args.standardize is False.
    # We only assume user wants to CHANGE default if they explicitly pass something? 
    # Actually for store_true, we can't distinguish "not passed" vs "passed".
    # But since default is True, passing --standardize (True) keeps it True.
    # Not passing (False) keeps it True (if we don't assign).
    # So we simply do nothing with standardize here to preserve default=True, 
    # unless we wanted to support --no-standardize.
    # For this specific request, the user passes --standardize so they want True.
    
    if args.eta_mode: cfg.eta_mode = args.eta_mode
    if args.holder_phi: cfg.holder_phi = args.holder_phi
    if args.holder_kappa is not None: cfg.holder_kappa = args.holder_kappa
    if args.chains: cfg.chains = args.chains
    if args.num_warmup: cfg.num_warmup = args.num_warmup
    if args.num_samples: cfg.num_samples = args.num_samples
    
    if args.out_dist: cfg.out_dist = args.out_dist
    if args.out_mean_x is not None: cfg.out_mean_x = args.out_mean_x
    if args.out_mean_y is not None: cfg.out_mean_y = args.out_mean_y
    if args.out_scale_x is not None: cfg.out_scale_x = args.out_scale_x
    if args.out_scale_y is not None: cfg.out_scale_y = args.out_scale_y
    
    if args.target_accept: cfg.target_accept = args.target_accept
    if args.init_step: cfg.init_step = args.init_step
    
    if args.contam is not None:
        cfg.eps_grid = [args.contam]
        
    if args.output_dir: cfg.output_dir = args.output_dir

    begin_run(cfg, __file__)
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    all_results_flat = []
    failures = []
    
    start_time = time.time()
    
    for scenario in cfg.scenarios:
        print(f"=== Starting Scenario {scenario} ===")
        for eps in cfg.eps_grid:
            print(f"  > Contamination {eps:.1%}...")
            for rep in range(cfg.replicates):
                try:
                    # Run single replicate
                    res = run_single_replicate(cfg, scenario, eps, rep)
                    
                    # Clear JAX caches and garbage collect to free memory
                    if hasattr(jax, 'clear_caches'):
                        jax.clear_caches()
                    gc.collect()
                    
                    # Pass n explicitly for JSON
                    res["n"] = cfg.n
                    
                    # Save per-run JSON and optionally Samples
                    # Save samples ONLY for the first replicate (rep=0) to save disk/time
                    # This allows reconstructing the 'representative' plots later.
                    save_samples_flag = (rep == 0)
                    
                    save_single_run_results(cfg.output_dir, scenario, eps, rep, cfg, res, save_samples=save_samples_flag)
                    
                    # Flatten for CSV summary (exclude _raw key)
                    for method in get_all_methods(cfg):
                        row = {
                            "scenario": scenario,
                            "eps_true": eps,
                            "replicate": rep,
                            "method": method,
                        }
                        row.update(res[method])
                        all_results_flat.append(row)
                    
                except Exception as e:
                    failures.append({"scenario": scenario, "eps_true": eps, "replicate": rep, "error": str(e)})
                    print(f"    ! Error in rep {rep}: {e}")
                    import traceback
                    traceback.print_exc()

    record_completion(cfg, len(cfg.scenarios) * len(cfg.eps_grid) * cfg.replicates,
        len(all_results_flat) // len(get_all_methods(cfg)), failures)

    # 5.2 Summary Files
    df = pd.DataFrame(all_results_flat)
    csv_path = os.path.join(cfg.output_dir, "summary_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved summary metrics to {csv_path}")
    
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f}s")
    
    # Aggregated Summary (mean/std)
    # Group by [scenario, eps_true, method]
    # Filter numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude grouping keys from numeric_cols if present
    group_keys = ["scenario", "eps_true", "method"]
    numeric_cols = [c for c in numeric_cols if c not in group_keys and c != "replicate"]
    
    df_agg = df.groupby(group_keys)[numeric_cols].agg(['mean', 'std']).reset_index()
    
    # Flatten column names (e.g. auprc_mean, auprc_std)
    df_agg.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_agg.columns.values]
    
    agg_path = os.path.join(cfg.output_dir, "summary_agg.csv")
    df_agg.to_csv(agg_path, index=False)
    print(f"Saved aggregated summary to {agg_path}")

    # 6. Generate Summary Plots
    generate_summary_plots(df_agg, cfg.output_dir)

def generate_summary_plots(df_agg, output_dir):
    """
    Generate plots for epsilon calibration and AUPRC curves.
    df_agg columns expected: scenario, eps_true, method, [metric]_mean, [metric]_std
    """
    scenarios = df_agg["scenario"].unique()
    
    for scen in scenarios:
        df_scen = df_agg[df_agg["scenario"] == scen]
        
        # 6.1 Epsilon Calibration (DPD only)
        # Filter DPD method
        proposal_methods = [
            m for m in df_scen["method"].unique()
            if "eps_post_mean_mean" in df_scen.columns and not df_scen[df_scen["method"] == m]["eps_post_mean_mean"].isna().all()
        ]
        if proposal_methods:
            fig_eps, ax_eps = plt.subplots(1, 1, figsize=(6, 5))
            eps_true = df_scen["eps_true"].drop_duplicates().sort_values()
            
            # Identity line
            max_eps = eps_true.max()
            ax_eps.plot([0, max_eps], [0, max_eps], "k--", label="True")
            
            # Plot estimates
            styles = {
                "DPD": {"color": "#4c78a8", "marker": "o"},
                "ExpSatDPD": {"color": "#e45756", "marker": "v"},
                "RatSatDPD": {"color": "#72b7b2", "marker": "P"},
            }
            for method in proposal_methods:
                df_prop = df_scen[df_scen["method"] == method].sort_values("eps_true")
                style = styles.get(method, {"color": "gray", "marker": "x"})
                ax_eps.errorbar(
                    df_prop["eps_true"],
                    df_prop["eps_post_mean_mean"],
                    yerr=df_prop["eps_post_mean_std"],
                    fmt=f"{style['marker']}-",
                    color=style["color"],
                    capsize=5,
                    label=f"{method} estimate",
                )
            
            ax_eps.set_xlabel(r"True Contamination $\epsilon_{true}$")
            ax_eps.set_ylabel(r"Posterior Mean $\epsilon$")
            ax_eps.set_title(f"Epsilon Estimation Calibration ({scen})")
            ax_eps.legend()
            ax_eps.grid(True, alpha=0.3)
            
            out_path_eps = os.path.join(output_dir, f"fig_eps_calibration_{scen}.png")
            fig_eps.savefig(out_path_eps, bbox_inches="tight")
            plt.close(fig_eps)
            print(f"Saved epsilon calibration plot to {out_path_eps}")

            # Coverage plot
            if "eps_ci95_cover_mean" in df_scen.columns:
                fig_cov, ax_cov = plt.subplots(1, 1, figsize=(6, 5))
                for method in proposal_methods:
                    df_prop = df_scen[df_scen["method"] == method].sort_values("eps_true")
                    style = styles.get(method, {"color": "gray", "marker": "x"})
                    ax_cov.plot(
                        df_prop["eps_true"],
                        df_prop["eps_ci95_cover_mean"],
                        f"{style['marker']}-",
                        color=style["color"],
                        label=method,
                    )
                ax_cov.axhline(0.95, color="k", linestyle="--", label="Target 0.95")
                ax_cov.set_ylim(0, 1.05)
                ax_cov.set_xlabel(r"True Contamination $\epsilon_{true}$")
                ax_cov.set_ylabel("Coverage Probability")
                ax_cov.set_title(f"Epsilon CI Coverage ({scen})")
                ax_cov.legend()
                ax_cov.grid(True, alpha=0.3)
                
                out_path_cov = os.path.join(output_dir, f"fig_eps_coverage_{scen}.png")
                fig_cov.savefig(out_path_cov, bbox_inches="tight")
                plt.close(fig_cov)
        
        # 6.2 AUPRC Curve (All methods)
        fig_auprc, ax_auprc = plt.subplots(1, 1, figsize=(6, 5))
        
        methods = df_scen["method"].unique()
        # Define styles for known methods
        styles = {
            "DPD": {"color": "#4c78a8", "marker": "o"},
            "ExpSatDPD": {"color": "#e45756", "marker": "v"},
            "RatSatDPD": {"color": "#72b7b2", "marker": "P"},
            "NLL": {"color": "#2ca02c", "marker": "s"},
            "GammaDiv": {"color": "#f58518", "marker": "^"},
            "StudentT": {"color": "#b279a2", "marker": "D"},
        }
        
        for method in methods:
            df_m = df_scen[df_scen["method"] == method]
            eps_x = df_m["eps_true"]
            auprc_mean = df_m["auprc_mean"]
            auprc_std = df_m["auprc_std"]
            
            style = styles.get(method, {"color": "gray", "marker": "x"})
            
            ax_auprc.errorbar(
                eps_x, auprc_mean, yerr=auprc_std,
                fmt=f"-{style['marker']}", color=style['color'],
                capsize=3, label=method, alpha=0.8
            )
            
        ax_auprc.set_xlabel(r"True Contamination $\epsilon_{true}$")
        ax_auprc.set_ylabel("AUPRC")
        ax_auprc.set_title(f"Detection Performance (AUPRC) - {scen}")
        ax_auprc.legend()
        ax_auprc.grid(True, alpha=0.3)
        
        out_path_auprc = os.path.join(output_dir, f"fig_auprc_vs_eps_{scen}.png")
        fig_auprc.savefig(out_path_auprc, bbox_inches="tight")
        plt.close(fig_auprc)
        print(f"Saved AUPRC plot to {out_path_auprc}")

if __name__ == "__main__":
    main()
