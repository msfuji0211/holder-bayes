# Inference and diagnostics

The source entry point is `experiments/run_temperature_workflow.py`, called by `hbayes paper run`. The runner supplies fixed paper settings and uses separate results, calibration and cache directories. Synthetic observations are generated from the native seed rules. Real data are downloaded with `--download-data` or loaded with `--local-data-dir`. No old outputs are required.

The nine stages are `gamma_pilot`, `pilot`, `pilot_realdata`, `synthetic`, `benchmark`, `cleaning`, `review`, `collect` and `plot`. Pilot gates run before production. The deferred diagnostic policy saves completed failures and continues the planned production grid. Final collection requires every fit to pass; all chains and replicates are retained.

Review produces `runs/workflow/diagnostic_review.json` and `runs/workflow/mcmc_repair_queue.json` from the current campaign. These identify its actual failed targets/cache contracts. Do not infer a new campaign's failures from the paper's previous repairs. The source-only package does not include a fixed list of historical failed fits, recovery manifests or adoption receipts.

H-B uses NUTS pilots and frozen Student-t mixture/prior proposals with exact independence MH. GammaDiv retains its multimode frozen proposal and exact correction. NLL and Student-t retain their original posterior targets. Reusable optional composed transitions are in `hb_experiments/kernels/repair_kernel.py` and `baseline_repair_kernel.py`; callers supply the target, proposals, start states and a fixed schedule. They do not read files or silently modify/adopt results. They are not an automatic repair command. If a new campaign remains unresolved, investigate its queue and declare additional sampling before regenerating its metrics/receipts; keep a separate output directory for changed sampler settings.

Restart contracts are generated from actual inputs, code, settings and runtime. Changed settings or altered checkpoints are rejected. This integrity mechanism needs no precomputed checksum file in the repository. The source-only release uses strict contracts; it does not migrate historical campaign receipts automatically.

Data generation remains float32 and inference enables float64 locally. The real-data loader validates the paper dimensions before fitting, including all 32 pumadyn features. The main posterior figure transforms the new regression draws back to original units and includes the intercept. Current summaries and representative inputs supply every plot.

Predictive diagnostic functions are in `hb_experiments/predictive_diagnostics.py`: `diagnose` computes pointwise mixture NLPD, PIT, predictive intervals, coverage and variance components from supplied regression draws and test observations. It uses all supplied draws and no frozen comparison numbers. A passing parameter diagnostic does not certify far-tail predictive accuracy.
