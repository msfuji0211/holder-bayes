# Source-only release

The public distribution is `holder-bayes`, with import/command `hbayes`. Its two entry points are the reusable `fit_density` / `fit_regression` API and `hbayes paper run` for the paper campaign.

The prior bundle's `reference/`, `provenance/`, numerical replay commands and fixed campaign recovery scripts have been removed. `hbayes check` now checks source and configuration availability. `hbayes paper tables` calculates tables from the selected new campaign's summary CSVs. Install the `test` extra for numerical tests, or the `experiments` extra for inference and the full campaign.

New runs require generated synthetic observations and raw real datasets, not old samples or result archives. Runtime-generated cache contracts remain strict; the new runner does not auto-migrate old receipts. Keep existing reviewed outputs separately and use a new output directory with this version.
