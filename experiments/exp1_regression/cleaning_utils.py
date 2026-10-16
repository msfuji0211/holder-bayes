"""Shared helpers for real-data cleaning / safe batch update experiments."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def sample_base_batch_test(
    x_all: np.ndarray,
    y_all: np.ndarray,
    n_base: int,
    n_batch: int,
    n_test: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_total = x_all.shape[0]
    required = n_base + n_batch + n_test
    if n_total < required:
        raise ValueError(f"Not enough rows: required {required}, got {n_total}")

    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=required, replace=False)

    base_idx = idx[:n_base]
    batch_idx = idx[n_base : n_base + n_batch]
    test_idx = idx[n_base + n_batch :]

    return (
        x_all[base_idx],
        y_all[base_idx],
        x_all[batch_idx],
        y_all[batch_idx],
        x_all[test_idx],
        y_all[test_idx],
    )


def contaminate_batch_response(
    y_clean: np.ndarray,
    eps: float,
    mu_out: float,
    seed: int,
    mode: str = "shift",
    unit: str = "std",
    t_df: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_clean = np.asarray(y_clean, dtype=float)
    n = len(y_clean)
    mask = rng.random(n) < eps
    y_out = y_clean.copy()

    y_scale = float(np.std(y_clean))
    if not np.isfinite(y_scale) or y_scale < 1e-12:
        y_scale = 1.0

    scale = y_scale if unit == "std" else 1.0

    if mask.any():
        m = int(mask.sum())
        mode_l = str(mode).lower().replace("-", "_")

        if mode_l == "shift":
            delta = rng.normal(loc=mu_out * scale, scale=1.0 * scale, size=m)
        elif mode_l in {"heavy_tail", "heavytail"}:
            delta = rng.standard_t(df=float(t_df), size=m) * mu_out * scale
        else:
            raise ValueError(f"Unknown batch contamination mode: {mode}")

        y_out[mask] = y_clean[mask] + delta

    return y_out, mask.astype(bool)


def evaluate_batch_filtering(
    batch_removed_local: np.ndarray,
    mask_batch_outlier: np.ndarray,
    n_batch: int,
) -> Dict[str, Any]:
    removed = np.zeros(n_batch, dtype=bool)
    if batch_removed_local.size > 0:
        removed[np.asarray(batch_removed_local, dtype=int)] = True

    true_out = np.asarray(mask_batch_outlier, dtype=bool)

    tp = int(np.sum(removed & true_out))
    fp = int(np.sum(removed & ~true_out))
    fn = int(np.sum(~removed & true_out))
    tn = int(np.sum(~removed & ~true_out))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    false_removal_rate = fp / max(int(np.sum(~true_out)), 1)

    return {
        "k_removed_batch": int(np.sum(removed)),
        "batch_retained_fraction": float(1.0 - np.mean(removed)),
        "removed_precision": float(precision),
        "removed_recall": float(recall),
        "removed_f1": float(f1),
        "batch_inlier_false_removal_rate": float(false_removal_rate),
        "tp_removed": tp,
        "fp_removed": fp,
        "fn_remaining_outliers": fn,
        "tn_retained_inliers": tn,
    }
