"""Scoring, reported per fold and never only pooled.

LOFO produces folds of wildly uneven size. A single pooled MAE over concatenated
predictions lets one catastrophic family hide behind eleven good ones, so the
per-fold array is kept and the median AND worst fold are both reported. For any
extrapolation claim the worst fold is the honest headline.
"""

from __future__ import annotations

import numpy as np


def _finite_pairs(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[m], y_pred[m]


def score(y_true, y_pred, with_mape: bool = True) -> dict:
    """Metrics on the original scale. MAE leads; see targets.py for why."""
    yt, yp = _finite_pairs(y_true, y_pred)
    if yt.size == 0:
        return {"n": 0}
    err = yp - yt
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    out = {
        "n": int(yt.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        # Reported for comparability with the literature, never led with: R2 is
        # the metric most inflated by grade leakage.
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "median_ae": float(np.median(np.abs(err))),
    }
    if with_mape:
        nz = yt != 0
        out["mape"] = float(np.mean(np.abs(err[nz] / yt[nz])) * 100.0) if nz.any() else float("nan")
    return out


def aggregate_folds(per_fold: list[dict], metric: str = "mae") -> dict:
    """Summarise a metric across folds, keeping the worst fold visible."""
    vals = [f[metric] for f in per_fold if f.get("n") and metric in f]
    if not vals:
        return {"n_folds": 0}
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n_folds": 0}
    return {
        "n_folds": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        # The fold-to-fold spread that licenses "no XGBoost": a model difference
        # smaller than this is not a difference.
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "sem": float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0,
        "worst": float(arr.max()),
        "best": float(arr.min()),
    }


def round_floats(obj, dp: int = 4):
    """Recursively round for readable, diff-able committed JSON."""
    if isinstance(obj, dict):
        return {k: round_floats(v, dp) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, dp) for v in obj]
    # bool before int: bool is a subclass of int, and a JSON `true` that silently
    # became `1` is the kind of thing nobody notices until a report reads wrong.
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else round(float(obj), dp)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    return obj
