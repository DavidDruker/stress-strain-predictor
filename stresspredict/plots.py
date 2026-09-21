"""Figures for the report, drawn from saved out-of-fold predictions.

Reads what `evaluate` already wrote rather than re-running cross-validation, so
a figure can never disagree with the table it sits next to.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import schema, splits  # noqa: E402

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

PALETTE = {
    "dummy": "#9aa0a6", "ridge": "#4c78a8", "random_forest": "#72b7b2",
    "extra_trees": "#54a24b", "hist_gbm": "#e45756",
}
LABEL = {"yield_strength": "Yield strength (MPa)",
         "tensile_strength": "Tensile strength (MPa)",
         "elongation": "Elongation (%)"}


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def parity(oof: pd.DataFrame, protocol: str, model: str, out: Path) -> Path:
    """Predicted vs measured for all three landmarks under one protocol."""
    sub = oof[(oof["protocol"] == protocol) & (oof["model"] == model)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    for ax, target in zip(axes, schema.TARGETS, strict=True):
        yt = sub[f"{target}__true"].to_numpy(dtype=float)
        yp = sub[f"{target}__pred"].to_numpy(dtype=float)
        m = np.isfinite(yt) & np.isfinite(yp)
        yt, yp = yt[m], yp[m]

        ax.scatter(yt, yp, s=11, alpha=0.45, color=PALETTE.get(model, "#4c78a8"),
                   edgecolors="none")
        lo = float(min(yt.min(), yp.min())) if len(yt) else 0.0
        hi = float(max(yt.max(), yp.max())) if len(yt) else 1.0
        pad = 0.04 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#333", lw=1.0, ls="--")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f"measured -- {LABEL[target]}")
        ax.set_ylabel("predicted")
        mae = float(np.mean(np.abs(yp - yt))) if len(yt) else float("nan")
        ax.set_title(f"{target}   MAE {mae:.1f}   n={len(yt)}", fontsize=10)
        _style(ax)

    fig.suptitle(f"Out-of-fold parity -- {model}, {protocol}", fontsize=12)
    fig.tight_layout()
    path = out / f"parity_{protocol}_{model}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def leak_plot(bundle: dict, target: str, out: Path) -> Path:
    """The cost of grade leakage: the same models under both protocols.

    The gap between the two bars is the size of the leak. Publishing it is the
    point -- a project that reported only the left-hand bar would look better and
    mean less.
    """
    protocols = [p for p in ("random_kfold", "gkf_grade") if p in bundle["protocols"]]
    models = list(bundle["protocols"][protocols[0]]["models"])

    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.8 / len(protocols)
    x = np.arange(len(models))

    for i, protocol in enumerate(protocols):
        vals = []
        for model in models:
            lm = bundle["protocols"][protocol]["models"][model].get("landmarks", {})
            vals.append(lm.get(target, {}).get("full", {}).get("mae", np.nan))
        bars = ax.bar(x + i * width, vals, width * 0.92,
                      label=f"{protocol}" + (" (leaky)" if protocol == "random_kfold" else ""),
                      color="#c8c8c8" if protocol == "random_kfold" else "#4c78a8",
                      edgecolor="white", linewidth=0.8)
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)

    ax.set_xticks(x + width * (len(protocols) - 1) / 2)
    ax.set_xticklabels(models, rotation=12, ha="right")
    ax.set_ylabel(f"MAE -- {LABEL[target]}")
    ax.set_title(f"What grade leakage buys you: {target}", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    path = out / f"leak_{target}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def floor_plot(bundle: dict, manifest: dict, target: str, out: Path) -> Path:
    """Model MAE against the measured physical floor.

    The shaded band is the within-composition spread on measured rows: the error
    a perfect composition-only model would still make, because the rows inside it
    differ by processing that the features cannot see.
    """
    protocol = splits.HEADLINE_PROTOCOL
    models = list(bundle["protocols"][protocol]["models"])
    vals = []
    for model in models:
        lm = bundle["protocols"][protocol]["models"][model].get("landmarks", {})
        vals.append(lm.get(target, {}).get("full", {}).get("mae", np.nan))

    floor = manifest["noise_floor"]["targets"][target]["measured_only"]
    lo, hi = floor.get("mae_floor_insample"), floor.get("mae_floor_loo")

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.bar(models, vals, 0.6,
                  color=[PALETTE.get(m, "#4c78a8") for m in models],
                  edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.0f", fontsize=9, padding=2)

    if lo is not None and hi is not None:
        ax.axhspan(lo, hi, color="#e45756", alpha=0.13, zorder=0)
        ax.axhline(lo, color="#e45756", lw=1.1, ls="--")
        ax.axhline(hi, color="#e45756", lw=1.1, ls="--")
        ax.text(len(models) - 0.45, hi, "  measured noise floor\n  (processing, not chemistry)",
                va="bottom", ha="right", fontsize=8.5, color="#a33")

    ax.set_ylabel(f"MAE -- {LABEL[target]}")
    ax.set_title(f"{target}: model error vs the physical floor ({protocol})", fontsize=12)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=12, ha="right")
    _style(ax)
    fig.tight_layout()
    path = out / f"floor_{target}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def lofo_plot(bundle: dict, model: str, target: str, out: Path) -> Path:
    """Per-family error, worst fold visible. A pooled mean would hide it."""
    protocol = "lofo_family"
    if protocol not in bundle["protocols"]:
        return Path()
    entry = bundle["protocols"][protocol]["models"][model]
    component = "tensile_strength" if target == "tensile_strength" else (
        "yield_ratio" if target == "yield_strength" else "elongation")
    comp = entry["components"].get(component, {})
    folds, labels = comp.get("per_fold", []), comp.get("fold_labels", [])
    if not folds:
        return Path()

    pairs = sorted(((lab, f.get("mae", np.nan)) for lab, f in zip(labels, folds, strict=True)),
                   key=lambda p: -(p[1] if np.isfinite(p[1]) else -1))
    labels, vals = zip(*pairs, strict=True)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#e45756"] + ["#4c78a8"] * (len(vals) - 1)
    bars = ax.barh(range(len(vals)), vals, 0.68, color=colors, edgecolor="white", linewidth=0.7)
    ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"held-out-family MAE -- {component}")
    ax.set_title(f"Leave-one-family-out, {model}: the worst family is the honest headline",
                 fontsize=11)
    _style(ax)
    fig.tight_layout()
    path = out / f"lofo_{model}_{component}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def banana_plot(oof: pd.DataFrame, protocol: str, model: str, out: Path) -> Path:
    """Strength-ductility space: the standard axis engineers actually read."""
    sub = oof[(oof["protocol"] == protocol) & (oof["model"] == model)]
    m = (sub["tensile_strength__true"].notna() & sub["elongation__true"].notna()
         & sub["elongation__pred"].notna())
    sub = sub[m]

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.scatter(sub["elongation__true"], sub["tensile_strength__true"],
               s=22, alpha=0.5, label="measured", color="#4c78a8", edgecolors="none")
    ax.scatter(sub["elongation__pred"], sub["tensile_strength__pred"],
               s=22, alpha=0.5, label="predicted", color="#e45756", marker="^",
               edgecolors="none")
    ax.set_xlabel("Elongation (%)")
    ax.set_ylabel("Tensile strength (MPa)")
    ax.set_title(f"Strength-ductility space ({model}, {protocol})", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    path = out / f"banana_{protocol}_{model}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Draw report figures from saved results.")
    ap.add_argument("--report", type=Path, default=REPORTS_DIR / "eval_ratio_comp_all.json")
    ap.add_argument("--oof", type=Path, default=REPORTS_DIR / "oof_eval_ratio_comp_all.csv")
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/data_manifest.json"))
    ap.add_argument("--model", default="hist_gbm")
    ap.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = ap.parse_args(argv)

    if not args.report.exists():
        raise FileNotFoundError(
            f"{args.report} not found. Run: python -m stresspredict.evaluate --protocol all"
        )
    bundle = json.loads(args.report.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    oof = pd.read_csv(args.oof)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = [
        parity(oof, splits.HEADLINE_PROTOCOL, args.model, args.out_dir),
        parity(oof, "random_kfold", args.model, args.out_dir),
        banana_plot(oof, splits.HEADLINE_PROTOCOL, args.model, args.out_dir),
    ]
    for target in schema.TARGETS:
        written.append(leak_plot(bundle, target, args.out_dir))
        written.append(floor_plot(bundle, manifest, target, args.out_dir))
    written.append(lofo_plot(bundle, args.model, "tensile_strength", args.out_dir))
    written.append(lofo_plot(bundle, args.model, "yield_strength", args.out_dir))

    for p in written:
        if p and str(p):
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
