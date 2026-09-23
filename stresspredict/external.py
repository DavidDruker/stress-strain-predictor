"""Score a fitted model on the external held-out set, which nothing trains on.

matminer `steel_strength` (Citrine dataset 153092): 312 steels with UTS and YS,
303 with elongation, compositions in wt%. It is the set the shipped model was
promoted on, so the rule was fixed before it was scored, and it has no
composition within 0.3 wt% (per element) of any SteelBench row and no
property overlap with either training source.

It is also a hard set, and that is the reason to use it: every steel has YS >= 1,000 MPa,
mostly maraging and secondary-hardening grades whose strength comes from Co, Ti and W, which are not model inputs. It measures
extrapolation to the strong end of the range, not typical-steel accuracy -- read
it next to the grade-grouped CV, never instead of it.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import ingest, metrics, predict, schema

RAW = ingest.RAW_DIR / "steel_strength.json.gz"
URL = "https://ndownloader.figshare.com/files/13354691"
SHA256 = "e36501d7057cd833223bb8ed9948668b5ac90fd585d29a749f45af51c1d7f6ad"
REPORT = Path("reports/external_steel_strength.json")


def load(path: Path = RAW) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compositions (wt%, Cu not reported) and landmark targets."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Download it first:\n  curl -L -o {path} {URL}")
    if ingest.sha256_of(path) != SHA256:
        raise ValueError(f"{path} does not match the checksum this evaluation was run on")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        raw = json.load(fh)
    df = pd.DataFrame(raw["data"], columns=raw["columns"])
    X = pd.DataFrame({el: pd.to_numeric(df[el.lower()], errors="coerce")
                      if el.lower() in df else np.nan for el in schema.ELEMENTS})
    y = pd.DataFrame({
        "yield_strength": df["yield strength"].astype(float),
        "tensile_strength": df["tensile strength"].astype(float),
        "elongation": pd.to_numeric(df["elongation"], errors="coerce"),
    })
    return X, y


def score(artifact_path: Path) -> dict:
    X, y = load()
    pred = predict.predict_batch(X, artifact_path=artifact_path)
    out: dict = {"artifact": str(artifact_path).replace("\\", "/")}
    for t in schema.TARGETS:
        s = metrics.score(y[t], pred[t])
        ok = y[t].notna().to_numpy()
        err = (pred[t] - y[t]).to_numpy()[ok]
        s["bias"] = float(err.mean())
        s["within_20pct"] = float(np.mean(np.abs(err) / y[t].to_numpy()[ok] <= 0.2))
        out[t] = s
    out["uts_gt_ys_violations"] = int((pred["tensile_strength"] <= pred["yield_strength"]).sum())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score models on the external held-out set.")
    ap.add_argument("--artifact", type=Path, default=predict.DEFAULT_ARTIFACT)
    ap.add_argument("--baseline", type=Path, help="a second artifact to compare against")
    ap.add_argument("--out", type=Path, default=REPORT)
    args = ap.parse_args(argv)

    result = {"candidate": score(args.artifact)}
    if args.baseline:
        result["baseline"] = score(args.baseline)
        result["mean_mae_ratio"] = float(np.mean([
            result["candidate"][t]["mae"] / result["baseline"][t]["mae"] for t in schema.TARGETS]))

    for name in ("baseline", "candidate"):
        if name in result:
            r = result[name]
            print(f"{name:10s} " + "  ".join(
                f"{t} MAE {r[t]['mae']:.1f}" for t in schema.TARGETS))
    if "mean_mae_ratio" in result:
        print(f"mean MAE ratio candidate / baseline: {result['mean_mae_ratio']:.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics.round_floats(result), indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
