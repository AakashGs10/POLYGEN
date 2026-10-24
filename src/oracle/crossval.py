"""Grouped scaffold cross-validation for the Oracle -- error bars on every metric.

`train_baseline.py` fits one deterministic scaffold split per property, which
gives a point estimate and no way to attach a confidence interval. This runs
k-fold Bemis-Murcko-grouped cross-validation instead, so each property reports
mean +/- sd over folds.

    python src/oracle/crossval.py --folds 5

Writes results/oracle_cv.md and results/oracle_cv.json. These are the figures
the manuscript should quote; the single-split numbers remain in
models/oracle/baseline_report.json for comparison.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.oracle import features

OUT_MD = config.ROOT / "results" / "oracle_cv.md"
OUT_JSON = config.ROOT / "results" / "oracle_cv.json"


def _model():
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=None,
        min_samples_leaf=8, l2_regularization=1.0, random_state=config.SEED)


def cv_property(df, X, prop, folds, seed):
    mask = df[prop].notna().values
    n = int(mask.sum())
    if n < 50:
        print(f"{prop:9s} skipped, only {n} labels")
        return None

    smiles = df.loc[mask, "smiles"].tolist()
    y = df.loc[mask, prop].values
    Xp = X[mask]

    maes, r2s, lifts = [], [], []
    for tr, te in features.scaffold_kfold(smiles, k=folds, seed=seed):
        if len(te) < 5:
            continue
        m = _model().fit(Xp[tr], y[tr])
        pred = m.predict(Xp[te])
        mae = mean_absolute_error(y[te], pred)
        base = mean_absolute_error(y[te], np.full_like(y[te], np.median(y[tr])))
        maes.append(mae)
        r2s.append(r2_score(y[te], pred))
        lifts.append((base - mae) / base if base else 0.0)

    if not maes:
        print(f"{prop:9s} skipped, no usable folds")
        return None

    res = {
        "property": prop, "n_labels": n, "folds": len(maes),
        "mae_mean": float(np.mean(maes)), "mae_sd": float(np.std(maes)),
        "r2_mean": float(np.mean(r2s)), "r2_sd": float(np.std(r2s)),
        "lift_mean": float(np.mean(lifts)), "lift_sd": float(np.std(lifts)),
    }
    print(f"{prop:9s} n={n:6,d}  folds={len(maes)}  "
          f"MAE {res['mae_mean']:.3f} +/- {res['mae_sd']:.3f}  "
          f"R2 {res['r2_mean']:.3f} +/- {res['r2_sd']:.3f}  "
          f"lift {res['lift_mean']:.1%} +/- {res['lift_sd']:.1%}")
    return res


def run(folds=5, seed=config.SEED):
    df = pd.read_csv(config.OPC_PROCESSED)
    X, keep = features.featurize(df["smiles"].tolist())
    df = df.loc[keep].reset_index(drop=True)
    print(f"{len(df):,} molecules survive featurisation; {folds}-fold "
          f"scaffold CV\n")

    rows = [r for r in (cv_property(df, X, p, folds, seed)
                        for p in config.PROPERTIES if p in df.columns)
            if r is not None]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rows, indent=2))
    with open(OUT_MD, "w") as fh:
        fh.write(f"# Oracle accuracy, {folds}-fold grouped scaffold CV\n\n")
        fh.write("Bemis-Murcko scaffold groups are kept intact within a fold. "
                 "Values are mean +/- sd over folds.\n\n")
        fh.write("| Property | Labels | MAE | Lift over median | R2 |\n")
        fh.write("|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['property']} | {r['n_labels']:,} | "
                     f"{r['mae_mean']:.3f} ± {r['mae_sd']:.3f} | "
                     f"{r['lift_mean']:.1%} ± {r['lift_sd']:.1%} | "
                     f"{r['r2_mean']:.3f} ± {r['r2_sd']:.3f} |\n")
    print(f"\nwrote {OUT_MD} and {OUT_JSON}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=config.SEED)
    a = ap.parse_args()
    run(a.folds, a.seed)
