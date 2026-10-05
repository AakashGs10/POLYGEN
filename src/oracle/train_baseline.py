import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.oracle import features


def baseline_mae(y_train, y_test):
    return mean_absolute_error(y_test, np.full_like(y_test, np.median(y_train)))


def train_property(df, X, prop):
    mask = df[prop].notna().values
    n = int(mask.sum())
    if n < 50:
        print(f"{prop:9s} skipped, only {n} labels")
        return None

    smiles = df.loc[mask, "smiles"].tolist()
    y = df.loc[mask, prop].values
    Xp = X[mask]

    tr, te, n_scaffolds = features.scaffold_split(smiles, test_frac=0.2)
    if len(te) < 10:
        print(f"{prop:9s} skipped, scaffold test set too small")
        return None

    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=None,
        min_samples_leaf=8, l2_regularization=1.0, random_state=config.SEED,
    )
    model.fit(Xp[tr], y[tr])
    pred = model.predict(Xp[te])

    mae = mean_absolute_error(y[te], pred)
    r2 = r2_score(y[te], pred)
    dumb = baseline_mae(y[tr], y[te])
    spread = float(np.std(y))

    resid = y[te] - pred
    sigma = float(np.std(resid))

    print(f"{prop:9s} n={n:6,d}  scaffolds={n_scaffolds:5,d}  "
          f"MAE={mae:9.3f}  median-MAE={dumb:9.3f}  "
          f"lift={1 - mae/dumb:6.1%}  R2={r2:7.3f}  sd(y)={spread:9.3f}")

    with open(config.ORACLE_DIR / f"{prop}.pkl", "wb") as fh:
        pickle.dump({"model": model, "sigma": sigma}, fh)

    return {
        "property": prop, "n_labels": n, "n_scaffolds": n_scaffolds,
        "mae": float(mae), "median_baseline_mae": float(dumb),
        "lift_over_median": float(1 - mae / dumb), "r2": float(r2),
        "std_y": spread, "residual_sigma": sigma,
    }


def run():
    if not config.OPC_PROCESSED.exists():
        print(f"missing {config.OPC_PROCESSED}; run src/data/properties.py first")
        return
    df = pd.read_csv(config.OPC_PROCESSED)
    print(f"featurizing {len(df):,} molecules")
    X, ok = features.featurize(df["smiles"].tolist())
    df = df.loc[ok].reset_index(drop=True)
    print(f"feature matrix: {X.shape}\n")

    print(f"{'property':9s} {'labels':>8s}  {'scaffolds':>9s}  {'MAE':>13s}  "
          f"{'median-MAE':>14s}  {'lift':>10s}  {'R2':>10s}  {'sd(y)':>13s}")
    results = [r for r in (train_property(df, X, p) for p in config.PROPERTIES) if r]

    out = config.ORACLE_DIR / "baseline_report.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    print("\nlift = reduction in MAE vs predicting the training median.")
    print("A property with low lift is not learnable from this data at this size.")


if __name__ == "__main__":
    run()
