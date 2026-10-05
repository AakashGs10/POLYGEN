import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import config

path = config.DATA_PROCESSED / "pi1m_pseudolabelled.csv"
df = pd.read_csv(path)
n = len(df)
ind = df["in_domain"].astype(bool)
print(f"rows {n:,}  |  in-domain {ind.sum():,} ({ind.mean():.1%})\n")

print("=== Tg pseudo-labels: in-domain vs out-of-domain ===")
print(f"{'group':14s} {'n':>9s} {'median':>9s} {'p95':>9s} {'max':>9s}")
for label, mask in [("in-domain", ind), ("out-of-domain", ~ind)]:
    t = df.loc[mask, "Tg_pred"]
    print(f"{label:14s} {len(t):9,d} {t.median():9.1f} {t.quantile(0.95):9.1f} {t.max():9.1f}")

print("\n=== where do high-Tg candidates come from? ===")
print(f"{'threshold':>10s} {'total':>10s} {'in-domain':>11s} {'out-of-dom':>11s} {'% OOD':>8s}")
for thr in [100, 150, 200, 250, 300]:
    hi = df["Tg_pred"] > thr
    a, b = int((hi & ind).sum()), int((hi & ~ind).sum())
    pct = b / (a + b) * 100 if (a + b) else 0.0
    print(f"{thr:10d} {a+b:10,d} {a:11,d} {b:11,d} {pct:7.1f}%")

print("\n=== pseudo-label correlations (Oracle-induced, NOT measured) ===")
props = [f"{p}_pred" for p in config.PROPERTIES]
print(df[props].corr().round(3).to_string())

print("\n=== recommended C-VAE training set ===")
train = df[ind].copy()
print(f"in-domain rows: {len(train):,}")
lo, hi = df["Tg_pred"].quantile([0.01, 0.99])
print(f"Tg 1st-99th pct: {lo:.1f} to {hi:.1f} C")
out = config.DATA_PROCESSED / "cvae_train.csv"
train.to_csv(out, index=False)
print(f"wrote {out}")