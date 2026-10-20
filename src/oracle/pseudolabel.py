"""Pseudo-label PI1M with the trained Oracle to build the C-VAE training set.

PI1M has structures but no property labels, and a conditional VAE needs
(structure, property) pairs. This script loads the Oracle, predicts all five
properties for every preprocessed PI1M polymer, flags whether each prediction
sits inside the Oracle's calibrated range (`in_domain`), and writes:

  data/processed/pi1m_pseudolabelled.csv   full table (analysis, analyze_pseudolabels.py)
  data/processed/cvae_train_slim.csv        in-domain selfies + *_pred (C-VAE input)

Run after the Oracle is trained (src/oracle/train_baseline.py):

    python src/oracle/pseudolabel.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.oracle import features


def run():
    if not config.SELFIES_ALL.exists():
        print(f"missing {config.SELFIES_ALL}; run src/data/preprocess.py first")
        return
    oracles = {p: pickle.load(open(config.ORACLE_DIR / f"{p}.pkl", "rb"))
               for p in config.PROPERTIES if (config.ORACLE_DIR / f"{p}.pkl").exists()}
    if not oracles:
        print(f"no oracle models in {config.ORACLE_DIR}; run train_baseline.py first")
        return

    df = pd.read_csv(config.SELFIES_ALL)
    print(f"featurizing {len(df):,} polymers")
    X, ok = features.featurize(df["smiles"].tolist())
    df = df.loc[ok].reset_index(drop=True)

    # calibrated range per property = 1st–99th percentile of the Oracle's
    # training labels; a prediction outside it is extrapolation (out-of-domain)
    label_q = {}
    if config.OPC_PROCESSED.exists():
        opc = pd.read_csv(config.OPC_PROCESSED)
        for p in oracles:
            s = opc[p].dropna()
            if len(s):
                label_q[p] = (float(s.quantile(0.01)), float(s.quantile(0.99)))

    in_domain = np.ones(len(df), dtype=bool)
    out = df[["smiles", "selfies"]].copy()
    for p, bundle in oracles.items():
        pred = bundle["model"].predict(X)
        out[f"{p}_pred"] = pred
        if p in label_q:
            lo, hi = label_q[p]
            in_domain &= (pred >= lo) & (pred <= hi)
    out["in_domain"] = in_domain

    full = config.DATA_PROCESSED / "pi1m_pseudolabelled.csv"
    out.to_csv(full, index=False)
    print(f"wrote {full}  ({len(out):,} rows, {int(in_domain.sum()):,} in-domain)")

    cols = ["selfies"] + [f"{p}_pred" for p in oracles]
    slim = config.DATA_PROCESSED / "cvae_train_slim.csv"
    out.loc[out["in_domain"], cols].to_csv(slim, index=False)
    print(f"wrote {slim}  ({int(in_domain.sum()):,} rows) -> C-VAE training input")


if __name__ == "__main__":
    run()
