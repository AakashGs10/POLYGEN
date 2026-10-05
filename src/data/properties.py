import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.data import chem

EXPECTED_COLS = {"SMILES", "Tg", "FFV", "Tc", "Density", "Rg"}


def audit(df):
    print(f"rows: {len(df):,}")
    print(f"{'property':10s} {'labelled':>9s} {'missing':>9s} {'min':>10s} {'median':>10s} {'max':>10s}")
    for prop in config.PROPERTIES:
        col = df[prop]
        n = col.notna().sum()
        print(f"{prop:10s} {n:9,d} {1 - n/len(df):8.1%} "
              f"{col.min():10.3f} {col.median():10.3f} {col.max():10.3f}")


def check_tg_units(df):
    tg = df["Tg"].dropna()
    if tg.empty:
        return
    if tg.min() < 0:
        verdict = "Celsius (negative values present)"
    elif tg.median() > 200:
        verdict = "Kelvin (median above 200)"
    else:
        verdict = "AMBIGUOUS - inspect manually"
    print(f"\nTg unit check: {verdict}")
    print(f"  range {tg.min():.1f} to {tg.max():.1f}, median {tg.median():.1f}")


def run():
    if not config.OPC_RAW.exists():
        print(f"missing: {config.OPC_RAW}")
        print("download train.csv from the Open Polymer Challenge and place it there")
        return None

    df = pd.read_csv(config.OPC_RAW)
    missing_cols = EXPECTED_COLS - set(df.columns)
    if missing_cols:
        print(f"unexpected schema, missing columns: {sorted(missing_cols)}")
        print(f"found: {list(df.columns)}")
        return None

    print("=== raw ===")
    audit(df)
    check_tg_units(df)

    canon, selfies, keep = [], [], []
    for smiles in df["SMILES"].astype(str):
        sel = chem.smiles_to_selfies(smiles)
        if sel is None:
            canon.append(None); selfies.append(None); keep.append(False)
            continue
        canon.append(smiles.replace("*", "[At]"))
        selfies.append(sel)
        keep.append(True)

    df = df.assign(selfies=selfies).loc[keep].copy()
    df["smiles"] = [chem.selfies_to_smiles(s) for s in df["selfies"]]
    df = df.dropna(subset=["smiles"]).drop_duplicates(subset="smiles")

    print(f"\n=== after chem validation ===")
    audit(df)

    cols = ["smiles", "selfies"] + config.PROPERTIES
    df[cols].to_csv(config.OPC_PROCESSED, index=False)
    print(f"\nwrote {config.OPC_PROCESSED}")
    return df


if __name__ == "__main__":
    run()
