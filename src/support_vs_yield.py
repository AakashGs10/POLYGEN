"""Joint training support vs. conditional generation yield, over property pairs.

The paper's central claim is that conditioning yield tracks the *joint* label
support of the conditioned subset, not its marginal counts. Reporting one pair
(Tg + Density, support 24) shows a collapse but not a relationship. This script
scans every pair across the full support range, so the claim becomes a curve:

    Density + Rg   support 610   -> expect high yield
    FFV + Tc       support 300
    Tc + Density   support 531
    Tg + Tc        support  32
    Tg + Density   support  24
    Tg + FFV       support   1   -> expect collapse

Joint support is recomputed from the labelled file rather than hard-coded, so
the x-axis is measured on the same data the Oracle was trained on.

    python src/support_vs_yield.py 300 3.0 --seeds 3

Writes results/support_vs_yield.csv and results/support_vs_yield.md. With
--seeds > 1 each arm is repeated under a different torch seed and the table
carries mean +/- sd, which is what the manuscript needs for error bars.
"""

import argparse
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.cvae import generate as gen
from src.oracle import features
from src.pipeline import _satisfies
from src.rag.parser import load_ranges

LATENT = 64
OUT_CSV = config.ROOT / "results" / "support_vs_yield.csv"
OUT_MD = config.ROOT / "results" / "support_vs_yield.md"

# Direction each property is pushed in: "high" -> p90, "low" -> p10.
# Chosen once here so every pair is a comparable "top/bottom decile" request.
DIRECTION = {"Tg": "high", "FFV": "high", "Tc": "low",
             "Density": "low", "Rg": "high"}


def joint_support():
    """Observed co-occurrence counts per property pair, from the labelled file."""
    df = pd.read_csv(config.OPC_PROCESSED)
    present = {p: df[p].notna().to_numpy() for p in config.PROPERTIES
               if p in df.columns}
    out = {}
    for a, b in itertools.combinations(sorted(present), 2):
        out[(a, b)] = int((present[a] & present[b]).sum())
    return out, {p: int(v.sum()) for p, v in present.items()}


def build_constraints(pair, ranges):
    """A two-property request at the p10/p90 corner, in parser output shape."""
    targets, mask, constraints = [], [], {}
    for p in config.PROPERTIES:
        if p in pair:
            hi = DIRECTION[p] == "high"
            value = float(ranges[p]["p90" if hi else "p10"])
            constraints[p] = {"op": ">=" if hi else "<=", "value": value,
                              "qualifier": "support-scan", "unit": ""}
            targets.append(value)
            mask.append(1.0)
        else:
            targets.append(float(ranges[p]["p50"]))
            mask.append(0.0)
    return {"constraints": constraints, "targets": targets, "mask": mask,
            "active": list(pair)}


def one_arm(model, vocab, mean, std, dev, oracles, train_smiles,
            parsed, n, guidance, conditional):
    """Generate n molecules and return (validity, novelty, yield)."""
    z = torch.randn(n, LATENT, device=dev)
    cond, uncond = gen.make_cond(model, z, parsed["targets"], parsed["mask"],
                                 mean, std, dev)
    if conditional:
        selfies = gen.sample(model, vocab, cond, n, dev,
                             uncond=uncond, guidance=guidance)
    else:
        # unconditional arm: the trained null reference, no guidance
        selfies = gen.sample(model, vocab, uncond, n, dev, guidance=0.0)

    smiles = []
    for s in selfies:
        sm = gen.chem.selfies_to_smiles(s) if s else None
        if sm and sm.count("*") == 2:
            smiles.append(sm)
    if not smiles:
        return 0.0, 0.0, 0.0

    validity = len(smiles) / n
    novelty = len(set(smiles) - train_smiles) / len(set(smiles))

    X, keep = features.featurize(smiles)
    smiles = [s for s, k in zip(smiles, keep) if k]
    if not smiles:
        return validity, novelty, 0.0

    ok = np.ones(X.shape[0], dtype=bool)
    for p, c in parsed["constraints"].items():
        if p not in oracles:
            continue
        pred = oracles[p]["model"].predict(X)
        ok &= _satisfies(c["op"], pred, c["value"], oracles[p].get("sigma", 1.0))
    return validity, novelty, float(ok.mean())


def run(n=300, guidance=3.0, seeds=1):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, mean, std = gen.load()
    model = model.to(dev).eval()
    oracles = {p: pickle.load(open(config.ORACLE_DIR / f"{p}.pkl", "rb"))
               for p in config.PROPERTIES
               if (config.ORACLE_DIR / f"{p}.pkl").exists()}
    train_smiles = set(pd.read_csv(config.SELFIES_ALL, usecols=["smiles"])["smiles"])
    ranges = load_ranges()
    supports, marginals = joint_support()

    rows = []
    for pair, support in sorted(supports.items(), key=lambda kv: -kv[1]):
        if not set(pair) <= set(oracles):
            continue
        parsed = build_constraints(pair, ranges)
        cond_y, unc_y, vals, novs = [], [], [], []
        for s in range(seeds):
            torch.manual_seed(config.SEED + s)
            v, nov, cy = one_arm(model, vocab, mean, std, dev, oracles,
                                 train_smiles, parsed, n, guidance, True)
            _, _, uy = one_arm(model, vocab, mean, std, dev, oracles,
                               train_smiles, parsed, n, guidance, False)
            cond_y.append(cy); unc_y.append(uy); vals.append(v); novs.append(nov)

        rows.append({
            "pair": " + ".join(pair),
            "joint_support": support,
            "marginal_a": marginals[pair[0]],
            "marginal_b": marginals[pair[1]],
            "validity": float(np.mean(vals)),
            "novelty": float(np.mean(novs)),
            "uncond_yield": float(np.mean(unc_y)),
            "uncond_sd": float(np.std(unc_y)),
            "cond_yield": float(np.mean(cond_y)),
            "cond_sd": float(np.std(cond_y)),
            "lift": float(np.mean(cond_y) - np.mean(unc_y)),
            "seeds": seeds,
        })
        r = rows[-1]
        print(f"{r['pair']:24s} support {support:5d}  "
              f"cond {r['cond_yield']:6.1%} +/- {r['cond_sd']:.1%}  "
              f"uncond {r['uncond_yield']:6.1%}  lift {r['lift']:+.1%}")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    if len(df) > 2:
        x = np.log10(np.maximum(df["joint_support"].to_numpy(), 1))
        y = df["cond_yield"].to_numpy()
        rho = float(np.corrcoef(x, y)[0, 1])
    else:
        rho = float("nan")

    with open(OUT_MD, "w") as fh:
        fh.write("# Joint support vs. conditional yield\n\n")
        fh.write(f"N={n} per arm, guidance={guidance}, seeds={seeds}. "
                 "Yield is the fraction of valid generations meeting every "
                 "constraint, judged by the Oracle.\n\n")
        fh.write("| pair | joint support | validity | novelty | uncond yield "
                 "| cond yield | lift |\n|---|---|---|---|---|---|---|\n")
        for _, r in df.iterrows():
            fh.write(f"| {r['pair']} | {int(r['joint_support'])} | "
                     f"{r['validity']:.0%} | {r['novelty']:.0%} | "
                     f"{r['uncond_yield']:.1%} | **{r['cond_yield']:.1%}** "
                     f"± {r['cond_sd']:.1%} | {r['lift']:+.1%} |\n")
        fh.write(f"\nPearson r(log10 joint support, conditional yield) = "
                 f"{rho:.3f} over {len(df)} pairs.\n")
    print(f"\nr(log10 support, yield) = {rho:.3f}\nwrote {OUT_CSV} and {OUT_MD}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=300)
    ap.add_argument("guidance", nargs="?", type=float, default=3.0)
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    run(a.n, a.guidance, a.seeds)
