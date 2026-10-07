"""Benchmark the closed loop across representative queries.

For each query: parse -> generate N candidates conditionally (guided) AND
unconditionally (property channel masked off) -> score with the Oracle ->
report the fraction satisfying every constraint. The conditional-minus-
unconditional lift is the evidence that conditioning does targeted work.

    python src/benchmark.py            # default suite, N=300
    python src/benchmark.py 500        # N per query/arm

Writes results/benchmark.md.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.rag.parser import ConstraintParser
from src.cvae import generate as gen
from src.cvae.train import LATENT
from src.oracle import features
from src.pipeline import _satisfies

GUIDANCE = 3.0
QUERIES = [
    "high Tg",
    "low density",
    "high free fractional volume",
    "large radius of gyration",
    "high Tg and low density",
    "heat resistant but lightweight",
    "glass transition above 200",
]


def _valid_smiles(selfies):
    out = []
    for s in selfies:
        sm = gen.chem.selfies_to_smiles(s) if s else None
        if sm and sm.count("*") == 2:
            out.append(sm)
    return out


def _score(smiles, constraints, oracles):
    """Return (per-prop satisfaction dict, all-constraint mask) for a smiles list."""
    if not smiles:
        return {}, np.zeros(0, dtype=bool)
    X, _ = features.featurize(smiles)
    per, overall = {}, np.ones(X.shape[0], dtype=bool)
    for p, c in constraints.items():
        if p not in oracles:
            continue
        pred = oracles[p]["model"].predict(X)
        sat = _satisfies(c["op"], pred, c["value"], oracles[p].get("sigma", 1.0))
        per[p] = (float(sat.mean()), float(pred.mean()))
        overall &= sat
    return per, overall


def run(n=300, guidance=GUIDANCE):
    parser = ConstraintParser()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, mean, std = gen.load()
    model = model.to(dev)
    oracles = {p: pickle.load(open(config.ORACLE_DIR / f"{p}.pkl", "rb"))
               for p in config.PROPERTIES if (config.ORACLE_DIR / f"{p}.pkl").exists()}
    train_smiles = set(pd.read_csv(config.SELFIES_ALL, usecols=["smiles"])["smiles"])

    rows = []
    detail = []
    for q in QUERIES:
        parsed = parser.parse(q)
        if not parsed["active"]:
            continue
        z = torch.randn(n, LATENT, device=dev)

        cond, uncond = gen.make_cond(model, z, parsed["targets"], parsed["mask"],
                                     mean, std, dev)
        cond_sf = gen.sample(model, vocab, cond, n, dev, uncond=uncond, guidance=guidance)
        # unconditional arm: property channel fully masked, no guidance
        zero = [0.0] * len(config.PROPERTIES)
        base_cond, _ = gen.make_cond(model, z, parsed["targets"], zero, mean, std, dev)
        base_sf = gen.sample(model, vocab, base_cond, n, dev, guidance=0.0)

        cond_sm, base_sm = _valid_smiles(cond_sf), _valid_smiles(base_sf)
        cper, coverall = _score(cond_sm, parsed["constraints"], oracles)
        bper, boverall = _score(base_sm, parsed["constraints"], oracles)

        validity = len(cond_sm) / n
        novelty = len(set(cond_sm) - train_smiles) / len(set(cond_sm)) if cond_sm else 0.0
        c_yield = float(coverall.mean()) if len(coverall) else 0.0
        b_yield = float(boverall.mean()) if len(boverall) else 0.0

        rows.append((q, "+".join(parsed["active"]), validity, novelty,
                     b_yield, c_yield))
        detail.append((q, cper, bper))

    # ---- render ----
    n_train = sum(1 for _ in open(config.DATA_PROCESSED / "cvae_train_slim.csv")) - 1
    lines = ["# Closed-loop benchmark", "",
             f"Model: C-VAE trained on {n_train:,} rows. N={n} per arm, guidance={guidance}.",
             "Yield = fraction of *valid* generations satisfying every constraint.",
             "`uncond` masks the property channel off (no steering); `cond` is the",
             "guided conditional generation. Lift = cond − uncond.", "",
             "| query | props | validity | novelty | uncond yield | cond yield | lift |",
             "|-------|-------|----------|---------|--------------|------------|------|"]
    for q, props, val, nov, by, cy in rows:
        lines.append(f"| {q} | {props} | {val:.0%} | {nov:.0%} | "
                     f"{by:.1%} | **{cy:.1%}** | {cy-by:+.1%} |")
    lines += ["", "## Per-property detail (satisfaction | Oracle mean)", ""]
    for q, cper, bper in detail:
        lines.append(f"**{q}**")
        for p in cper:
            cs, cm = cper[p]
            bs, bm = bper.get(p, (0.0, float('nan')))
            lines.append(f"- {p}: cond {cs:.0%} (mean {cm:.2f}) vs "
                         f"uncond {bs:.0%} (mean {bm:.2f})")
        lines.append("")

    out = Path(config.ROOT) / "results" / "benchmark.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    run(n)
