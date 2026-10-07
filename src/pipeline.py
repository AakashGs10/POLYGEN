"""End-to-end closed loop: natural language -> polymers that meet the request.

    python src/pipeline.py "high Tg, low density" 200

Tier 1 (parser) turns the text into a constraint block, Tier 2 (C-VAE)
generates candidates conditioned on those targets, Tier 3 (Oracle) scores them,
and we report the fraction of generations that actually satisfy each constraint.
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

GUIDANCE = 3.0


def _satisfies(op, pred, value, sigma):
    if op == ">=":
        return pred >= value
    if op == ">":
        return pred > value
    if op == "<=":
        return pred <= value
    if op == "<":
        return pred < value
    if op == "~":
        return np.abs(pred - value) <= max(sigma, 1e-9)
    return np.zeros_like(pred, dtype=bool)


def run(query, n=200, guidance=GUIDANCE):
    parser = ConstraintParser()
    parsed = parser.parse(query)

    print(f'query     : "{query}"')
    if not parsed["active"]:
        print("no parseable property constraints found; nothing to condition on.")
        if parsed["unparsed"]:
            print("unparsed  :", parsed["unparsed"])
        return
    print("constraints:")
    for p, c in parsed["constraints"].items():
        print(f"    {p:8s} {c['op']:>2s} {c['value']}  [{c['qualifier']}]")
    if parsed["unparsed"]:
        print("unparsed  :", parsed["unparsed"])

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, mean, std = gen.load()
    model = model.to(dev)

    oracles = {}
    for p in config.PROPERTIES:
        f = config.ORACLE_DIR / f"{p}.pkl"
        if f.exists():
            oracles[p] = pickle.load(open(f, "rb"))

    train_smiles = set(pd.read_csv(config.SELFIES_ALL, usecols=["smiles"])["smiles"])

    z = torch.randn(n, LATENT, device=dev)
    cond, uncond = gen.make_cond(model, z, parsed["targets"], parsed["mask"],
                                 mean, std, dev)
    selfies = gen.sample(model, vocab, cond, n, dev, uncond=uncond, guidance=guidance)

    # decode to valid two-endpoint polymers
    smiles = []
    for s in selfies:
        out = gen.chem.selfies_to_smiles(s) if s else None
        if out and out.count("*") == 2:
            smiles.append(out)
    validity = len(smiles) / n
    novel = len(set(smiles) - train_smiles) / len(set(smiles)) if smiles else 0.0
    print(f"\ngenerated : {n} sampled | validity {validity:.1%} | "
          f"novelty {novel:.1%} | guidance {guidance}")

    if not smiles:
        print("no valid polymers to score.")
        return

    X, ok = features.featurize(smiles)
    print(f"\n{'property':8s} {'target':>12s} {'oracle mean':>12s} {'satisfied':>10s}")
    overall = np.ones(X.shape[0], dtype=bool)
    for p, c in parsed["constraints"].items():
        if p not in oracles:
            print(f"{p:8s} (no oracle model)")
            continue
        pred = oracles[p]["model"].predict(X)
        sigma = oracles[p].get("sigma", 1.0)
        sat = _satisfies(c["op"], pred, c["value"], sigma)
        overall &= sat
        print(f"{p:8s} {c['op']+' '+str(c['value']):>12s} "
              f"{pred.mean():12.3f} {sat.mean():9.1%}")
    print(f"\nmeet ALL constraints simultaneously: {overall.mean():.1%} "
          f"({overall.sum()}/{len(overall)} valid polymers)")

    keep = [smiles[i] for i in range(len(smiles)) if overall[i]][:5]
    if keep:
        print("\nexamples meeting every constraint:")
        for s in keep:
            print(" ", s)


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "high Tg, low density"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    g = float(sys.argv[3]) if len(sys.argv) > 3 else GUIDANCE
    run(query, n, g)
