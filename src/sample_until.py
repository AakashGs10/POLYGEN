"""Rejection sampling: keep generating until N polymers meet the request.

Single-shot generation gives a low yield on hard (multi-property / extreme)
targets. This wrapper samples in batches, keeps only the candidates whose
Oracle-predicted properties satisfy every parsed constraint, and stops once it
has collected `k` accepted polymers or exhausts a sampling budget.

    python src/sample_until.py "high Tg and low density"            # k=20
    python src/sample_until.py "glass transition above 250" 30 4.0  # k, guidance

Writes the accepted set to results/accepted_latest.csv.
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
BATCH = 256
MAX_SAMPLES = 8000


def run(query, k=20, guidance=GUIDANCE, batch=BATCH, max_samples=MAX_SAMPLES):
    parsed = ConstraintParser().parse(query)
    print(f'query      : "{query}"')
    if not parsed["active"]:
        print("no parseable constraints; nothing to sample toward.")
        return
    for p, c in parsed["constraints"].items():
        print(f"    {p:8s} {c['op']:>2s} {c['value']}  [{c['qualifier']}]")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, mean, std = gen.load()
    model = model.to(dev)
    oracles = {p: pickle.load(open(config.ORACLE_DIR / f"{p}.pkl", "rb"))
               for p in parsed["active"] if (config.ORACLE_DIR / f"{p}.pkl").exists()}
    train_smiles = set(pd.read_csv(config.SELFIES_ALL, usecols=["smiles"])["smiles"])

    accepted, seen = [], set()
    total, valid = 0, 0
    while len(accepted) < k and total < max_samples:
        z = torch.randn(batch, LATENT, device=dev)
        cond, uncond = gen.make_cond(model, z, parsed["targets"], parsed["mask"],
                                     mean, std, dev)
        selfies = gen.sample(model, vocab, cond, batch, dev,
                             uncond=uncond, guidance=guidance)
        total += batch

        smiles = []
        for s in selfies:
            sm = gen.chem.selfies_to_smiles(s) if s else None
            if sm and sm.count("*") == 2:
                smiles.append(sm)
        valid += len(smiles)
        if not smiles:
            continue

        X, _ = features.featurize(smiles)
        ok = np.ones(X.shape[0], dtype=bool)
        preds = {}
        for p in parsed["active"]:
            if p not in oracles:
                continue
            preds[p] = oracles[p]["model"].predict(X)
            ok &= _satisfies(parsed["constraints"][p]["op"], preds[p],
                             parsed["constraints"][p]["value"], oracles[p].get("sigma", 1.0))

        for i in range(len(smiles)):
            if ok[i] and smiles[i] not in seen:
                seen.add(smiles[i])
                row = {"smiles": smiles[i], "novel": smiles[i] not in train_smiles}
                for p in preds:
                    row[f"{p}_pred"] = round(float(preds[p][i]), 4)
                accepted.append(row)
                if len(accepted) >= k:
                    break

    yield_rate = len(accepted) / total if total else 0.0
    print(f"\nsampled {total} | valid {valid} ({valid/total:.0%}) | "
          f"accepted {len(accepted)} | yield {yield_rate:.2%}")
    if not accepted:
        print("no candidate satisfied every constraint within the sampling budget.")
        print("try raising guidance, raising the budget, or relaxing the request.")
        return

    df = pd.DataFrame(accepted)
    out = Path(config.ROOT) / "results" / "accepted_latest.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    n_novel = int(df["novel"].sum())
    print(f"novel: {n_novel}/{len(df)}   ->  wrote {out}\n")
    print(df.head(min(k, 15)).to_string(index=False))


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "high Tg and low density"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    g = float(sys.argv[3]) if len(sys.argv) > 3 else GUIDANCE
    run(query, k, g)
