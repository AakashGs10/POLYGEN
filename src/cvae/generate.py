import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import selfies as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.cvae.train import CVAE, CKPT, LATENT
from src.data import chem
from src.oracle import features

TEMPERATURE = 0.8


def load():
    ck = torch.load(CKPT / "cvae.pt", map_location="cpu", weights_only=False)
    vocab = ck["vocab"]
    model = CVAE(vocab["vocab_size"], len(config.PROPERTIES),
                 vocab["stoi"][config.PAD_TOKEN])
    model.load_state_dict(ck["model"])
    model.eval()
    return model, vocab, ck["prop_mean"], ck["prop_std"]


@torch.no_grad()
def sample(model, vocab, cond, n, dev, temperature=TEMPERATURE):
    stoi, itos = vocab["stoi"], vocab["itos"]
    bos, eos, pad = (stoi[config.BOS_TOKEN], stoi[config.EOS_TOKEN],
                     stoi[config.PAD_TOKEN])
    h = torch.tanh(model.init_h(cond)).unsqueeze(0).contiguous()
    tok = torch.full((n, 1), bos, dtype=torch.long, device=dev)
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    seqs = [[] for _ in range(n)]

    for _ in range(vocab["max_len"] - 1):
        e = model.embed(tok)
        y, h = model.dec(torch.cat([e, cond.unsqueeze(1)], dim=-1), h)
        logits = model.out(y[:, -1]) / temperature
        logits[:, pad] = -1e9
        nxt = torch.multinomial(torch.softmax(logits, -1), 1)
        for i in range(n):
            if not done[i]:
                t = int(nxt[i])
                if t == eos:
                    done[i] = True
                elif t != bos:
                    seqs[i].append(itos[t])
        if done.all():
            break
        tok = nxt
    return ["".join(s) for s in seqs]


def make_cond(model, targets, mask_flags, mean, std, n, dev):
    z = torch.randn(n, LATENT, device=dev)
    p = np.tile((np.array(targets, dtype=np.float32) - mean) / std, (n, 1))
    m = np.tile(np.array(mask_flags, dtype=np.float32), (n, 1))
    return torch.cat([z, torch.tensor(p * m, device=dev),
                      torch.tensor(m, device=dev)], dim=-1)


def evaluate(selfies_list, train_smiles, oracles):
    smiles, valid = [], 0
    for s in selfies_list:
        if not s:
            continue
        out = chem.selfies_to_smiles(s)
        if out and out.count("*") == 2:
            smiles.append(out)
            valid += 1
    n = len(selfies_list)
    uniq = set(smiles)
    novel = uniq - train_smiles
    stats = {
        "validity": valid / n if n else 0.0,
        "uniqueness": len(uniq) / valid if valid else 0.0,
        "novelty": len(novel) / len(uniq) if uniq else 0.0,
        "smiles": smiles,
    }
    if smiles and oracles:
        X, ok = features.featurize(smiles)
        for prop, bundle in oracles.items():
            stats[prop] = float(np.mean(bundle["model"].predict(X)))
    return stats


def run(n=300):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, mean, std = load()
    model = model.to(dev)
    print(f"device: {dev}  |  sampling {n} per condition\n")

    oracles = {}
    for prop in config.PROPERTIES:
        p = config.ORACLE_DIR / f"{prop}.pkl"
        if p.exists():
            oracles[prop] = pickle.load(open(p, "rb"))

    train_smiles = set(pd.read_csv(config.SELFIES_ALL, usecols=["smiles"])["smiles"])

    tg_i = config.PROPERTIES.index("Tg")
    base = list(mean)
    conditions = []
    for label, tg in [("Tg target  0 C", 0.0), ("Tg target 100 C", 100.0),
                      ("Tg target 200 C", 200.0)]:
        t = list(base)
        t[tg_i] = tg
        flags = [0.0] * len(config.PROPERTIES)
        flags[tg_i] = 1.0
        conditions.append((label, t, flags))

    print(f"{'condition':17s} {'validity':>9s} {'unique':>8s} {'novel':>8s} "
          f"{'Oracle Tg':>10s}")
    results = []
    for label, targets, flags in conditions:
        cond = make_cond(model, targets, flags, mean, std, n, dev)
        gen = sample(model, vocab, cond, n, dev)
        st = evaluate(gen, train_smiles, oracles)
        results.append((label, st))
        print(f"{label:17s} {st['validity']:8.1%} {st['uniqueness']:7.1%} "
              f"{st['novelty']:7.1%} {st.get('Tg', float('nan')):10.1f}")

    lo, hi = results[0][1].get("Tg"), results[-1][1].get("Tg")
    if lo is not None and hi is not None:
        print(f"\nconditioning effect: Oracle Tg shifts {hi - lo:+.1f} C "
              f"across a 200 C target sweep")
        print("(a shift near zero means the conditioning is being ignored)")

    ex = results[-1][1]["smiles"][:5]
    print("\nexamples at Tg target 200 C:")
    for s in ex:
        print(" ", s)


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 300)