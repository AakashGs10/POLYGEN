import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.cvae.train import CVAE, CKPT, LATENT
from src.data import chem
from src.oracle import features

TEMPERATURE = 0.8
GUIDANCE = 3.0   # classifier-free guidance scale; 0 = plain conditional sampling


def load():
    ck = torch.load(CKPT / "cvae.pt", map_location="cpu", weights_only=False)
    vocab = ck["vocab"]
    model = CVAE(vocab["vocab_size"], len(config.PROPERTIES),
                 vocab["stoi"][config.PAD_TOKEN])
    model.load_state_dict(ck["model"])
    model.eval()
    return model, vocab, ck["prop_mean"], ck["prop_std"]


@torch.no_grad()
def sample(model, vocab, cond, n, dev, temperature=TEMPERATURE,
           uncond=None, guidance=0.0):
    stoi, itos = vocab["stoi"], vocab["itos"]
    bos, eos, pad = (stoi[config.BOS_TOKEN], stoi[config.EOS_TOKEN],
                     stoi[config.PAD_TOKEN])
    use_cfg = guidance and uncond is not None

    h = torch.tanh(model.init_h(cond)).unsqueeze(0).contiguous()
    h_u = torch.tanh(model.init_h(uncond)).unsqueeze(0).contiguous() if use_cfg else None
    tok = torch.full((n, 1), bos, dtype=torch.long, device=dev)
    done = torch.zeros(n, dtype=torch.bool, device=dev)
    seqs = [[] for _ in range(n)]

    for _ in range(vocab["max_len"] - 1):
        e = model.embed(tok)
        y, h = model.dec(torch.cat([e, cond.unsqueeze(1)], dim=-1), h)
        logits = model.out(y[:, -1])
        if use_cfg:
            # same generated prefix, two conditioning vectors: extrapolate away from uncond
            y_u, h_u = model.dec(torch.cat([e, uncond.unsqueeze(1)], dim=-1), h_u)
            logits_u = model.out(y_u[:, -1])
            logits = logits_u + guidance * (logits - logits_u)
        logits = logits / temperature
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


def make_cond(model, z, targets, mask_flags, mean, std, dev):
    """Return (cond, uncond). z is shared across conditions for a clean read."""
    n = z.size(0)
    p = (np.array(targets, dtype=np.float32) - mean) / std
    p = torch.tensor(np.tile(p, (n, 1)), dtype=torch.float32, device=dev)
    m = torch.tensor(np.tile(np.array(mask_flags, dtype=np.float32), (n, 1)),
                     dtype=torch.float32, device=dev)
    zero = torch.zeros_like(m)
    return model.build_cond(z, p, m), model.build_cond(z, zero, zero)


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


def run(n=300, guidance=GUIDANCE):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, mean, std = load()
    model = model.to(dev)
    print(f"device: {dev}  |  sampling {n} per condition  |  guidance {guidance}\n")

    oracles = {}
    for prop in config.PROPERTIES:
        p = config.ORACLE_DIR / f"{prop}.pkl"
        if p.exists():
            oracles[prop] = pickle.load(open(p, "rb"))

    train_smiles = set(pd.read_csv(config.SELFIES_ALL, usecols=["smiles"])["smiles"])

    # sweep Tg over the empirical p10/p50/p90 the model actually trained on,
    # not fixed 0/100/200 C (which extrapolate past the pseudo-label range)
    tg = pd.read_csv(config.DATA_PROCESSED / "cvae_train_slim.csv",
                     usecols=["Tg_pred"])["Tg_pred"]
    q10, q50, q90 = (float(tg.quantile(x)) for x in (0.10, 0.50, 0.90))

    tg_i = config.PROPERTIES.index("Tg")
    base = list(mean)
    # ONE z, reused across conditions, so only Tg changes between rows
    z = torch.randn(n, LATENT, device=dev)

    conditions = []
    for label, t_val in [(f"Tg p10 ({q10:5.0f}C)", q10),
                         (f"Tg p50 ({q50:5.0f}C)", q50),
                         (f"Tg p90 ({q90:5.0f}C)", q90)]:
        t = list(base)
        t[tg_i] = t_val
        flags = [0.0] * len(config.PROPERTIES)
        flags[tg_i] = 1.0
        conditions.append((label, t, flags))

    print(f"{'condition':18s} {'validity':>9s} {'unique':>8s} {'novel':>8s} "
          f"{'Oracle Tg':>10s}")
    results = []
    for label, targets, flags in conditions:
        cond, uncond = make_cond(model, z, targets, flags, mean, std, dev)
        gen = sample(model, vocab, cond, n, dev, uncond=uncond, guidance=guidance)
        st = evaluate(gen, train_smiles, oracles)
        results.append((label, st))
        print(f"{label:18s} {st['validity']:8.1%} {st['uniqueness']:7.1%} "
              f"{st['novelty']:7.1%} {st.get('Tg', float('nan')):10.1f}")

    lo, hi = results[0][1].get("Tg"), results[-1][1].get("Tg")
    if lo is not None and hi is not None:
        print(f"\nconditioning effect: Oracle Tg shifts {hi - lo:+.1f} C "
              f"across the p10->p90 sweep ({q90 - q10:.0f} C of target range)")
        print("(a shift near zero means the conditioning is being ignored)")

    ex = results[-1][1]["smiles"][:5]
    print("\nexamples at Tg p90:")
    for s in ex:
        print(" ", s)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    g = float(sys.argv[2]) if len(sys.argv) > 2 else GUIDANCE
    run(n, g)
