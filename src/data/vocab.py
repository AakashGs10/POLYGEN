import json
import sys
from pathlib import Path

import pandas as pd
import selfies as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


def build():
    df = pd.read_csv(config.SELFIES_ALL)
    alphabet = sf.get_alphabet_from_selfies(df["selfies"].tolist())
    tokens = [config.PAD_TOKEN, config.BOS_TOKEN, config.EOS_TOKEN] + sorted(alphabet)
    stoi = {t: i for i, t in enumerate(tokens)}
    max_len = int(df["n_tokens"].max()) + 2

    payload = {"stoi": stoi, "itos": tokens, "max_len": max_len, "vocab_size": len(tokens)}
    config.VOCAB_FILE.write_text(json.dumps(payload, indent=2))
    print(f"vocab_size {len(tokens)} | max_len {max_len} -> {config.VOCAB_FILE}")
    return payload


def load():
    return json.loads(config.VOCAB_FILE.read_text())


def encode(selfies_str, vocab):
    stoi, max_len = vocab["stoi"], vocab["max_len"]
    tokens = [config.BOS_TOKEN] + list(sf.split_selfies(selfies_str)) + [config.EOS_TOKEN]
    ids = [stoi[t] for t in tokens]
    return ids + [stoi[config.PAD_TOKEN]] * (max_len - len(ids))


def decode(ids, vocab):
    itos = vocab["itos"]
    out = []
    for i in ids:
        tok = itos[i]
        if tok == config.EOS_TOKEN:
            break
        if tok in (config.BOS_TOKEN, config.PAD_TOKEN):
            continue
        out.append(tok)
    return "".join(out)


if __name__ == "__main__":
    build()
