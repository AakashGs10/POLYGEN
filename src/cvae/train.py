import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
import selfies as sf

EMBED = 128
HIDDEN = 512
LATENT = 64              # was 128 — smaller latent so z cannot memorise the whole molecule
PROP_EMBED = 64          # the property condition is projected to real width (was raw 5 dims)
BATCH = 256
EPOCHS = 12
LR = 1e-3
BETA_MAX = 0.2           # was 0.05 — real KL pressure so z stays information-poor
BETA_WARMUP_EPOCHS = 4
PROP_DROPOUT = 0.3       # was 0.5 — more conditional exposure, still trains the uncond path for guidance
FREE_BITS = 0.015        # was 0.08 — 0.015*64≈1 nat free, down from ~10 nats
TOKEN_DROPOUT = 0.4      # was 0.25 — decoder cannot lean on teacher-forced tokens alone
CKPT = config.ROOT / "models" / "cvae"
CKPT.mkdir(parents=True, exist_ok=True)


class PolymerData(Dataset):
    def __init__(self, df, vocab, stats):
        self.stoi = vocab["stoi"]
        self.max_len = vocab["max_len"]
        self.pad = self.stoi[config.PAD_TOKEN]
        self.bos = self.stoi[config.BOS_TOKEN]
        self.eos = self.stoi[config.EOS_TOKEN]
        self.selfies = df["selfies"].tolist()
        props = df[[f"{p}_pred" for p in config.PROPERTIES]].values.astype(np.float32)
        self.props = (props - stats["mean"]) / stats["std"]

    def __len__(self):
        return len(self.selfies)

    def __getitem__(self, i):
        toks = list(sf.split_selfies(self.selfies[i]))
        ids = [self.bos] + [self.stoi[t] for t in toks] + [self.eos]
        ids = ids[: self.max_len]
        ids += [self.pad] * (self.max_len - len(ids))
        return torch.tensor(ids), torch.tensor(self.props[i])


class CVAE(nn.Module):
    def __init__(self, vocab_size, n_props, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.n_props = n_props
        self.embed = nn.Embedding(vocab_size, EMBED, padding_idx=pad_id)
        self.enc = nn.GRU(EMBED, HIDDEN, batch_first=True, bidirectional=True)
        self.to_mu = nn.Linear(HIDDEN * 2, LATENT)
        self.to_lv = nn.Linear(HIDDEN * 2, LATENT)
        # project [props*mask, mask] -> PROP_EMBED so the condition has decoder width
        self.prop_proj = nn.Sequential(
            nn.Linear(n_props * 2, PROP_EMBED), nn.ReLU(),
            nn.Linear(PROP_EMBED, PROP_EMBED),
        )
        cond_dim = LATENT + PROP_EMBED
        self.cond_dim = cond_dim
        self.init_h = nn.Linear(cond_dim, HIDDEN)
        self.dec = nn.GRU(EMBED + cond_dim, HIDDEN, batch_first=True)
        self.out = nn.Linear(HIDDEN, vocab_size)

    def encode(self, x):
        h = self.enc(self.embed(x))[1]
        h = torch.cat([h[0], h[1]], dim=-1)
        return self.to_mu(h), self.to_lv(h)

    def build_cond(self, z, props, mask):
        pcond = self.prop_proj(torch.cat([props * mask, mask], dim=-1))
        return torch.cat([z, pcond], dim=-1)

    def decode(self, x_in, cond):
        h0 = torch.tanh(self.init_h(cond)).unsqueeze(0).contiguous()
        e = self.embed(x_in)
        c = cond.unsqueeze(1).expand(-1, e.size(1), -1)
        y, _ = self.dec(torch.cat([e, c], dim=-1), h0)
        return self.out(y)

    def forward(self, x, props, mask, token_dropout=0.0):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        cond = self.build_cond(z, props, mask)
        x_in = x[:, :-1]
        if token_dropout > 0:
            drop = torch.rand(x_in.shape, device=x_in.device) < token_dropout
            drop[:, 0] = False
            x_in = x_in.masked_fill(drop, self.pad_id)
        return self.decode(x_in, cond), mu, lv


def loss_fn(logits, target, mu, lv, beta, pad_id):
    rec = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target.reshape(-1),
        ignore_index=pad_id, reduction="mean")
    kl_dim = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean(dim=0)
    kl_free = torch.clamp(kl_dim, min=FREE_BITS).sum()
    return rec + beta * kl_free, rec.item(), kl_dim.sum().item()


def token_accuracy(logits, target, pad_id):
    pred = logits.argmax(-1)
    keep = target != pad_id
    return (pred[keep] == target[keep]).float().mean().item()


def run(limit=None, epochs=EPOCHS):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))

    vocab = json.loads(config.VOCAB_FILE.read_text())
    df = pd.read_csv(config.DATA_PROCESSED / "cvae_train_slim.csv")
    if limit:
        df = df.head(limit)
    print(f"training rows: {len(df):,}  vocab: {vocab['vocab_size']}  max_len: {vocab['max_len']}")

    cols = [f"{p}_pred" for p in config.PROPERTIES]
    arr = df[cols].values.astype(np.float32)
    stats = {"mean": arr.mean(0), "std": arr.std(0) + 1e-8}
    np.savez(CKPT / "prop_stats.npz", **stats)

    n_val = max(1, int(len(df) * 0.02))
    val_df, train_df = df.iloc[:n_val], df.iloc[n_val:]
    pad = vocab["stoi"][config.PAD_TOKEN]

    tl = DataLoader(PolymerData(train_df, vocab, stats), batch_size=BATCH,
                    shuffle=True, drop_last=True)
    vl = DataLoader(PolymerData(val_df, vocab, stats), batch_size=BATCH)

    model = CVAE(vocab["vocab_size"], len(config.PROPERTIES), pad).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    print(f"parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M\n")

    steps = len(tl)
    for ep in range(1, epochs + 1):
        model.train()
        beta = BETA_MAX * min(1.0, ep / BETA_WARMUP_EPOCHS)
        agg = np.zeros(4)
        t0 = time.time()
        for i, (x, p) in enumerate(tl, 1):
            x, p = x.to(dev), p.to(dev)
            mask = (torch.rand(p.shape, device=dev) > PROP_DROPOUT).float()
            mask[:, 0] = torch.maximum(mask[:, 0], (mask.sum(1) == 0).float())
            logits, mu, lv = model(x, p, mask, TOKEN_DROPOUT)
            loss, rec, kl = loss_fn(logits, x[:, 1:], mu, lv, beta, pad)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            agg += [loss.item(), rec, kl, token_accuracy(logits, x[:, 1:], pad)]
            if i % 50 == 0 or i == steps:
                a = agg / i
                sys.stdout.write(
                    f"\r  ep{ep:02d} {i}/{steps}  loss {a[0]:.3f}  rec {a[1]:.3f}  "
                    f"kl {a[2]:.2f}  acc {a[3]:.3f}  beta {beta:.3f}  "
                    f"{time.time()-t0:.0f}s   ")
                sys.stdout.flush()

        model.eval()
        vacc, vrec, nb = 0.0, 0.0, 0
        with torch.no_grad():
            for x, p in vl:
                x, p = x.to(dev), p.to(dev)
                mask = torch.ones_like(p)
                logits, mu, lv = model(x, p, mask)
                _, rec, _ = loss_fn(logits, x[:, 1:], mu, lv, beta, pad)
                vacc += token_accuracy(logits, x[:, 1:], pad)
                vrec += rec
                nb += 1
        print(f"\n  -> val token-acc {vacc/nb:.4f}  val rec {vrec/nb:.4f}")
        torch.save({"model": model.state_dict(), "vocab": vocab, "epoch": ep,
                    "prop_mean": stats["mean"], "prop_std": stats["std"]},
                   CKPT / "cvae.pt")

    print(f"\nsaved {CKPT/'cvae.pt'}")


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    eps = int(sys.argv[2]) if len(sys.argv) > 2 else EPOCHS
    run(lim, eps)
