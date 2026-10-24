"""Does low joint support depress yield once target difficulty is controlled for?

support_vs_yield.py measures raw pair yields, but those confound two causes:

  (1) the pair was barely co-observed in training (the paper's claim), and
  (2) the requested corner is physically hard -- "high Tg + low density" asks
      for a rigid backbone that is also light, which chemistry resists however
      many labels exist.

This script separates them. For each pair it computes

    shortfall = log10( y_ab ) - log10( y_a * y_b )

where y_a and y_b are the SINGLE-property yields measured in the same run at
the same targets. y_a * y_b is what the joint yield would be if the two
constraints were satisfied independently, so the shortfall is the part of the
collapse not explained by either constraint being hard on its own.

It then regresses shortfall on two predictors:

    log10(joint support)  -- the paper's proposed cause
    compat                -- directional correlation of the two properties in
                             the generator's training distribution, signed by
                             the requested direction. compat > 0 means the two
                             requests pull the same way (easy corner),
                             compat < 0 means they oppose (hard corner).

Reading the result:

  * shortfall tracks log-support and not compat  -> disjointness is the cause;
    the paper's thesis survives in a sharper, controlled form.
  * shortfall tracks compat and not log-support  -> the collapse is chemistry,
    not label structure, and Section 6.3 is a negative result.

    python src/support_controlled.py 300 3.0 --seeds 3

Writes results/support_controlled.{csv,md}. With ten pairs this regression is
underpowered; it is reported as descriptive evidence, not a significance test.
"""

import argparse
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.cvae import generate as gen
from src.rag.parser import load_ranges
from src.support_vs_yield import (DIRECTION, build_constraints, joint_support,
                                  one_arm)

OUT_CSV = config.ROOT / "results" / "support_controlled.csv"
OUT_MD = config.ROOT / "results" / "support_controlled.md"


def build_single(prop, ranges):
    """Same corner as build_constraints, but with one property active."""
    targets, mask, constraints = [], [], {}
    for p in config.PROPERTIES:
        if p == prop:
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
            "active": [prop]}


def training_correlations():
    """Pearson correlation between properties in the generator's training set.

    Computed on the oracle pseudo-labels because the observed labels are the
    very thing that is disjoint -- a Tg/FFV correlation cannot be estimated
    from one co-labelled molecule. These are therefore correlations between
    oracle outputs, which is the right quantity here: they describe the
    distribution the generator was actually trained on.
    """
    path = config.DATA_PROCESSED / "cvae_train_slim.csv"
    if not path.exists():
        path = config.DATA_PROCESSED / "pi1m_pseudolabelled.csv"
    df = pd.read_csv(path)
    cols = {p: f"{p}_pred" for p in config.PROPERTIES
            if f"{p}_pred" in df.columns}
    out = {}
    for a, b in itertools.combinations(sorted(cols), 2):
        out[(a, b)] = float(df[cols[a]].corr(df[cols[b]]))
    return out


def _partial(x, y, z):
    """Correlation of x and y with z regressed out of both."""
    if len(x) < 4:
        return float("nan")
    zc = np.column_stack([np.ones_like(z), z])
    rx = x - zc @ np.linalg.lstsq(zc, x, rcond=None)[0]
    ry = y - zc @ np.linalg.lstsq(zc, y, rcond=None)[0]
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def run(n=300, guidance=3.0, seeds=1):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, mean, std = gen.load()
    model = model.to(dev).eval()
    oracles = {p: pickle.load(open(config.ORACLE_DIR / f"{p}.pkl", "rb"))
               for p in config.PROPERTIES
               if (config.ORACLE_DIR / f"{p}.pkl").exists()}
    train_smiles = set(pd.read_csv(config.SELFIES_ALL, usecols=["smiles"])["smiles"])
    ranges = load_ranges()
    supports, _ = joint_support()
    corr = training_correlations()

    # ---- single-property yields, the independence baseline ----
    single = {}
    for p in sorted(oracles):
        ys = []
        for s in range(seeds):
            torch.manual_seed(config.SEED + s)
            _, _, y = one_arm(model, vocab, mean, std, dev, oracles,
                              train_smiles, build_single(p, ranges),
                              n, guidance, True)
            ys.append(y)
        single[p] = float(np.mean(ys))
        print(f"single  {p:8s} yield {single[p]:6.1%}")
    print()

    floor = 0.5 / max(n, 1)        # half a count, so log10 stays finite

    rows = []
    for pair, support in sorted(supports.items(), key=lambda kv: -kv[1]):
        if not set(pair) <= set(oracles):
            continue
        a, b = pair
        ys = []
        for s in range(seeds):
            torch.manual_seed(config.SEED + 1000 + s)
            _, _, y = one_arm(model, vocab, mean, std, dev, oracles,
                              train_smiles, build_constraints(pair, ranges),
                              n, guidance, True)
            ys.append(y)
        obs = float(np.mean(ys))
        exp = single[a] * single[b]
        sign = (1 if DIRECTION[a] == "high" else -1) * \
               (1 if DIRECTION[b] == "high" else -1)
        compat = sign * corr.get(tuple(sorted((a, b))), 0.0)
        shortfall = np.log10(max(obs, floor)) - np.log10(max(exp, floor))

        rows.append({
            "pair": " + ".join(pair), "joint_support": support,
            "single_a": single[a], "single_b": single[b],
            "expected_independent": exp, "observed": obs,
            "obs_sd": float(np.std(ys)),
            "shortfall_log10": shortfall, "compat": compat, "seeds": seeds,
        })
        r = rows[-1]
        print(f"{r['pair']:22s} supp {support:5d}  exp {exp:6.1%}  "
              f"obs {obs:6.1%}  shortfall {shortfall:+.2f} dex  "
              f"compat {compat:+.2f}")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    ls = np.log10(np.maximum(df["joint_support"].to_numpy(float), 1.0))
    sf = df["shortfall_log10"].to_numpy(float)
    cp = df["compat"].to_numpy(float)
    raw = df["observed"].to_numpy(float)

    r_sf_supp = float(np.corrcoef(ls, sf)[0, 1])
    r_sf_comp = float(np.corrcoef(cp, sf)[0, 1])
    pr_supp = _partial(sf, ls, cp)      # support, compat held out
    pr_comp = _partial(sf, cp, ls)      # compat, support held out
    r_raw_supp = float(np.corrcoef(ls, raw)[0, 1])

    if r_sf_supp > 0.5 and abs(r_sf_supp) > abs(r_sf_comp):
        verdict = ("Shortfall tracks joint support more strongly than target "
                   "compatibility: consistent with disjointness depressing "
                   "yield beyond single-constraint difficulty.")
    elif abs(r_sf_comp) > abs(r_sf_supp):
        verdict = ("Shortfall tracks target compatibility more strongly than "
                   "joint support: the collapse is better explained by the "
                   "requested corner being chemically opposed than by the "
                   "pair being unobserved.")
    else:
        verdict = ("Neither predictor dominates at this sample size; the "
                   "controlled test is inconclusive as run.")

    with open(OUT_MD, "w") as fh:
        fh.write("# Joint support vs. yield, controlled for target difficulty\n\n")
        fh.write(f"N={n} per arm, guidance={guidance}, seeds={seeds}. "
                 "`expected_independent` is the product of the two "
                 "single-property yields measured in this same run; "
                 "`shortfall` is log10(observed) - log10(expected), in dex. "
                 "`compat` is the directional correlation of the two "
                 "properties in the generator's training distribution "
                 "(positive = the two requests pull the same way).\n\n")
        fh.write("| pair | support | single a | single b | expected | observed "
                 "| shortfall (dex) | compat |\n")
        fh.write("|---|---|---|---|---|---|---|---|\n")
        for _, r in df.iterrows():
            fh.write(f"| {r['pair']} | {int(r['joint_support'])} | "
                     f"{r['single_a']:.1%} | {r['single_b']:.1%} | "
                     f"{r['expected_independent']:.1%} | "
                     f"**{r['observed']:.1%}** | {r['shortfall_log10']:+.2f} | "
                     f"{r['compat']:+.2f} |\n")
        fh.write(f"\n## Correlations over {len(df)} pairs\n\n")
        fh.write(f"- r(log10 support, raw yield)      = {r_raw_supp:+.3f}\n")
        fh.write(f"- r(log10 support, shortfall)      = {r_sf_supp:+.3f}\n")
        fh.write(f"- r(compat, shortfall)             = {r_sf_comp:+.3f}\n")
        fh.write(f"- partial r(support, shortfall | compat) = {pr_supp:+.3f}\n")
        fh.write(f"- partial r(compat, shortfall | support) = {pr_comp:+.3f}\n")
        fh.write(f"\n**Reading.** {verdict}\n\n")
        fh.write("Ten pairs cannot support a significance claim; these are "
                 "descriptive. Yields are Oracle-judged and inherit its "
                 "error.\n")

    print(f"\nr(log10 support, raw yield)  = {r_raw_supp:+.3f}")
    print(f"r(log10 support, shortfall)  = {r_sf_supp:+.3f}")
    print(f"r(compat, shortfall)         = {r_sf_comp:+.3f}")
    print(f"partial r(support|compat)    = {pr_supp:+.3f}")
    print(f"partial r(compat|support)    = {pr_comp:+.3f}")
    print(f"\n{verdict}\nwrote {OUT_CSV} and {OUT_MD}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=300)
    ap.add_argument("guidance", nargs="?", type=float, default=3.0)
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    run(a.n, a.guidance, a.seeds)
