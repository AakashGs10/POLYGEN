import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

RDLogger.DisableLog("rdApp.*")


def process_one(smiles):
    mol = Chem.MolFromSmiles(smiles.replace("*", "[At]"))
    if mol is None:
        return None
    heavy = mol.GetNumHeavyAtoms()
    if sum(a.GetSymbol() == "At" for a in mol.GetAtoms()) != 2:
        return None
    canon = Chem.MolToSmiles(mol, canonical=True)
    try:
        selfies_str = sf.encoder(canon)
    except Exception:
        return None
    length = sf.len_selfies(selfies_str)
    if length < config.MIN_TOKENS or length > config.MAX_TOKENS:
        return None
    row = {
        "smiles": canon.replace("[At]", "*"),
        "selfies": selfies_str,
        "n_tokens": length,
    }
    if config.COMPUTE_DESCRIPTORS:
        aromatic = sum(a.GetIsAromatic() for a in mol.GetAtoms())
        row.update({
            "mol_wt": Descriptors.MolWt(mol),
            "logp": Crippen.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "n_heavy": heavy,
            "n_rot_bonds": Descriptors.NumRotatableBonds(mol),
            "n_hba": Descriptors.NumHAcceptors(mol),
            "n_hbd": Descriptors.NumHDonors(mol),
            "frac_aromatic": aromatic / heavy if heavy else 0.0,
            "ring_count": Descriptors.RingCount(mol),
        })
    return row


def validate_sample(rows, n=2000):
    checked = failed = 0
    for row in rows[:n]:
        decoded = sf.decoder(row["selfies"])
        mol = Chem.MolFromSmiles(decoded)
        checked += 1
        if mol is None or sum(a.GetSymbol() == "At" for a in mol.GetAtoms()) != 2:
            failed += 1
    print(f"decode validation: {checked - failed}/{checked} valid two-endpoint polymers")


def run(limit=None):
    print("reading PI1M", flush=True)
    df = pd.read_csv(config.PI1M_RAW)
    raw = df[df.columns[0]].dropna().astype(str).str.strip()
    raw = raw[raw.str.count(r"\*") == 2].drop_duplicates()
    if limit:
        raw = raw.head(limit)
    items = raw.tolist()
    total = len(items)
    print(f"input molecules: {total:,}  |  workers: {config.N_WORKERS}", flush=True)

    rows = []
    start = time.time()
    step = max(1, total // 100)

    def report(done):
        elapsed = time.time() - start
        rate = done / elapsed if elapsed else 0
        eta = (total - done) / rate if rate else 0
        sys.stdout.write(
            f"\r  {done:,}/{total:,} ({done/total:5.1%})  "
            f"{rate:,.0f} mol/s  elapsed {elapsed/60:.1f}m  eta {eta/60:.1f}m   "
        )
        sys.stdout.flush()

    if config.N_WORKERS > 1:
        with Pool(config.N_WORKERS) as pool:
            for i, result in enumerate(pool.imap_unordered(process_one, items, chunksize=200), 1):
                if result is not None:
                    rows.append(result)
                if i % step == 0 or i == total:
                    report(i)
    else:
        for i, s in enumerate(items, 1):
            result = process_one(s)
            if result is not None:
                rows.append(result)
            if i % step == 0 or i == total:
                report(i)
    print()

    print(f"survived conversion: {len(rows):,} ({len(rows)/total:.1%})")
    validate_sample(rows)

    out = pd.DataFrame(rows).drop_duplicates(subset="selfies").reset_index(drop=True)
    print(f"unique after canonicalization: {len(out):,}")
    out.to_csv(config.SELFIES_ALL, index=False)

    rng = np.random.default_rng(config.SEED)
    idx = rng.permutation(len(out))
    n_test = int(len(out) * config.TEST_FRAC)
    n_val = int(len(out) * config.VAL_FRAC)
    test_idx, val_idx, train_idx = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]

    out.iloc[train_idx].to_csv(config.SPLIT_TRAIN, index=False)
    out.iloc[val_idx].to_csv(config.SPLIT_VAL, index=False)
    out.iloc[test_idx].to_csv(config.SPLIT_TEST, index=False)
    print(f"train {len(train_idx):,} | val {len(val_idx):,} | test {len(test_idx):,}")
    print(f"total time: {(time.time()-start)/60:.1f} min")
    return out


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(limit)
