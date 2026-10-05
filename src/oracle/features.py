import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

RDLogger.DisableLog("rdApp.*")

_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=config.MORGAN_RADIUS, fpSize=config.MORGAN_BITS
)

DESCRIPTOR_NAMES = [
    "mol_wt", "logp", "tpsa", "n_heavy", "n_rot_bonds", "n_hba", "n_hbd",
    "frac_aromatic", "ring_count", "n_arom_rings", "frac_csp3",
    "heavy_per_ring", "balaban_j", "bertz_ct",
]


def _mol(smiles):
    return Chem.MolFromSmiles(smiles.replace("*", "[At]"))


def descriptor_vector(mol):
    heavy = mol.GetNumHeavyAtoms()
    aromatic = sum(a.GetIsAromatic() for a in mol.GetAtoms())
    rings = Descriptors.RingCount(mol)
    try:
        balaban = Descriptors.BalabanJ(mol)
    except Exception:
        balaban = 0.0
    return np.array([
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        Descriptors.TPSA(mol),
        heavy,
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        aromatic / heavy if heavy else 0.0,
        rings,
        Descriptors.NumAromaticRings(mol),
        Descriptors.FractionCSP3(mol),
        heavy / rings if rings else heavy,
        balaban,
        Descriptors.BertzCT(mol),
    ], dtype=np.float64)


def featurize(smiles_list):
    fps, descs, ok = [], [], []
    for smiles in smiles_list:
        mol = _mol(smiles)
        if mol is None:
            ok.append(False)
            continue
        fps.append(np.array(_GEN.GetFingerprintAsNumPy(mol), dtype=np.float32))
        descs.append(descriptor_vector(mol))
        ok.append(True)
    X = np.hstack([np.vstack(fps), np.vstack(descs)])
    return X, np.array(ok)


def scaffold_of(smiles):
    mol = _mol(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def scaffold_split(smiles_list, test_frac=0.2, seed=config.SEED):
    groups = defaultdict(list)
    for i, smiles in enumerate(smiles_list):
        groups[scaffold_of(smiles)].append(i)

    buckets = sorted(groups.values(), key=len, reverse=True)
    n_test = int(len(smiles_list) * test_frac)
    test_idx, train_idx = [], []
    for bucket in buckets:
        if len(test_idx) + len(bucket) <= n_test:
            test_idx.extend(bucket)
        else:
            train_idx.extend(bucket)
    return np.array(sorted(train_idx)), np.array(sorted(test_idx)), len(groups)
