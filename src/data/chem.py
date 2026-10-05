import selfies as sf
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen

RDLogger.DisableLog("rdApp.*")

WILDCARD = "*"
PLACEHOLDER = "[At]"


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def smiles_to_selfies(smiles):
    swapped = smiles.replace(WILDCARD, PLACEHOLDER)
    mol = Chem.MolFromSmiles(swapped)
    if mol is None:
        return None
    if sum(a.GetSymbol() == "At" for a in mol.GetAtoms()) != 2:
        return None
    canon = Chem.MolToSmiles(mol, canonical=True)
    try:
        return sf.encoder(canon)
    except Exception:
        return None


def selfies_to_smiles(selfies_str):
    try:
        raw = sf.decoder(selfies_str)
    except Exception:
        return None
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True).replace(PLACEHOLDER, WILDCARD)


def n_tokens(selfies_str):
    return sf.len_selfies(selfies_str)


def descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles.replace(WILDCARD, PLACEHOLDER))
    if mol is None:
        return None
    heavy = mol.GetNumHeavyAtoms()
    aromatic = sum(a.GetIsAromatic() for a in mol.GetAtoms())
    return {
        "mol_wt": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "n_heavy": heavy,
        "n_rot_bonds": Descriptors.NumRotatableBonds(mol),
        "n_hba": Descriptors.NumHAcceptors(mol),
        "n_hbd": Descriptors.NumHDonors(mol),
        "frac_aromatic": aromatic / heavy if heavy else 0.0,
        "ring_count": Descriptors.RingCount(mol),
    }
