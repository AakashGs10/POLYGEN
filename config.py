import os
from pathlib import Path

ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

PI1M_URL = "https://raw.githubusercontent.com/RUIMINMA1996/PI1M/master/PI1M.csv"
PI1M_RAW = DATA_RAW / "PI1M.csv"

SELFIES_ALL = DATA_PROCESSED / "pi1m_selfies.csv"
SPLIT_TRAIN = DATA_PROCESSED / "train.csv"
SPLIT_VAL = DATA_PROCESSED / "val.csv"
SPLIT_TEST = DATA_PROCESSED / "test.csv"
VOCAB_FILE = DATA_PROCESSED / "vocab.json"

MIN_TOKENS = 4
MAX_TOKENS = 120
VAL_FRAC = 0.05
TEST_FRAC = 0.05
SEED = 42
N_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))
COMPUTE_DESCRIPTORS = True

PAD_TOKEN = "[nop]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"

for d in (DATA_RAW, DATA_PROCESSED):
    d.mkdir(parents=True, exist_ok=True)

OPC_RAW = DATA_RAW / "train.csv"
OPC_PROCESSED = DATA_PROCESSED / "opc_labelled.csv"
ORACLE_DIR = ROOT / "models" / "oracle"
PROPERTIES = ["Tg", "FFV", "Tc", "Density", "Rg"]
MORGAN_BITS = 2048
MORGAN_RADIUS = 2
ORACLE_DIR.mkdir(parents=True, exist_ok=True)
