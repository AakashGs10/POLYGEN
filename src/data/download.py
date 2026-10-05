import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

EXPECTED_BYTES = 47562943
EXPECTED_ROWS = 995799


def _progress(block_num, block_size, total_size):
    done = block_num * block_size
    if total_size > 0:
        pct = min(100.0, done * 100.0 / total_size)
        sys.stdout.write(f"\r  {done / 1e6:6.1f} / {total_size / 1e6:.1f} MB  ({pct:5.1f}%)")
    else:
        sys.stdout.write(f"\r  {done / 1e6:6.1f} MB")
    sys.stdout.flush()


def verify(path):
    if not path.exists():
        return False, "missing"
    size = path.stat().st_size
    if size != EXPECTED_BYTES:
        return False, f"size {size:,} bytes != expected {EXPECTED_BYTES:,}"
    try:
        import pandas as pd
        rows = len(pd.read_csv(path))
    except Exception as exc:
        return False, f"unreadable: {exc}"
    if rows != EXPECTED_ROWS:
        return False, f"rows {rows:,} != expected {EXPECTED_ROWS:,}"
    return True, f"{size / 1e6:.1f} MB, {rows:,} rows"


def download(force=False):
    path = config.PI1M_RAW
    if not force:
        ok, detail = verify(path)
        if ok:
            print(f"verified: {path} ({detail})")
            return path
        if path.exists():
            print(f"INCOMPLETE OR CORRUPT: {detail}")
            print("re-downloading")
            path.unlink()

    print(f"downloading {config.PI1M_URL}")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(config.PI1M_URL, tmp, _progress)
    print()
    if path.exists():
        path.unlink()
    tmp.replace(path)

    ok, detail = verify(path)
    if not ok:
        raise RuntimeError(f"download failed verification: {detail}")
    print(f"verified: {path} ({detail})")
    return path


if __name__ == "__main__":
    download(force="--force" in sys.argv)
