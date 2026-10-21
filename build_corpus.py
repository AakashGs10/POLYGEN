"""Expand the RAG paper corpus from OpenAlex (free, no API key).

Fetches polymer structure-property paper abstracts for each target property and
merges them into src/rag/papers.jsonl (deduped by title). Run on a machine with
internet access to OpenAlex:

    python build_corpus.py --per-property 20

The repo ships a seed corpus already; this grows it (e.g. adds Density coverage).
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: F401  (ensures repo root on path)

CORPUS = Path(__file__).resolve().parent / "src" / "rag" / "papers.jsonl"
MAILTO = "polygen@example.com"  # OpenAlex etiquette; replace with your email

QUERIES = {
    "Tg": "polymer glass transition temperature chemical structure",
    "FFV": "polymer fractional free volume gas separation membrane",
    "Tc": "polymer thermal conductivity molecular structure",
    "Density": "polymer density structure property relationship",
    "Rg": "polymer radius of gyration chain dimension",
}


def _abstract(inv):
    if not inv:
        return ""
    positions = [(pos, word) for word, poss in inv.items() for pos in poss]
    return " ".join(w for _, w in sorted(positions))[:900]


def fetch(query, n):
    url = ("https://api.openalex.org/works?" + urllib.parse.urlencode({
        "search": query, "per_page": n, "mailto": MAILTO,
        "filter": "has_abstract:true,type:article",
    }))
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    out = []
    for w in data.get("results", []):
        ab = _abstract(w.get("abstract_inverted_index"))
        if not ab:
            continue
        authors = [a["author"]["display_name"] for a in w.get("authorships", [])[:1]]
        src = (w.get("primary_location") or {}).get("source") or {}
        out.append({
            "title": (w.get("title") or "").strip(),
            "authors": (authors[0] + " et al." if authors else "Unknown"),
            "year": w.get("publication_year"),
            "journal": src.get("display_name") or "",
            "url": w.get("doi") or w.get("id") or "",
            "abstract": ab,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-property", type=int, default=15)
    args = ap.parse_args()

    existing = []
    if CORPUS.exists():
        existing = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
    seen = {p["title"].lower() for p in existing}

    added = 0
    for prop, q in QUERIES.items():
        try:
            papers = fetch(q, args.per_property)
        except Exception as exc:
            print(f"{prop}: fetch failed ({exc})")
            continue
        for p in papers:
            if p["title"] and p["title"].lower() not in seen:
                seen.add(p["title"].lower())
                existing.append({"prop": prop, **p})
                added += 1
        print(f"{prop}: {len(papers)} fetched")
        time.sleep(1)

    for i, p in enumerate(existing):
        p["id"] = f"p{i:02d}"
    CORPUS.write_text("\n".join(json.dumps(p) for p in existing) + "\n")
    print(f"\ncorpus now {len(existing)} papers (+{added}) -> {CORPUS}")


if __name__ == "__main__":
    main()
