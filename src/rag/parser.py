"""Tier-1 parser: natural-language request -> strict property constraint block.

    parse("high Tg and low density")
    -> {"constraints": {"Tg": {"op": ">=", "value": 160.49, ...},
                        "Density": {"op": "<=", "value": 0.93, ...}},
        "targets": [...5...], "mask": [1,0,0,1,0], "active": ["Tg","Density"],
        "unparsed": []}

The `targets`/`mask` fields plug straight into src.cvae.generate.make_cond.
Retrieval (char n-gram TF-IDF over an exemplar corpus) resolves paraphrases
that contain no property keyword. No torch, no network.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.rag import knowledge as kb

# prefer the packaged ranges (tracked in git); fall back to the generated copy
_PACKAGED_RANGES = Path(__file__).resolve().parent / "property_ranges.json"
RANGES_FILE = _PACKAGED_RANGES if _PACKAGED_RANGES.exists() else config.DATA_PROCESSED / "property_ranges.json"
_NUM = r"[-+]?\d+(?:\.\d+)?"
_CLAUSE_SPLIT = re.compile(r"\s*(?:,|;|\band\b|\bwith\b|\bbut\b|\bwhile\b|&)\s*", re.I)
_PROP_MIN_SIM = 0.30   # accept a property from retrieval (no keyword present)
_DIR_MIN_SIM = 0.15    # borrow only a direction for an already-known property


def load_ranges():
    if RANGES_FILE.exists():
        return json.loads(RANGES_FILE.read_text())
    # fallback if the grounding file was not generated yet
    return {p: {"p10": 0.0, "p50": 0.5, "p90": 1.0} for p in config.PROPERTIES}


class ConstraintParser:
    def __init__(self):
        self.ranges = load_ranges()
        self._phrases = [e[0] for e in kb.EXEMPLARS]
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self._mat = self._vec.fit_transform(self._phrases)

    # --- clause-level analysis -------------------------------------------
    def _match_property(self, clause):
        """Return (property, matched_alias_len) via keyword, else (None, 0)."""
        best, best_len = None, 0
        for prop, info in kb.PROPERTY_INFO.items():
            for alias in [prop.lower()] + info["aliases"]:
                if re.search(r"\b" + re.escape(alias) + r"\b", clause) and len(alias) > best_len:
                    best, best_len = prop, len(alias)
        return best, best_len

    def _retrieve(self, clause):
        """Nearest exemplar -> (property, direction, similarity)."""
        q = self._vec.transform([clause])
        sims = linear_kernel(q, self._mat)[0]
        i = int(sims.argmax())
        _, prop, direction = kb.EXEMPLARS[i]
        return prop, direction, float(sims[i])

    def _direction(self, clause):
        toks = set(re.findall(r"[a-z]+", clause))
        if toks & set(kb.MID_WORDS):
            return "mid"
        hi, lo = toks & set(kb.HIGH_WORDS), toks & set(kb.LOW_WORDS)
        if hi and not lo:
            return "high"
        if lo and not hi:
            return "low"
        return None

    def _numeric(self, clause):
        """Return (op, value) for an explicit number, else (None, None)."""
        for phrase, op in kb.COMPARATORS:
            m = re.search(re.escape(phrase) + r"\s*(" + _NUM + r")", clause)
            if m:
                return op, float(m.group(1))
        # bare number with no comparator, e.g. "Tg 150"
        m = re.search(_NUM, clause)
        if m:
            return "~", float(m.group())
        return None, None

    def _resolve(self, prop, direction, op, value):
        """Combine detected pieces into (op, value, qualifier)."""
        r = self.ranges[prop]
        if value is not None:
            return op or "~", round(value, 4), "explicit"
        if direction == "high":
            return ">=", r["p90"], "high"
        if direction == "low":
            return "<=", r["p10"], "low"
        if direction == "mid":
            return "~", r["p50"], "mid"
        return None, None, None

    # --- public API -------------------------------------------------------
    def parse(self, query):
        clauses = [c.strip() for c in _CLAUSE_SPLIT.split(query.lower()) if c.strip()]
        constraints, unparsed = {}, []

        for clause in clauses:
            prop, _ = self._match_property(clause)
            direction = self._direction(clause)
            op, value = self._numeric(clause)
            rprop, rdir, sim = self._retrieve(clause)

            # no explicit property keyword -> fall back to retrieval
            if prop is None:
                if sim >= _PROP_MIN_SIM:
                    prop = rprop
                else:
                    unparsed.append(clause)
                    continue

            # property known but no high/low word and no number ->
            # borrow the direction from the nearest exemplar of this property
            if direction is None and value is None:
                if sim >= _DIR_MIN_SIM and rprop == prop:
                    direction = rdir

            rop, rval, qual = self._resolve(prop, direction, op, value)
            if rop is None:
                unparsed.append(clause)
                continue
            constraints[prop] = {
                "op": rop, "value": rval, "qualifier": qual,
                "unit": kb.PROPERTY_INFO[prop]["unit"], "clause": clause,
            }

        # build the C-VAE conditioning vector (inactive props -> dataset median)
        targets, mask, active = [], [], []
        for p in config.PROPERTIES:
            if p in constraints:
                targets.append(constraints[p]["value"])
                mask.append(1.0)
                active.append(p)
            else:
                targets.append(self.ranges[p]["p50"])
                mask.append(0.0)

        return {
            "query": query,
            "constraints": constraints,
            "targets": targets,
            "mask": mask,
            "active": active,
            "unparsed": unparsed,
        }


_PARSER = None


def parse(query):
    """Module-level convenience wrapper (builds the index once)."""
    global _PARSER
    if _PARSER is None:
        _PARSER = ConstraintParser()
    return _PARSER.parse(query)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]
    else:
        queries = [
            "high Tg, low density",
            "heat resistant but lightweight",
            "glass transition above 200 and density under 1.0",
            "a permeable membrane with low thermal conductivity",
            "rubbery flexible polymer with large radius of gyration",
            "Tg around 120 C",
            "something strong and colourful",  # partially unparseable
        ]
    p = ConstraintParser()
    for q in queries:
        out = p.parse(q)
        print(f"\n> {q}")
        for prop, c in out["constraints"].items():
            print(f"    {prop:8s} {c['op']:>2s} {c['value']:<10} [{c['qualifier']}]")
        if out["unparsed"]:
            print(f"    (unparsed: {out['unparsed']})")
        print(f"    mask={out['mask']}  active={out['active']}")
