"""LLM step of the RAG: retrieved papers -> property constraints + cited rationale.

Uses the official Anthropic SDK (Claude). The model reads the user's request plus
the retrieved paper abstracts and returns a strict JSON constraint block over the
five properties, with a short explanation that cites the papers by index.

Auth: the Anthropic client resolves ANTHROPIC_API_KEY from the environment (a
local .env is loaded if python-dotenv is installed). Model is claude-opus-5-5 by
default; override with POLYGEN_LLM_MODEL.
"""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.rag import retriever

MODEL = os.environ.get("POLYGEN_LLM_MODEL", "claude-opus-5-5")

_UNITS = {"Tg": "C", "FFV": "unitless", "Tc": "unitless",
          "Density": "g/cm3", "Rg": "Angstrom"}


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except Exception:
        pass


def _system_prompt(ranges):
    lines = ["You convert a polymer design request into a strict property constraint block.",
             "The only properties you may target are:"]
    for p in config.PROPERTIES:
        r = ranges.get(p, {})
        lines.append(f"  - {p} ({_UNITS[p]}): data p10={r.get('p10')}, p50={r.get('p50')}, "
                     f"p90={r.get('p90')}, range [{r.get('min')}, {r.get('max')}]")
    lines += [
        "",
        "You are given retrieved polymer-science paper abstracts, each with an index [i].",
        "Use them to decide which properties to target and in which direction, and to justify it.",
        "For a qualitative ask, ground numeric targets in the data percentiles: 'high' -> p90, 'low' -> p10, 'moderate' -> p50.",
        "Respect explicit numbers the user gives (e.g. 'Tg above 200' -> op '>' value 200).",
        "Only include a property the user actually asked for. If the request names no real property, return an empty constraints list.",
        "",
        "Return ONLY a JSON object, no prose outside it, with this schema:",
        '{"constraints": [{"property": <one of the five>, "op": one of ">"|">="|"<"|"<="|"~",',
        '  "value": <number in the property\'s native unit>, "citations": [<paper indices>], "reason": <short string>}],',
        ' "explanation": <one short paragraph citing papers by [index]>}',
    ]
    return "\n".join(lines)


def _user_prompt(query, papers):
    blocks = [f"USER REQUEST: {query}", "", "RETRIEVED PAPERS:"]
    for i, p in enumerate(papers):
        blocks.append(f"[{i}] ({p['prop']}) {p['title']} — {p['authors']} {p['year']}, "
                      f"{p['journal']}. {p['abstract']}")
    blocks.append("\nProduce the JSON constraint block now.")
    return "\n".join(blocks)


def _parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError(f"no JSON object in model output: {text[:200]}")
        return json.loads(m.group(0))


def extract(query, k=5, ranges=None, max_tokens=1200):
    """Retrieve papers, ask Claude for constraints + rationale. Returns a dict."""
    import anthropic  # lazy import so non-RAG paths don't require the package

    if ranges is None:
        from src.rag.parser import load_ranges
        ranges = load_ranges()
    papers = retriever.retrieve(query, k)

    _load_env()
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY from env / .env
    msg = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        output_config={"effort": "low"},
        system=_system_prompt(ranges),
        messages=[{"role": "user", "content": _user_prompt(query, papers)}],
    )
    if msg.stop_reason == "refusal":
        raise RuntimeError("model refused the request")
    text = "".join(b.text for b in msg.content if b.type == "text")
    data = _parse_json(text)
    return {
        "constraints": data.get("constraints", []),
        "explanation": data.get("explanation", ""),
        "papers": papers,
        "model": MODEL,
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "a heat-resistant but lightweight polymer"
    out = extract(q)
    print(json.dumps(out["constraints"], indent=2))
    print("\n" + out["explanation"])
    print("\npapers:")
    for i, p in enumerate(out["papers"]):
        print(f"  [{i}] {p['title']} ({p['url']})")
