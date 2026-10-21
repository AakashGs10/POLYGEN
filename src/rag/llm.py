"""LLM step of the RAG: retrieved papers -> property constraints + cited rationale.

Supports two providers, auto-selected by which API key is present:
  - Google Gemini  (free tier)  -> set GEMINI_API_KEY   (SDK: google-genai)
  - Anthropic Claude            -> set ANTHROPIC_API_KEY (SDK: anthropic)
Force one with POLYGEN_LLM_PROVIDER=google|anthropic; pick the model with
POLYGEN_LLM_MODEL. The model reads the retrieved abstracts and returns a strict
JSON constraint block over the five properties plus a cited explanation.

Keys are read from the environment / a local .env (never committed).
"""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.rag import retriever

_UNITS = {"Tg": "C", "FFV": "unitless", "Tc": "unitless",
          "Density": "g/cm3", "Rg": "Angstrom"}
_DEFAULT_MODEL = {"google": "gemini-2.5-flash", "anthropic": "claude-opus-5-5"}


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except Exception:
        pass


def _resolve_provider():
    p = os.environ.get("POLYGEN_LLM_PROVIDER", "").lower()
    if p in ("google", "gemini"):
        return "google"
    if p in ("anthropic", "claude"):
        return "anthropic"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "google"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    raise RuntimeError("no LLM API key found — set GEMINI_API_KEY (free) or "
                       "ANTHROPIC_API_KEY in your .env")


def _model(provider):
    return os.environ.get("POLYGEN_LLM_MODEL") or _DEFAULT_MODEL[provider]


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


def _call_google(system, user, model, max_tokens):
    from google import genai
    from google.genai import types
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )
    return resp.text


def _call_anthropic(system, user, model, max_tokens):
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model, max_tokens=max_tokens,
        output_config={"effort": "low"},
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if msg.stop_reason == "refusal":
        raise RuntimeError("model refused the request")
    return "".join(b.text for b in msg.content if b.type == "text")


def extract(query, k=5, ranges=None, max_tokens=1200):
    """Retrieve papers, ask the LLM for constraints + rationale. Returns a dict."""
    if ranges is None:
        from src.rag.parser import load_ranges
        ranges = load_ranges()
    _load_env()
    provider = _resolve_provider()
    model = _model(provider)
    papers = retriever.retrieve(query, k)
    system, user = _system_prompt(ranges), _user_prompt(query, papers)

    if provider == "google":
        text = _call_google(system, user, model, max_tokens)
    else:
        text = _call_anthropic(system, user, model, max_tokens)

    data = _parse_json(text)
    return {
        "constraints": data.get("constraints", []),
        "explanation": data.get("explanation", ""),
        "papers": papers,
        "model": f"{provider}:{model}",
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "a heat-resistant but lightweight polymer"
    out = extract(q)
    print("model:", out["model"])
    print(json.dumps(out["constraints"], indent=2))
    print("\n" + out["explanation"])
    for i, p in enumerate(out["papers"]):
        print(f"  [{i}] {p['title']} ({p['url']})")
