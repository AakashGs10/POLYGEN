"""RAG orchestrator: query -> retrieve papers -> Claude -> constraint block.

Returns the same `targets`/`mask`/`constraints` shape the local parser produces,
so the C-VAE pipeline is unchanged, plus a literature-grounded `explanation` and
the `papers` cited. If the LLM step fails (no API key, network, etc.) it falls
back to the local TF-IDF rule parser so the app keeps working.

    from src.rag import rag
    out = rag.parse("heat resistant but lightweight")
    out["constraints"], out["targets"], out["mask"], out["explanation"], out["papers"]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.rag.parser import load_ranges, parse as local_parse
from src.rag import llm


def _assemble(constraints_list, ranges):
    constraints = {}
    for c in constraints_list:
        p = c.get("property")
        if p not in config.PROPERTIES or "value" not in c:
            continue
        try:
            value = round(float(c["value"]), 4)
        except (TypeError, ValueError):
            continue
        op = c.get("op", "~")
        if op not in (">", ">=", "<", "<=", "~"):
            op = "~"
        constraints[p] = {"op": op, "value": value, "qualifier": "rag",
                          "unit": llm._UNITS[p], "citations": c.get("citations", []),
                          "reason": c.get("reason", "")}
    targets, mask, active = [], [], []
    for p in config.PROPERTIES:
        if p in constraints:
            targets.append(constraints[p]["value"])
            mask.append(1.0)
            active.append(p)
        else:
            targets.append(ranges[p]["p50"])
            mask.append(0.0)
    return constraints, targets, mask, active


def parse(query, k=5):
    ranges = load_ranges()
    try:
        res = llm.extract(query, k=k, ranges=ranges)
        constraints, targets, mask, active = _assemble(res["constraints"], ranges)
        return {"query": query, "constraints": constraints, "targets": targets,
                "mask": mask, "active": active, "unparsed": [],
                "explanation": res["explanation"], "papers": res["papers"],
                "source": "rag", "model": res["model"]}
    except Exception as exc:
        out = local_parse(query)
        out.update({"explanation": "", "papers": [], "source": "local-fallback",
                    "error": f"{type(exc).__name__}: {exc}"})
        return out


if __name__ == "__main__":
    import json
    q = " ".join(sys.argv[1:]) or "high Tg, low density"
    out = parse(q)
    print(f"source: {out['source']}" + (f"  ({out.get('error')})" if out.get("error") else ""))
    print(json.dumps(out["constraints"], indent=2))
    if out.get("explanation"):
        print("\n" + out["explanation"])
    for i, p in enumerate(out.get("papers", [])):
        print(f"  [{i}] {p['title']} ({p['url']})")
