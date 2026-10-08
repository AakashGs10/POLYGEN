"""PolyGen-RAG demo UI.

    streamlit run app.py

A text box takes a plain-English polymer request, the parser turns it into a
property constraint block, and the C-VAE + Oracle generate candidates that meet
it. The parser is a HARD GATE: if no real property (Tg/FFV/Tc/Density/Rg) is
found, nothing is generated — so unrelated input ("write me a poem") can never
hallucinate a polymer.
"""

import pickle
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from src.rag.parser import ConstraintParser
from src.cvae import generate as gen
from src import sample_until

st.set_page_config(page_title="PolyGen-RAG", page_icon="🧪", layout="wide")


@st.cache_resource(show_spinner="Loading model, oracle and vocab…")
def load_bundle():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, mean, std = gen.load()
    model = model.to(dev)
    oracles = {p: pickle.load(open(config.ORACLE_DIR / f"{p}.pkl", "rb"))
               for p in config.PROPERTIES if (config.ORACLE_DIR / f"{p}.pkl").exists()}
    train_smiles = set(pd.read_csv(config.SELFIES_ALL, usecols=["smiles"])["smiles"])
    return (model, vocab, mean, std, oracles, train_smiles, dev), ConstraintParser()


def extrapolation_warnings(parsed, ranges):
    """Flag explicit numeric targets that fall outside the training range."""
    msgs = []
    for p, c in parsed["constraints"].items():
        if c["qualifier"] != "explicit":
            continue
        lo, hi = ranges[p]["min"], ranges[p]["max"]
        if c["value"] < lo or c["value"] > hi:
            msgs.append(f"**{p} = {c['value']}** is outside the model's training "
                        f"range ([{lo}, {hi}]). Results there are extrapolation and "
                        f"the Oracle is unreliable.")
    return msgs


bundle, parser = load_bundle()
ranges = parser.ranges
dev_label = "GPU" if bundle[-1].type == "cuda" else "CPU"

st.title("🧪 PolyGen-RAG")
st.caption(f"Natural-language → polymer repeat units, conditioned on predicted "
           f"properties. Running on **{dev_label}**. Properties: "
           f"{', '.join(config.PROPERTIES)}.")

with st.sidebar:
    st.header("Settings")
    guidance = st.slider("Guidance (conditioning strength)", 0.0, 6.0, 3.0, 0.5)
    k = st.slider("Target number of candidates", 3, 40, 12)
    budget = st.select_slider("Sampling budget", [1024, 2048, 4096, 8192], value=4096)
    st.markdown("---")
    st.markdown("**Example requests**")
    examples = ["high Tg, low density", "heat resistant but lightweight",
                "glass transition above 200", "high free fractional volume membrane",
                "large radius of gyration"]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["query"] = ex

query = st.text_input("Describe the polymer you want:",
                      value=st.session_state.get("query", ""),
                      placeholder="e.g. high Tg and low density")
go = st.button("Generate", type="primary")

if go and query.strip():
    parsed = parser.parse(query)

    # ---- HARD GATE: refuse to generate when nothing real was understood ----
    if not parsed["active"]:
        st.error("I couldn't find any polymer property target in that request.")
        st.info("I can only steer the five properties "
                f"({', '.join(config.PROPERTIES)}). Try wording like "
                "*“high Tg, low density”* or *“glass transition above 200”*. "
                "Nothing was generated — the model only runs when it recognises "
                "a real property, so it can't invent an answer to unrelated input.")
        if parsed["unparsed"]:
            st.caption("Ignored (not a recognised property): "
                       + ", ".join(f"“{u}”" for u in parsed["unparsed"]))
        st.stop()

    # ---- show exactly what was understood (transparency) ----
    st.subheader("Understood as")
    rows = [{"property": p, "constraint": f"{c['op']} {c['value']} {c['unit']}".strip(),
             "from": c["qualifier"], "your words": c["clause"]}
            for p, c in parsed["constraints"].items()]
    st.table(pd.DataFrame(rows))
    if parsed["unparsed"]:
        st.caption("Ignored parts of your request: "
                   + ", ".join(f"“{u}”" for u in parsed["unparsed"]))
    for w in extrapolation_warnings(parsed, ranges):
        st.warning(w)

    # ---- generate with rejection sampling ----
    bar = st.progress(0.0, text="Generating and scoring candidates…")
    res = sample_until.collect(query, k=k, guidance=guidance, max_samples=budget,
                               bundle=bundle, progress=lambda f: bar.progress(f))
    bar.empty()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sampled", res["sampled"])
    c2.metric("Valid", f"{res['valid']} ({res['valid']/max(res['sampled'],1):.0%})")
    c3.metric("Meet all constraints", len(res["accepted"]))
    c4.metric("Yield", f"{res['yield']:.1%}")

    if not res["accepted"]:
        st.warning("No candidate met every constraint within the budget. "
                   "Try raising guidance, raising the sampling budget, or relaxing "
                   "the request (e.g. one property instead of two).")
    else:
        df = pd.DataFrame(res["accepted"])
        df.insert(0, "#", range(1, len(df) + 1))
        st.subheader(f"{len(df)} candidate polymers")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False),
                           "polygen_candidates.csv", "text/csv")

    st.caption("⚠️ Property values are **Oracle predictions**, not lab measurements, "
               "and are bounded by Oracle accuracy (Tg is the weakest, test R²≈0.37). "
               "`*` marks the two polymer connection points.")
elif go:
    st.info("Type a request first.")
