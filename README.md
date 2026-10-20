# PolyGen-RAG

Natural-language-conditioned polymer generation. You describe a polymer in plain
English — *"high Tg, low density"* — and the system parses that into a property
constraint block, generates candidate repeat units conditioned on those targets,
scores them with a property Oracle, and closes the loop by rejection sampling
until candidates meet the request.

**Status: complete.** All tiers built, the full-data C-VAE trained, results
recorded, and a demo UI with safe handling of off-topic input.

## Architecture (runtime order)

```
text query ──▶ Tier 1: RAG parser ──▶ {Tg: ≥160, Density: ≤0.93, ...}
                                           │
                                           ▼
                        Tier 2: Conditional VAE  ──▶ candidate SELFIES
                                           │
                                           ▼
                        Tier 3: Oracle (property predictor)
                                           │
                              accept / reject  (rejection sampling)
```

## Components

The repo is built *backwards from the data*: each stage produces the artifact the
next one consumes, so there is always something to test against.

| Stage | What it is | Code | Status |
|-------|------------|------|--------|
| 0. Data foundation | PI1M ingest, SMILES→SELFIES (two-endpoint enforced), vocab, splits | `src/data/` | ✅ |
| 1. Oracle (Tier 3) | Open Polymer Challenge labels → Morgan + RDKit descriptors → gradient-boosted regressors, **scaffold-split** | `src/oracle/` | ✅ trained |
| 1b. Pseudo-labelling | Oracle labels PI1M → `cvae_train_slim.csv` (the C-VAE's structure+property set) | `analyze_pseudolabels.py` | ✅ |
| 2. Conditional VAE (Tier 2) | bi-GRU encoder / GRU decoder, conditioned on a masked property vector; trained on the pseudo-labelled set | `src/cvae/` | ✅ full 669k / 12-epoch model |
| 1'. RAG parser (Tier 1) | char-ngram TF-IDF retrieval + keyword/comparator rules → strict constraint block, grounded in the data's quantiles | `src/rag/` | ✅ |
| —. Closed loop | text → parse → C-VAE → Oracle → constraint-satisfaction check | `src/pipeline.py` | ✅ |
| —. Rejection sampling | sample until N candidates meet every constraint | `src/sample_until.py` | ✅ |
| —. Benchmark | conditional-vs-unconditional yield over a query suite | `src/benchmark.py` | ✅ |
| —. Demo UI | text box → candidate table, with an anti-hallucination gate | `app.py` | ✅ |

## Final results (full model: 669,642 rows, 12 epochs, val token-acc 0.85)

**Targeted generation** tracks the Tg request almost 1:1:

| Tg target | Oracle Tg of generated | validity | novelty |
|-----------|------------------------|----------|---------|
| p10 (−10 °C) | −5.5 | 85% | 87% |
| p50 (46 °C) | 41.3 | 89% | 91% |
| p90 (161 °C) | 173.0 | 86% | 91% |

Conditioning effect: **+178.5 °C** across the sweep.

**Closed-loop benchmark** — fraction of valid generations satisfying every
constraint, conditional (guided) vs unconditional:

| query | validity | uncond | **cond** | lift |
|-------|----------|--------|----------|------|
| high Tg | 85% | 9.6% | **68.0%** | +58.4% |
| low density | 84% | 5.5% | **91.7%** | +86.2% |
| high free fractional volume | 80% | 4.2% | **80.3%** | +76.2% |
| large radius of gyration | 92% | 10.2% | **39.1%** | +28.9% |
| glass transition above 200 | 77% | 3.6% | **43.0%** | +39.4% |
| high Tg **and** low density | 60% | 0.0% | **1.1%** → rejection sampling: 20/20 novel | — |

Full tables and interpretation in [`results/benchmark.md`](results/benchmark.md);
the conditioning fix write-up in [`results/phase3_conditioning.md`](results/phase3_conditioning.md).

## Key design decisions

- **SELFIES can't encode polymer wildcards.** `*CC*` → `EncoderError`. The two
  connection points are swapped to `[At]` before encoding and back on decode
  (`src/data/chem.py`), and **exactly two** endpoints are required across the
  round-trip. "Valid molecule" ≠ "valid repeat unit", so the two-endpoint check
  is also applied as a generation filter.
- **PI1M has no labels.** A conditional VAE needs (structure, property) pairs, so
  the Oracle is trained first on the Open Polymer Challenge set, then used to
  pseudo-label PI1M for C-VAE training.
- **Scaffold splits for the Oracle** (`src/oracle/features.py`) so near-identical
  backbones don't leak across train/test and inflate metrics.
- **Conditioning fix.** The first C-VAE ignored the property targets (shift ≈ 0).
  Cause: an under-regularised latent `z` memorised each molecule. Fix: shrink
  `z` (128→64), real KL pressure (β 0.05→0.2, free-bits 0.08→0.015), project the
  property to full decoder width, more token dropout, and **classifier-free
  guidance** at sampling. Result: near 1:1 target tracking (see above).
- **Anti-hallucination gate.** The parser decides whether generation runs at all.
  If no real property (Tg/FFV/Tc/Density/Rg) is recognised, the model is never
  invoked — unrelated input (*"write me a poem"*) produces zero constraints and a
  helpful message, not a made-up polymer. The UI also flags out-of-range numeric
  targets as extrapolation and labels every value an Oracle prediction.

## Running

```bash
pip install -r requirements.txt          # + torch for Tiers 2-3, streamlit for the UI

# Stage 0 — data foundation
python src/data/download.py              # fetch PI1M (~48 MB)
python src/data/preprocess.py            # full 1M; pass a number to limit, e.g. 30000
python src/data/vocab.py

# Stage 1 — Oracle (needs Open Polymer Challenge train.csv in data/raw/)
python src/data/properties.py
python src/oracle/train_baseline.py

# Tier 1 — natural language -> constraint block (no torch needed)
python src/rag/parser.py "high Tg, low density"

# Tier 2 — train the conditional VAE (needs cvae_train_slim.csv + a GPU is ~20 min)
python src/cvae/train.py                 # full data, 12 epochs
python src/cvae/generate.py 300 3.0      # targeted sweep + conditioning effect

# Closed loop / rejection sampling / benchmark
python src/pipeline.py "heat resistant but lightweight" 200
python src/sample_until.py "high Tg and low density" 20 3.0
python src/benchmark.py 300

# Demo UI (pip install streamlit)
streamlit run app.py
```

`data/` and `models/` are gitignored — they are regenerated by the commands above.

## Limitations (state these up front)

- Satisfaction is **Oracle-judged**, not lab-measured, so results are bounded by
  Oracle accuracy. Tg is the weakest predictor (test R² ≈ 0.37).
- Simultaneous multi-property targets have low single-shot yield (a rare corner
  of chemical space); rejection sampling is the intended remedy.

## Layout

```
config.py                   paths, hyperparameters, property list
app.py                      Streamlit demo UI (gated on the parser)
src/data/                   download, preprocess (SMILES→SELFIES), vocab, chem, properties
src/oracle/                 features (Morgan + descriptors, scaffold split), train_baseline
src/cvae/                   train (conditional VAE), generate (guided sampling)
src/rag/                    knowledge (lexicon + exemplars), parser, property_ranges.json
src/pipeline.py             text → generate → Oracle check (one query)
src/sample_until.py         rejection sampling to a target count
src/benchmark.py            conditional-vs-unconditional yield suite
analyze_pseudolabels.py     in-domain analysis of Oracle pseudo-labels
results/                    benchmark.md, phase3_conditioning.md, accepted_latest.csv
```
