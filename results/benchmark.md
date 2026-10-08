# Closed-loop benchmark

Model: C-VAE trained on 669,642 rows, 12 epochs (RTX 4060). N=300 per arm, guidance=3.0.
Yield = fraction of *valid* generations satisfying every constraint.
`uncond` masks the property channel off (no steering); `cond` is the
guided conditional generation. Lift = cond − uncond.

| query | props | validity | novelty | uncond yield | cond yield | lift |
|-------|-------|----------|---------|--------------|------------|------|
| high Tg | Tg | 85% | 88% | 9.6% | **68.0%** | +58.4% |
| low density | Density | 84% | 93% | 5.5% | **91.7%** | +86.2% |
| high free fractional volume | FFV | 80% | 81% | 4.2% | **80.3%** | +76.2% |
| large radius of gyration | Rg | 92% | 88% | 10.2% | **39.1%** | +28.9% |
| high Tg and low density | Tg+Density | 60% | 93% | 0.0% | **1.1%** | +1.1% |
| heat resistant but lightweight | Tg+Density | 64% | 90% | 0.0% | **3.7%** | +3.7% |
| glass transition above 200 | Tg | 77% | 98% | 3.6% | **43.0%** | +39.4% |

## Per-property detail (satisfaction | Oracle mean)

- high Tg — Tg: cond 68% (mean 174.71) vs uncond 10% (mean 66.48)
- low density — Density: cond 92% (mean 0.89) vs uncond 5% (mean 1.05)
- high FFV — FFV: cond 80% (mean 0.41) vs uncond 4% (mean 0.36)
- large radius of gyration — Rg: cond 39% (mean 21.96) vs uncond 10% (mean 18.40)
- high Tg and low density — Tg: cond 3% (mean 96.67) / Density: cond 50% (mean 0.93)
- heat resistant but lightweight — Tg: cond 9% (mean 105.60) / Density: cond 49% (mean 0.94)
- glass transition above 200 — Tg: cond 43% (mean 191.85) vs uncond 4% (mean 69.84)

## Targeted-generation sweep (generate.py, guidance 3.0)

| Tg target | Oracle Tg of generated | validity | novelty |
|-----------|------------------------|----------|---------|
| p10 (−10 C) | −5.5 | 84.7% | 87.4% |
| p50 ( 46 C) | 41.3 | 89.0% | 90.9% |
| p90 (161 C) | 173.0 | 86.0% | 90.6% |

Conditioning effect: Oracle Tg shifts **+178.5 C** across the 171 C target sweep
— near 1:1 target tracking. Val token-accuracy 0.85.

## Rejection sampling (sample_until.py)

"high Tg and low density" — the hard multi-property corner (1.1% single-shot):
sampled 1024, valid 660 (64%), **20 accepted, 20/20 novel**, all with
Tg_pred ≥ 160 C and Density_pred ≤ 0.927. Low single-shot yield becomes a
usable candidate set in ~1000 samples.

## Reading of results

- **Single-property conditioning is strong and well-controlled** (68–92% yield,
  +58 to +86 points over the unconditional baseline; Tg tracks target ~1:1).
- **Full data fixed Rg** — conditioning rose from ~6% (quick models) to 39%,
  showing Rg was data-starved rather than structurally unconditionable.
- **Simultaneous multi-property targets stay rare single-shot** (1–4%), as
  expected for a top-decile × bottom-decile corner; rejection sampling turns
  this into usable candidates.
- Satisfaction is **Oracle-judged**, so results are bounded by Oracle accuracy
  (Tg is the weakest predictor, test R²=0.37).
