# Closed-loop benchmark

Model: 60k-subset / 4-epoch C-VAE. N=300 per arm, guidance=3.0.
Yield = fraction of *valid* generations satisfying every constraint.
`uncond` masks the property channel off (no steering); `cond` is the
guided conditional generation. Lift = cond − uncond.

| query | props | validity | novelty | uncond yield | cond yield | lift |
|-------|-------|----------|---------|--------------|------------|------|
| high Tg | Tg | 61% | 100% | 5.1% | **67.2%** | +62.2% |
| low density | Density | 64% | 99% | 17.9% | **95.9%** | +78.0% |
| high free fractional volume | FFV | 52% | 99% | 4.3% | **93.6%** | +89.3% |
| large radius of gyration | Rg | 64% | 100% | 0.6% | **5.2%** | +4.7% |
| high Tg and low density | Tg+Density | 62% | 99% | 0.0% | **3.8%** | +3.8% |
| heat resistant but lightweight | Tg+Density | 62% | 99% | 0.0% | **4.3%** | +4.3% |
| glass transition above 200 | Tg | 53% | 99% | 2.1% | **32.9%** | +30.8% |

## Per-property detail (satisfaction | Oracle mean)

**high Tg**
- Tg: cond 67% (mean 175.55) vs uncond 5% (mean 51.96)

**low density**
- Density: cond 96% (mean 0.87) vs uncond 18% (mean 0.98)

**high free fractional volume**
- FFV: cond 94% (mean 0.41) vs uncond 4% (mean 0.37)

**large radius of gyration**
- Rg: cond 5% (mean 18.85) vs uncond 1% (mean 16.18)

**high Tg and low density**
- Tg: cond 8% (mean 78.54) vs uncond 2% (mean 42.99)
- Density: cond 63% (mean 0.91) vs uncond 17% (mean 0.99)

**heat resistant but lightweight**
- Tg: cond 6% (mean 80.22) vs uncond 3% (mean 46.40)
- Density: cond 63% (mean 0.91) vs uncond 18% (mean 0.99)

**glass transition above 200**
- Tg: cond 33% (mean 181.06) vs uncond 2% (mean 44.57)
