# Closed-loop benchmark

Model: C-VAE trained on 120,000 rows, 5 epochs. N=300 per arm, guidance=3.0.
Yield = fraction of *valid* generations satisfying every constraint.
`uncond` masks the property channel off (no steering); `cond` is the
guided conditional generation. Lift = cond − uncond.

| query | props | validity | novelty | uncond yield | cond yield | lift |
|-------|-------|----------|---------|--------------|------------|------|
| high Tg | Tg | 77% | 100% | 3.4% | **69.8%** | +66.4% |
| low density | Density | 72% | 97% | 11.4% | **90.8%** | +79.4% |
| high free fractional volume | FFV | 59% | 99% | 3.2% | **78.4%** | +75.2% |
| large radius of gyration | Rg | 88% | 100% | 2.5% | **6.0%** | +3.5% |
| high Tg and low density | Tg+Density | 57% | 100% | 0.0% | **2.3%** | +2.3% |
| heat resistant but lightweight | Tg+Density | 60% | 100% | 0.0% | **1.7%** | +1.7% |
| glass transition above 200 | Tg | 65% | 100% | 1.3% | **29.9%** | +28.6% |

## Per-property detail (satisfaction | Oracle mean)

**high Tg**
- Tg: cond 70% (mean 180.17) vs uncond 3% (mean 49.06)

**low density**
- Density: cond 91% (mean 0.88) vs uncond 11% (mean 1.01)

**high free fractional volume**
- FFV: cond 78% (mean 0.41) vs uncond 3% (mean 0.36)

**large radius of gyration**
- Rg: cond 6% (mean 19.20) vs uncond 2% (mean 17.06)

**high Tg and low density**
- Tg: cond 6% (mean 94.95) vs uncond 3% (mean 44.92)
- Density: cond 49% (mean 0.93) vs uncond 10% (mean 1.01)

**heat resistant but lightweight**
- Tg: cond 7% (mean 99.42) vs uncond 4% (mean 48.73)
- Density: cond 53% (mean 0.93) vs uncond 10% (mean 1.00)

**glass transition above 200**
- Tg: cond 30% (mean 182.64) vs uncond 1% (mean 51.21)
