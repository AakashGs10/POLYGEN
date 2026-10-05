# Phase 3 — C-VAE conditioning result

Validation run: fixed C-VAE trained on a 60,000-row subset of the Oracle
pseudo-labelled set for 4 epochs (CPU), then targeted generation with
classifier-free guidance. Tg targets are the empirical p10/p50/p90 of the
training set's `Tg_pred`. Oracle Tg is the mean predicted Tg of the generated
polymers, scored by the Phase-1 Tg model.

## Before the fix
Oracle Tg shift across the target sweep was ~0 C — the decoder ignored the
property channel (latent z was under-regularised and carried molecule identity).

## After the fix

| guidance | Tg p10 (-10C) | Tg p50 (46C) | Tg p90 (160C) | shift (p10->p90) |
|----------|---------------|--------------|---------------|------------------|
| 1.0 (plain conditional) | 13.6 | 32.8 | 125.9 | **+112.3 C** |
| 3.0 (amplified)         | -4.0 | 23.1 | 175.6 | **+179.5 C** |

Target range swept: 170 C. Validity ~55-64%, uniqueness ~100%, novelty ~100%.

Training health: val token-accuracy 0.71 -> 0.78 over 4 epochs, no collapse.

## What changed (src/cvae/)
- LATENT 128->64, FREE_BITS 0.08->0.015, BETA_MAX 0.05->0.2: squeeze z so the
  decoder must read the property to minimise reconstruction loss.
- project [props*mask, mask] to a 64-dim property embedding (was 5 raw dims).
- TOKEN_DROPOUT 0.25->0.4, PROP_DROPOUT 0.5->0.3.
- classifier-free guidance at sampling; empirical-quantile Tg sweep; z held
  fixed across conditions for a clean read.

## Caveat
60k-subset / 4-epoch quick model. A full run (669k rows, 12 epochs) is expected
to raise validity and sharpen conditioning further.
