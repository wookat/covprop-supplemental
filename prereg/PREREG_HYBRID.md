# PREREG_HYBRID — CQR × CovProp hybrid (convergence round)

- Date registered: 2026-08-12 (this file is written and synced to xu-1 **before**
  any L2 experiment is run; results JSONs will carry later mtimes).
- Motivation (R3 review, D3): CQR leads Winkler on 5/7 real streams; the paper
  calls a CQR×CovProp hybrid "natural" without numbers. This preregisters the
  hybrid and commits to publishing the outcome either way.

## Designs (frozen before running)

Protocol: identical R2/R3 real-stream protocol (7 streams NP, FR, ETTh1,
weather, ETTh2, ETTm2, electricity; ridge upstream forecaster; linear backbone;
chronological 60/20/20; matured feedback; alpha = 0.1; seeds 0–4; K = 64;
no per-dataset tuning). Code: `experiments/covprop_hybrid.py`, importing the
frozen R2/R3 modules unchanged.

- **Hybrid A (primary), `hybrid_mc_cqr` — input-side propagation into the CQR
  heads.** Per-horizon linear pinball heads trained on realized covariates
  (identical to `run_cqr`). At calibration/test time, K upstream-error samples
  from the split-A error pool perturb the observed covariates; the per-sample
  head outputs are pooled and the interval is the mixture quantile
  lo = q_{alpha/2}({lo_k}), hi = q_{1-alpha/2}({hi_k}) over the K samples.
  Matured online CQR-score correction on calibration split B, scores
  max(lo−Y, Y−hi)/sd (identical normalization to `run_cqr`).
- **Hybrid B (secondary), `hybrid_cqr_covwidth` — CovProp-corrected CQR
  scores.** Standard CQR intervals on observed covariates; the conformal
  correction is normalized by the CovProp-MC propagated half-width instead of
  sd: scores max(lo−Y, Y−hi)/half_t, correction lo − q·half_t, hi + q·half_t.
  half_t is the same label-free MC half-width already computed by CovProp-MC.

Baselines recomputed in the same runs for paired per-window significance:
recal, cqr, covprop_mc, covprop_share (definitions unchanged from R3).

## Hypotheses and decision criteria (frozen)

- **H8a (efficiency):** supported iff the better preregistered hybrid
  (selected per stream **by the same calibration-split backtest as §Policy**,
  never by test data) has mean Winkler ≤ CQR on ≥ 5/7 streams.
- **H8b (conditional coverage):** supported iff that hybrid's worst
  |eta|-tercile coverage is ≥ 0.85 on ≥ 6/7 streams (the CovProp property CQR
  lacks on the propagation-dominated streams).
- Significance: seed-clustered moving-block bootstrap on per-window Winkler
  differences (block 48, B = 4000, two-sided), Holm-corrected within stream —
  identical machinery to the paper (`significance_r2`).
- **Publication commitment:** the outcome enters the paper §Real-stream text
  and the response letter regardless of direction. If both hybrids lose to
  plain CQR on ≥ 4/7 streams, that is reported as a negative result closing
  the "natural hybrid" promise; no post-hoc variants will be added to this
  round.

## Outputs
`results/hybrid/l2_<stream>.json` (+ per-window npz), summary in
`RESPONSE_TO_R3.md`.
