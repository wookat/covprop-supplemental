# CovProp — Supplemental Material (ICDE 2027)

Supplemental artifacts for *"CovProp: Propagating Upstream Forecast Uncertainty of
Future Covariates into Conformal Prediction Intervals"* (ICDE 2027 submission).

This repository contains the experiment drivers, pre-registered protocols, raw
result files, and overhead benchmarks referenced in the paper. Datasets are
public and are fetched in their published formats (see Data below); no
proprietary data ships here.

## Layout

```
code/
  covprop_r4.py            Main experiment driver (Tables I–II, Figs. 1, 3, 4)
  covprop_r3.py            Real-stream / share-conditioning driver (Tables IV–VI, Fig. 6)
  covprop_l2.py            Pre-registered hybrid experiment (Sec. VII-C, "natural hybrid")
  covprop_r5.py            Biased-log + propagation-only ablation (Sec. VII-C, pool-size para.)
  bench_fleet.py           Fleet-level overhead benchmark (Sec. VIII, Table IX)
  bench_fleet_scaling.py   Scaling variant of the above
  make_fig_ribbon.py       Fig. 5 ribbon plot
  make_fig_arch.py         Deployment architecture figure
  redraw_batchB.py         Batch re-render for Figs. 1/3/4 and share plots
  extract_method_figs.py   Figure extraction utilities
  extract_retry.py         ^
prereg/
  PREREGISTRATION.md       Main pre-registered protocol (v2)
  PREREG_L2.md             Hybrid-experiment criteria frozen before the run (cited in Sec. VII-C)
  PREREG_R1FIX.md          R1 fix protocol
results/
  ad_analysis.json         Applicability-domain analysis (Sec. VII-C stress test)
  results_l2/              l2_<stream>.json + perwin_l2_<stream>.npz (Tables IV–VI)
  r4_results/              poolsize_<stream>.json (Table VII), run logs, PROVENANCE.md
  results_r5/              bias_<stream>.json — biased-log fractions + no-stage-2 variant
  ribbon/                  ribbon_<stream>.npz — per-window Y/mag/rho/half + all method
                           intervals (seed 0); basis of the width–|η| correlation claim
                           in Sec. VII-D (regenerate via code/dump_ribbon*.py)
```

## Artifact → paper claim map

| Paper item | Produced by |
|---|---|
| Table I (SynDeg main results) | `code/covprop_r4.py` |
| Table II (nine real datasets) | `code/covprop_r4.py` |
| Fig. 1 (coverage collapse curve) | `code/covprop_r4.py` + `redraw_batchB.py` |
| Figs. 3–4, 6 (conditional coverage, share bins) | `code/covprop_r3.py` + `redraw_batchB.py` |
| Tables IV–VI (share-conditional, Winkler) | `code/covprop_r3.py`; raw files in `results/results_l2/` |
| Table VII (error-pool subsampling) | `results/r4_results/poolsize_*.json` |
| Sec. VII-C biased-log + propagation-only floor | `code/covprop_r5.py`; `results/results_r5/bias_*.json` |
| Sec. VII-D per-window width–|η| correlation | `results/ribbon/ribbon_*.npz` (`half`, `mag` fields); `code/dump_ribbon.py`, `dump_ribbon3.py` |
| Table IX + Sec. VIII overhead | `code/bench_fleet.py`, `bench_fleet_scaling.py` |
| Sec. VII-C negative results | `code/covprop_l2.py`, `results/ad_analysis.json` |

## Pre-specifications

The hypotheses, thresholds, grids, and pass/fail criteria that were frozen
*before* the corresponding runs are in `prereg/`. In particular,
`PREREG_L2.md` fixes the two hybrid designs and the ≥5/7 acceptance criteria
reported as not supported in Sec. VII-C.

## Data

- Electricity markets (NP, PJM, BE, FR, DE): epftoolbox format
  (Lago et al., 2021).
- Long-horizon benchmarks: ETTh1/ETTm1/ETTh2/ETTm2, weather, electricity
  (Informer/Autoformer releases).
- The replay protocol corrupts deployment covariates with AR(1) noise +
  regime-switching volatility; the real-stream protocol uses the out-of-sample
  errors of the disclosed ridge upstream forecaster (per-channel per-horizon,
  24 lags + hour-of-day, train split only).

## Run

Each driver is a single-file Python script (numpy/pandas/sklearn only).
`PROVENANCE.md` records the exact environment and seeds used for the reported
numbers. Run logs (`full_r4*.log`, `smoke_r4.log`) ship under
`results/r4_results/` for audit.
