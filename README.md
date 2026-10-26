# CovProp — Supplemental Material (ICDE 2027)

Supplemental artifacts for *"CovProp: Propagating Upstream Forecast Uncertainty of
Future Covariates into Conformal Prediction Intervals"* (ICDE 2027 submission).

This repository contains the experiment drivers, pre-registered protocols, raw
result files, and overhead benchmarks referenced in the paper. Datasets are
public and are fetched in their published formats (see `datasets/`); no
proprietary data ships here.

## Requirements

Python 3.9+ with `numpy`, `scipy`, `matplotlib`. CPU-only. No GPU, pandas, or
deep-learning framework is required — all backbones are the linear /
per-horizon MLP models disclosed in the paper.

```bash
pip install numpy scipy matplotlib
```

## Layout

```
code/
  core/                    Shared library (windowing, calibration, baselines, loaders)
    baselines.py           ACI, NexCP, Winkler scoring
    data.py                epftoolbox market loader + split helpers
    rc_cp.py               MondrianCP (RC-CP baseline)
    syndeg.py              SynDeg / SynDeg-hetero generators
  covprop_full.py          Core CovProp machinery (MC propagation, stage-2 wrapper)
  covprop_r1.py            Round-1 driver: 10 datasets, extra baselines, sig. tests
  covprop_r2.py            Round-2 driver: upstream forecaster + replay protocol
  covprop_r3.py            Round-3 driver: share-conditioning (Tables IV–VI, Fig. 6)
  covprop_r3_dump.py       Round-3 driver with per-window ribbon export
  covprop_r4.py            Round-4 driver: main real-stream protocol (Tables I–II)
  covprop_r5.py            Round-5 driver: biased-log + propagation-only ablation
  covprop_l2.py            Pre-registered hybrid experiment (Sec. VII-C)
  dump_ribbon.py           Per-window ribbon export, all 7 streams (Sec. VII-D)
  bench_fleet.py           Fleet-level overhead benchmark (Sec. VIII)
  bench_fleet_scaling.py   Scaling variant of the above
  figures/                 Figure-rendering utilities (not needed for results)
datasets/                  Data fetch instructions (README only; no data ships)
prereg/
  PREREGISTRATION.md       Main pre-registered protocol (v2)
  PREREG_L2.md             Hybrid-experiment criteria frozen before the run
  PREREG_R1FIX.md          R1 fix protocol
results/
  ad_analysis.json         Applicability-domain analysis (Sec. VII-D)
  results_l2/              l2_<stream>.json + perwin_l2_<stream>.npz (Tables IV–VI)
  r4_results/              poolsize_<stream>.json (Table VII), run logs, PROVENANCE
  results_r5/              bias_<stream>.json — biased-log + no-stage-2 variants
  ribbon/                  ribbon_<stream>.npz — per-window half/mag/rho + intervals
                           (slim: claim-relevant fields; full set via dump_ribbon.py)
PROVENANCE.md              Environment and seed record for the reported numbers
```

Naming: `covprop_r<N>.py` are the frozen protocol *rounds* cited in the paper
text (e.g. `\texttt{covprop\_r5.py}` in Sec. VII-C); `t1_harness` internals are
under `code/core/`.

## Run

Each driver is a self-contained script; it locates `code/` and `code/core/`
via its own path, so run it from anywhere:

```bash
export LTSF_DIR=/path/to/ltsf_csvs     # ETT/weather/electricity
export EPF_DIR=/path/to/epf_csvs       # NP/PJM/BE/FR/DE markets

python code/covprop_r5.py              # biased-log + no-stage-2 grid, ~9 min (7 streams)
python code/dump_ribbon.py             # per-window ribbons, ~15 min (7 streams, seed 0)
python code/covprop_r4.py              # main real-stream protocol (long; see PROVENANCE)
```

Results are written next to the driver outputs documented in `PROVENANCE.md`;
run logs ship under `results/r4_results/` for audit.

## Artifact → paper claim map

| Paper item | Produced by |
|---|---|
| Table I (SynDeg main results) | `code/covprop_r4.py` |
| Table II (nine real datasets) | `code/covprop_r4.py` |
| Fig. 1 (coverage collapse curve) | `code/covprop_r4.py` + `code/figures/redraw_batchB.py` |
| Figs. 3–4, 6 (conditional coverage, share bins) | `code/covprop_r3.py` + `code/figures/` |
| Tables IV–VI (share-conditional, Winkler) | `code/covprop_r3.py`; raw files in `results/results_l2/` |
| Table VII (error-pool subsampling) | `results/r4_results/poolsize_*.json` |
| Sec. VII-C biased-log + propagation-only floor | `code/covprop_r5.py`; `results/results_r5/bias_*.json` |
| Sec. VII-D per-window width–\|η\| correlation | `results/ribbon/ribbon_*.npz` (`half`, `mag` fields); `code/dump_ribbon.py` |
| Table IX + Sec. VIII overhead | `code/bench_fleet.py`, `bench_fleet_scaling.py` |
| Sec. VII-C negative results | `code/covprop_l2.py`, `results/ad_analysis.json` |

## Pre-specifications

The hypotheses, thresholds, grids, and pass/fail criteria that were frozen
*before* the corresponding runs are in `prereg/`. In particular,
`PREREG_L2.md` fixes the two hybrid designs and the ≥5/7 acceptance criteria
reported as not supported in Sec. VII-C.
