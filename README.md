# CovProp — Supplemental Material (ICDE 2027)

Supplemental artifacts for *"CovProp: Propagating Upstream Forecast Uncertainty of
Future Covariates into Conformal Prediction Intervals"* (ICDE 2027 submission).

## Requirements

Python 3.9+ with `numpy`, `scipy`, `matplotlib`. CPU-only; no pandas, sklearn,
or deep-learning framework needed — all backbones are the linear /
per-horizon MLP models disclosed in the paper.

```bash
pip install numpy scipy matplotlib
```

## Layout

```
src/                          Shared protocol stack (the common machinery every
                              experiment imports; cumulative layers, not versions)
  covprop_full.py             Windowing, conformal machinery, evaluation, Winkler
  covprop_r1.py               Protocol layer 1: 10 datasets, CQR/NorCP/SPCI impls
  covprop_r2.py               Protocol layer 2: upstream forecaster, MC propagation
  covprop_r3.py               Protocol layer 3: share statistic, binned lambda
  covprop_r3_dump.py          Layer 3 + per-window ribbon export
  core/                       Low-level library
    baselines.py              ACI, NexCP, Winkler scoring
    data.py                   epftoolbox market loader, split helpers
    rc_cp.py                  MondrianCP (RC-CP baseline)
    syndeg.py                 SynDeg / SynDeg-hetero generators
experiments/                  Entry-point drivers — each reproduces one protocol round
  covprop_r4.py               Round 4: pool-size ablation (Table VII)
  covprop_r5.py               Round 5: biased-log + propagation-only (Sec. VII-C)
  covprop_l2.py               Pre-registered hybrid experiment (Sec. VII-C)
  covprop_v2.py               Degradation sweep driver (Table VIII, Sec. VII-E)
  backtest_extrap.py          Temporal-extrapolation extension (H5)
  dump_ribbon.py              Per-window ribbon export, all 7 streams (Sec. VII-D)
  policy_analysis.py          Selector analysis over the sweep (Table VIII)
  policy_analysis_streamlevel.py  Stream-level selector check
  bench_overhead.py           Per-window latency/memory benchmark (Sec. VIII)
  bench_fleet.py              Fleet-level overhead benchmark (Table IX)
  bench_fleet_scaling.py      Multiprocessing scaling variant (H-S3)
datasets/                     Data fetch instructions (README only; no data ships)
prereg/                       Pre-registered protocols frozen before each run
  PREREGISTRATION.md  PREREG_L2.md  PREREG_R1FIX.md
results/                      Raw artifacts, organized by claim
  ad_analysis.json            Applicability-domain analysis (Sec. VII-D)
  results_l2/                 l2_<stream>.json + perwin_l2_<stream>.npz (Tables IV–VI)
  poolsize/                   poolsize_<stream>.json (Table VII)
  biased_log/                 bias_<stream>.json (Sec. VII-C r5 ablation)
  policy/                     sweep_*.json + sweep5_*.json + policy_*.json (Table VIII)
  overhead/                   overhead_<stream>.json (Sec. VIII)
  ribbon/                     ribbon_<stream>.npz (Sec. VII-D; slim fields)
  logs/                       full_r4*.log, sweep5.log, bench.log (run logs)
figures/                      Figure-rendering utilities (not needed for results)
PROVENANCE.md                 Environment and seed record for the reported numbers
```

Two naming conventions are deliberate and cited in the paper text:
`covprop_r<N>.py` are the frozen protocol *rounds* (later rounds import earlier
protocol layers unchanged — that layering is what kept protocols frozen
between rounds), and `results_l2/` / `ad_analysis.json` appear verbatim in
`\texttt{}` in the paper.

## Run

Drivers self-locate `src/` relative to their own file — run from anywhere:

```bash
export LTSF_DIR=/path/to/ltsf_csvs     # ETT/weather/electricity (else src/datasets/)
export EPF_DIR=/path/to/epf_csvs       # NP/PJM/BE/FR/DE markets (else ~/epf/)

python experiments/covprop_r5.py                 # ~9 min, 7 streams
python experiments/dump_ribbon.py                # ~15 min, 7 streams, seed 0
python experiments/policy_analysis.py            # seconds; reads results/policy/
python experiments/bench_overhead.py --streams NP --seeds 0
```

Each driver writes its outputs under `results/<topic>/` alongside the shipped
files (regeneration is additive; shipped JSONs are byte-identical provenance).

## Artifact → paper claim map

| Paper item | Produced by |
|---|---|
| Table I (SynDeg main results) | `experiments/covprop_r4.py` |
| Table II (nine real datasets) | `experiments/covprop_r4.py` |
| Fig. 1 (coverage collapse curve) | `experiments/covprop_r4.py` + `figures/redraw_batchB.py` |
| Figs. 3–4, 6 (conditional coverage, share bins) | `src/covprop_r3_dump.py` + `figures/` |
| Tables IV–VI (share-conditional, Winkler) | `src/covprop_r3.py`; raw files in `results/results_l2/` |
| Table VII (error-pool subsampling) | `results/poolsize/poolsize_*.json` |
| Sec. VII-C biased-log + propagation-only floor | `experiments/covprop_r5.py`; `results/biased_log/` |
| Sec. VII-D per-window width–\|η\| correlation | `results/ribbon/ribbon_*.npz` (`half`, `mag`); `experiments/dump_ribbon.py` |
| Table VIII + Sec. VII-E backtest selector | `experiments/covprop_v2.py`, `policy_analysis*.py`; `results/policy/` |
| Sec. VII-E H5 temporal extrapolation | `experiments/backtest_extrap.py`; `results/policy/sweep5_*.json` |
| Table IX + Sec. VIII overhead | `experiments/bench_overhead.py`, `bench_fleet*.py`; `results/overhead/` |
| Sec. VII-C negative results | `experiments/covprop_l2.py`, `results/ad_analysis.json` |

## Pre-specifications

The hypotheses, thresholds, grids, and pass/fail criteria frozen *before* the
corresponding runs are in `prereg/`. `PREREG_L2.md` fixes the two hybrid
designs and the ≥5/7 acceptance criteria reported as not supported in Sec.
VII-C; `PREREGISTRATION.md` carries H1–H7 for the policy experiments.
