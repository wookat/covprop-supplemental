# CovProp — Supplemental Material (ICDE 2027)

Supplemental artifacts for *"CovProp: Propagating Upstream Forecast Uncertainty of
Future Covariates into Conformal Prediction Intervals"* (ICDE 2027 submission).

## Requirements

Python 3.9+ with `numpy`, `scipy`, `matplotlib` (see `requirements.txt`).
CPU-only; no pandas, sklearn, or deep-learning framework needed — all
backbones are the linear / per-horizon MLP models disclosed in the paper.

```bash
pip install -r requirements.txt
```

## Layout

```
src/                       Shared protocol stack — the common machinery every
                           experiment imports (frozen between rounds)
  covprop_core.py          Windowing, conformal machinery, evaluation, Winkler
  adaptive_baselines.py    Dataset loaders + CQR/NorCP/SPCI implementations
  upstream.py              Upstream forecaster, MC propagation, significance tests
  share.py                 Propagation-share statistic, binned lambda, loaders
  share_dump.py            Share layer + per-window ribbon export
  core/
    baselines.py           ACI, NexCP, Winkler scoring
    data.py                epftoolbox market loader, split helpers
    rc_cp.py               MondrianCP (RC-CP baseline)
    syndeg.py              SynDeg / SynDeg-hetero generators
experiments/               Entry-point drivers — one per paper claim
  covprop_poolsize.py      Error-pool subsampling ablation (Table VII)
  covprop_biaslog.py       Biased-log + propagation-only ablation (Sec. VII-C)
  covprop_hybrid.py        Pre-registered hybrid experiment (Sec. VII-C)
  covprop_sweep.py         Degradation sweep driver (Table VIII, Sec. VII-E)
  backtest_extrap.py       Temporal-extrapolation extension (H5)
  dump_ribbon.py           Per-window ribbon export, all 7 streams (Sec. VII-D)
  policy_analysis.py       Selector analysis over the sweep (Table VIII)
  policy_analysis_streamlevel.py  Stream-level selector check
  bench_overhead.py        Per-window latency/memory benchmark (Sec. VIII)
  bench_fleet.py           Fleet-level overhead benchmark (Table IX)
  bench_fleet_scaling.py   Multiprocessing scaling variant (H-S3)
datasets/                  Data fetch instructions (README only; no data ships)
prereg/                    Pre-registered protocols frozen before each run
  PREREGISTRATION.md       Main protocol (H1–H7, grid, thresholds)
  PREREG_HYBRID.md         Hybrid-experiment criteria (Sec. VII-C)
  PREREG_FIX.md            Fix-protocol amendment
results/                   Raw artifacts, organized by claim
  ad_analysis.json         Applicability-domain analysis (Sec. VII-D)
  hybrid/                  hybrid_<stream>.json + perwin_*.npz (Tables IV–VI)
  poolsize/                poolsize_<stream>.json (Table VII)
  biased_log/              bias_<stream>.json (Sec. VII-C ablation)
  policy/                  sweep_*.json + sweep5_*.json + policy_*.json (Table VIII)
  overhead/                overhead_<stream>.json (Sec. VIII)
  ribbon/                  ribbon_<stream>.npz (Sec. VII-D; claim-relevant fields)
  logs/                    Run logs for audit
figures/                   Figure-rendering utilities (not needed for results)
supplementary/             supplementary.pdf — long-form tables omitted from the
                           paper body, generated mechanically from results/
                           (build_supplementary.py --pdf rebuilds it)
PROVENANCE.md              Environment and seed record for the reported numbers
```

## Run

Drivers self-locate `src/` relative to their own file — run from anywhere:

```bash
export LTSF_DIR=/path/to/ltsf_csvs     # ETT/weather/electricity (else src/datasets/)
export EPF_DIR=/path/to/epf_csvs       # NP/PJM/BE/FR/DE markets (else ~/epf/)

python experiments/covprop_biaslog.py              # ~9 min, 7 streams
python experiments/dump_ribbon.py                  # ~15 min, 7 streams, seed 0
python experiments/policy_analysis.py              # seconds; reads results/policy/
python experiments/bench_overhead.py --streams NP --seeds 0
```

Each driver writes its outputs under `results/<topic>/` alongside the shipped
files (regeneration is additive; shipped JSONs are byte-identical provenance).
The `src/` protocol layers can also be run standalone; they write their
intermediate artifacts to `results/baselines/`, `results/propagation/`, and
`results/share/` respectively (created on demand, not shipped).

## Artifact → paper claim map

| Paper item | Produced by |
|---|---|
| Table I (SynDeg main results) | `experiments/covprop_poolsize.py` |
| Table II (nine real datasets) | `experiments/covprop_poolsize.py` |
| Fig. 1 (coverage collapse curve) | `experiments/covprop_poolsize.py` + `figures/redraw_batchB.py` |
| Figs. 3–4, 6 (conditional coverage, share bins) | `src/share_dump.py` + `figures/` |
| Tables IV–VI (share-conditional, Winkler) | `src/share.py`; raw files in `results/hybrid/` |
| Table VII (error-pool subsampling) | `results/poolsize/poolsize_*.json` |
| Sec. VII-C biased-log + propagation-only floor | `experiments/covprop_biaslog.py`; `results/biased_log/` |
| Sec. VII-D per-window width–\|η\| correlation | `results/ribbon/ribbon_*.npz` (`half`, `mag`); `experiments/dump_ribbon.py` |
| Table VIII + Sec. VII-E backtest selector | `experiments/covprop_sweep.py`, `policy_analysis*.py`; `results/policy/` |
| Sec. VII-E H5 temporal extrapolation | `experiments/backtest_extrap.py`; `results/policy/sweep5_*.json` |
| Table IX + Sec. VIII overhead | `experiments/bench_overhead.py`, `bench_fleet*.py`; `results/overhead/` |
| Sec. VII-C negative results | `experiments/covprop_hybrid.py`, `results/ad_analysis.json` |

## Pre-specifications

The hypotheses, thresholds, grids, and pass/fail criteria frozen *before* the
corresponding runs are in `prereg/`. `PREREG_HYBRID.md` fixes the two hybrid
designs and the ≥5/7 acceptance criteria reported as not supported in Sec.
VII-C; `PREREGISTRATION.md` carries H1–H7 for the policy experiments.
