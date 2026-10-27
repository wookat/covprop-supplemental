# Provenance — pool-size sensitivity of the upstream error pool

Date: 2026-08-09 (host clock 2025-08-10 UTC on xu-1)
Host: xu-1 (dell@100.109.120.28), CPU-only
Python: /home/dell/anaconda3/envs/tsf-research/bin/python (numpy/scipy from tsf-research env)

## What was run
Driver: `experiments/covprop_poolsize.py`. It imports the *unmodified*
protocol stack under `src/` (covprop_core.py, adaptive_baselines.py,
upstream.py, share.py, core/) — no frozen file was
edited. Only change vs. the layer-3 protocol: the MC error pool `eps_ca[A]` is
subsampled without replacement to m windows (per seed, seed-dependent subset)
before both calibration-side and test-side draws.

Commands (run from `experiments/`):
  python covprop_poolsize.py --streams NP --seeds 0 --grid 25 0                    # smoke (poolsize_smoke.log)
  python covprop_poolsize.py --grid 5 10 25 50 100 200 0                           # NP, FR (poolsize_run.log; stopped at ETTh1 on data path)
  python covprop_poolsize.py --streams ETTh1 weather ETTh2 ETTm2 electricity --grid 5 10 25 50 100 200 0   # poolsize_run2.log (ETTm2 csv missing)
  python covprop_poolsize.py --streams ETTm2 electricity --grid 5 10 25 50 100 200 0                       # poolsize_run3.log

Seeds: 0 1 2 3 4 (default). Grid: m ∈ {5,10,25,50,100,200,full≈800}.
Methods re-evaluated: recal, covprop_mc, covprop_share (others don't consume the pool).

## Data
EPF markets: EPF_DIR=~/epf (NP/FR csv, same as R1–R3).
LTSF csvs symlinked in `src/datasets/`:
  ETTh1/ETTm1/electricity/weather -> ~/wookat/tsf-2027/t1_rc_cp_deep_v1/t1_rc_cp/datasets/
  ETTh2 -> ~/wookat/tsf-paper/external/OnlineTSF/dataset/ETTh2.csv
  ETTm2 -> ~/wookat/time-series-forecast/tslib/dataset/ETT-small/ETTm2.csv
All are the standard public benchmark files (byte-identical content to the
versions used in R3; loader identical).

## Outputs
poolsize_<stream>.json — per-stream: seeds, per-m aggregate (CF.agg), per-seed
runs, diagnostics (effective pool size, share-tercile coverage).
Raw R1/R2/R3 artifacts untouched; R4 outputs live only here (and in the supp zip).

## Consistency check vs. R3
Full-pool (m=full) covprop_mc Winkler matches the published R3 numbers:
NP 2.68, FR 2.52, ETTh1 1.37, weather 0.46, ETTh2 2.23, ETTm2 4.05,
electricity 3.31 (Table "r3winkler" in the paper).

## Headline reading (paper §sharecond + Appendix poolsize)
Coverage flat in m on all 7 streams (max drift 0.005); Winkler within 2–4% of
full pool at m=10, ~1% by m=100. ETTm2 non-monotone (failure stream; short
pools shrink honestly-huge widths). Negative/failure results retained.
