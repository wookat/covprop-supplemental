"""CovProp R4 experiment: how much logged upstream history is enough?

Addresses the "strong premise" critique: CovProp needs a characterized
upstream error distribution.  In deployment the pool is estimated from m
logged forecast--realization residual windows.  R4 quantifies the coverage /
efficiency of CovProp-MC and CovProp-Share as a function of the pool size m,
on all seven real upstream-error streams (unchanged R2/R3 protocol; the only
change is that the MC sampler draws from a random m-window subsample of the
calibration error pool eps_ca[A]).

Outputs: results_r4/  (CPU-only).  Raw R1/R2/R3 untouched.
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "t1_harness"))

import covprop_full as CF                                        # noqa: E402
from covprop_full import (ALPHA, H, build_windows, train_base, split_q,
                          online_q, evaluate)                    # noqa: E402
from covprop_r1 import winkler_pw                                # noqa: E402
from covprop_r2 import (upstream_forecast, mc_quantiles_any,
                        preds_any)                               # noqa: E402
from covprop_r3 import (load_any_r3, share_variable, binned_lambda,
                        cov_by_bins, N_BINS, ALL_STREAMS)        # noqa: E402

OUT = os.path.join(HERE, "results_r4")
POOL_GRID = [10, 25, 50, 100, 200, 0]        # 0 = full pool (R3 reference)


def run_once_pool(data, seed, pool_m):
    """R3 real-mode run restricted to {recal, covprop_mc, covprop_share},
    with the MC error pool subsampled to pool_m windows (0 = full)."""
    rng = np.random.RandomState(seed)
    T = len(data)
    tr_end, cal_end = int(T * 0.6), int(T * 0.8)
    Xh_tr, Xc_tr, Y_tr, ts_tr = build_windows(data, 0, tr_end, 4000)
    Xh_ca, Xc_ca, Y_ca, ts_ca = build_windows(data, tr_end, cal_end, 1600)
    Xh_te, Xc_te, Y_te, ts_te = build_windows(data, cal_end, T, 1500)

    eps_ca = upstream_forecast(data, ts_ca, tr_end) - Xc_ca
    eps_te = upstream_forecast(data, ts_te, tr_end) - Xc_te
    Xc_ca_obs, Xc_te_obs = Xc_ca + eps_ca, Xc_te + eps_te
    mag_te = np.abs(eps_te).mean(axis=(1, 2))

    sd_ca = Xh_ca.std(axis=1, keepdims=True) + 1e-5
    sd_te = Xh_te.std(axis=1, keepdims=True) + 1e-5

    ws = train_base(Xh_tr, Xc_tr, Y_tr, ts_tr)

    nA = len(Y_ca) // 2
    A, B = slice(0, nA), slice(nA, len(Y_ca))

    p_ca_clean = preds_any(ws, Xh_ca, Xc_ca, ts_ca)
    p_caB_obs = preds_any(ws, Xh_ca[B], Xc_ca_obs[B], ts_ca[B])
    p_te_obs = preds_any(ws, Xh_te, Xc_te_obs, ts_te)

    res_A_clean = np.abs(Y_ca[A] - p_ca_clean[A]) / sd_ca[A]
    res_B_noisy = np.abs(Y_ca[B] - p_caB_obs) / sd_ca[B]
    sc_clean_all = np.abs(Y_ca - p_ca_clean) / sd_ca
    endo_w = float((2 * split_q(sc_clean_all, ALPHA) * sd_te).mean())
    q_clean_static = split_q(sc_clean_all, ALPHA)
    q_rec = online_q(res_B_noisy, np.abs(Y_te - p_te_obs) / sd_te,
                     ts_te, ALPHA)

    out, per_win = {}, {}

    def ev(name, lo, hi, endo=None):
        out[name] = evaluate(Y_te, lo, hi, mag_te, endo_width=endo)
        per_win[name] = winkler_pw(Y_te, lo, hi)

    ev("recal", p_te_obs - q_rec * sd_te, p_te_obs + q_rec * sd_te, endo_w)

    rng2 = np.random.RandomState(seed + 1)
    res_signed = res_A_clean * np.sign(rng2.normal(size=res_A_clean.shape))
    full_pool = eps_ca[A]
    if pool_m and pool_m < len(full_pool):
        idx = rng.choice(len(full_pool), pool_m, replace=False)
        pool = full_pool[idx]
    else:
        pool = full_pool
    m_eff = len(pool)
    samp_ca = lambda k: pool[rng2.randint(0, len(pool), len(Xc_ca_obs[B]))]
    samp_te = lambda k: pool[rng2.randint(0, len(pool), len(Xc_te_obs))]
    lo_B, hi_B = mc_quantiles_any(ws, Xh_ca[B], Xc_ca_obs[B], ts_ca[B],
                                  samp_ca, res_signed, sd_ca[B], rng2)
    mid_B, half_B = (lo_B + hi_B) / 2, np.maximum((hi_B - lo_B) / 2, 1e-8)
    lo_T, hi_T = mc_quantiles_any(ws, Xh_te, Xc_te_obs, ts_te,
                                  samp_te, res_signed, sd_te, rng2)
    mid_T, half_T = (lo_T + hi_T) / 2, np.maximum((hi_T - lo_T) / 2, 1e-8)
    sc_B = np.abs(Y_ca[B] - mid_B) / half_B
    sc_T = np.abs(Y_te - mid_T) / half_T
    lam_u = online_q(sc_B, sc_T, np.asarray(ts_te), ALPHA)
    ev("covprop_mc", mid_T - lam_u * half_T, mid_T + lam_u * half_T, endo_w)

    endo_half_B = np.broadcast_to(q_clean_static[None, :] * sd_ca[B],
                                  half_B.shape)
    endo_half_T = np.broadcast_to(q_clean_static[None, :] * sd_te,
                                  half_T.shape)
    rho_B = share_variable(half_B, endo_half_B)
    rho_T = share_variable(half_T, endo_half_T)
    lam_s, bins_T, _ = binned_lambda(sc_B, sc_T, rho_B, rho_T, ts_te, ALPHA)
    ev("covprop_share", mid_T - lam_s * half_T, mid_T + lam_s * half_T,
       endo_w)
    share_terc = cov_by_bins(Y_te, mid_T - lam_s * half_T,
                             mid_T + lam_s * half_T, bins_T)
    return out, per_win, {"pool_m_effective": m_eff,
                          "share_cov_by_share_tercile": share_terc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--streams", nargs="+", default=ALL_STREAMS)
    ap.add_argument("--grid", nargs="+", type=int, default=POOL_GRID)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    for ds in args.streams:
        data, _ = load_any_r3(ds)
        rep = {"dataset": ds, "seeds": list(args.seeds), "grid": {}}
        for m in args.grid:
            res = [run_once_pool(data, s, m) for s in args.seeds]
            runs = [r[0] for r in res]
            diags = [r[2] for r in res]
            key = str(m if m else "full")
            rep["grid"][key] = {
                "m_effective": diags[0]["pool_m_effective"],
                "aggregate": CF.agg(runs),
                "runs": runs,
                "diagnostics": diags}
            a = rep["grid"][key]["aggregate"]
            print(f"[{ds} m={key}] mc cov={a['covprop_mc']['coverage']['mean']:.3f} "
                  f"wink={a['covprop_mc']['winkler']['mean']:.3f} "
                  f"share worst-terc={a['covprop_share']['worst_tercile_cov']['mean']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        with open(f"{OUT}/poolsize_{ds}.json", "w") as f:
            json.dump(rep, f, indent=1)
        print(f"== {ds} done", flush=True)
    print(f"done in {time.time()-t0:.0f}s -> {OUT}/", flush=True)


if __name__ == "__main__":
    main()
