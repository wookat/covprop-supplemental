"""CovProp R5: two ablations on the real-stream protocol.

(A) Biased upstream error log: restrict the MC pool to the smallest
    |eps|-fraction of logged error windows -> F_t under-reports the tail.
    Grid f in {1.0, 0.5, 0.25, 0.1} (1.0 = unbiased full pool).
(B) Propagation-only: same MC intervals with stage-2 lambda fixed to 1
    (no conformal correction), evaluated at every bias level.

Roster per grid point: recal, covprop_nocal (lambda=1), covprop_mc.
Outputs: results_r5/bias_<ds>.json   (CPU-only). Raw results untouched.
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "t1_harness"))

import covprop_full as CF                                        # noqa: E402
from covprop_full import (ALPHA, build_windows, train_base, split_q,
                          online_q, evaluate)                    # noqa: E402
from covprop_r1 import winkler_pw                                # noqa: E402
from covprop_r2 import (upstream_forecast, mc_quantiles_any,
                        preds_any)                               # noqa: E402
from covprop_r3 import (load_any_r3, ALL_STREAMS)                 # noqa: E402

OUT = os.path.join(HERE, "results_r5")
BIAS_GRID = [1.0, 0.5, 0.25, 0.1]


def run_once_bias(data, seed, keep_frac):
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
    if keep_frac < 1.0:
        # biased log: keep only the smallest-|eps| windows
        row_mag = np.abs(full_pool).mean(axis=(1, 2))
        n_keep = max(5, int(len(full_pool) * keep_frac))
        idx = np.argsort(row_mag)[:n_keep]
        pool = full_pool[idx]
    else:
        pool = full_pool
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
    # (B) propagation-only: no stage-2 correction
    ev("covprop_nocal", mid_T - half_T, mid_T + half_T, endo_w)
    # (A) full CovProp-MC under the biased log
    ev("covprop_mc", mid_T - lam_u * half_T, mid_T + lam_u * half_T, endo_w)
    return out, per_win, {"pool_m_effective": len(pool)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--streams", nargs="+", default=ALL_STREAMS)
    ap.add_argument("--grid", nargs="+", type=float, default=BIAS_GRID)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    for ds in args.streams:
        data, _ = load_any_r3(ds)
        rep = {"dataset": ds, "seeds": list(args.seeds), "grid": {}}
        for f in args.grid:
            res = [run_once_bias(data, s, f) for s in args.seeds]
            runs = [r[0] for r in res]
            diags = [r[2] for r in res]
            key = str(f)
            rep["grid"][key] = {
                "pool_m_effective": diags[0]["pool_m_effective"],
                "aggregate": CF.agg(runs),
                "runs": runs, "diagnostics": diags}
            a = rep["grid"][key]["aggregate"]
            print(f"[{ds} f={key}] mc cov={a['covprop_mc']['coverage']['mean']:.3f} "
                  f"worst={a['covprop_mc']['worst_tercile_cov']['mean']:.3f} | "
                  f"nocal cov={a['covprop_nocal']['coverage']['mean']:.3f} "
                  f"worst={a['covprop_nocal']['worst_tercile_cov']['mean']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        with open(f"{OUT}/bias_{ds}.json", "w") as fh:
            json.dump(rep, fh, indent=1)
        print(f"== {ds} done", flush=True)
    print(f"done in {time.time()-t0:.0f}s -> {OUT}/", flush=True)


if __name__ == "__main__":
    main()
