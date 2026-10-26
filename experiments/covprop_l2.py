"""L2 (R3 convergence round): preregistered CQR x CovProp hybrids.

See PREREG_L2.md (registered before running).  Two designs:
  hybrid_mc_cqr       -- input-side MC propagation into the CQR pinball heads.
  hybrid_cqr_covwidth -- CQR intervals, conformal correction normalized by the
                         CovProp-MC propagated half-width.
Baselines recomputed in-run for paired per-window tests: recal, cqr,
covprop_mc, covprop_share.  Plus a calibration-split backtest between the two
hybrids (same selector as the paper's SS Policy) so H8a uses a label-free
per-stream hybrid choice.

Outputs: results_l2/   (CPU-only).
"""
import argparse, json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "core"))

import covprop_full as CF                                        # noqa: E402
from covprop_full import (ALPHA, H, K_MC, build_windows, train_base, split_q,
                          online_q, features, evaluate)          # noqa: E402
from covprop_r1 import (run_cqr, fit_pinball, pred_lin,
                        winkler_pw)                              # noqa: E402
from covprop_r2 import (upstream_forecast, mc_quantiles_any, preds_any,
                        significance_r2)                         # noqa: E402
from covprop_r3 import (load_any_r3, share_variable, binned_lambda,
                        ALL_STREAMS)                             # noqa: E402

OUT = os.path.join(ROOT, "results", "results_l2")


def train_cqr_heads(Xh_tr, Xc_tr, Y_tr, ts_tr):
    ws_lo, ws_hi = [], []
    for h in range(H):
        Ftr = features(Xh_tr, Xc_tr, ts_tr, h)
        ws_lo.append(fit_pinball(Ftr, Y_tr[:, h], ALPHA / 2))
        ws_hi.append(fit_pinball(Ftr, Y_tr[:, h], 1 - ALPHA / 2))
    return ws_lo, ws_hi


def cqr_pred(ws_lo, ws_hi, Xh, Xc, ts):
    lo = np.stack([pred_lin(ws_lo[h], features(Xh, Xc, ts, h))
                   for h in range(H)], 1)
    hi = np.stack([pred_lin(ws_hi[h], features(Xh, Xc, ts, h))
                   for h in range(H)], 1)
    return lo, hi


def mc_cqr(ws_lo, ws_hi, Xh, Xc_obs, ts, sampler, rng, k_mc=K_MC):
    """Hybrid A core: pool the pinball-head endpoints over K propagated
    covariate samples; mixture quantiles as the pre-correction interval."""
    LO, HI = [], []
    for k in range(k_mc):
        lo_k, hi_k = cqr_pred(ws_lo, ws_hi, Xh, Xc_obs + sampler(k), ts)
        LO.append(lo_k); HI.append(hi_k)
    LO, HI = np.stack(LO, 0), np.stack(HI, 0)
    return (np.quantile(LO, ALPHA / 2, axis=0),
            np.quantile(HI, 1 - ALPHA / 2, axis=0))


def run_once_l2(data, seed, source):
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

    out, per_win = {}, {}

    def ev(name, lo, hi, endo=None):
        out[name] = evaluate(Y_te, lo, hi, mag_te, endo_width=endo)
        per_win[name] = winkler_pw(Y_te, lo, hi)
        return lo, hi

    q_rec = online_q(res_B_noisy, np.abs(Y_te - p_te_obs) / sd_te,
                     ts_te, ALPHA)
    ev("recal", p_te_obs - q_rec * sd_te, p_te_obs + q_rec * sd_te, endo_w)

    lo, hi = run_cqr(Xh_tr, Xc_tr, Y_tr, ts_tr,
                     (Xh_ca[B], Xc_ca_obs[B], Y_ca[B], ts_ca[B], sd_ca[B]),
                     (Xh_te, Xc_te_obs, Y_te, ts_te, sd_te), seed)
    ev("cqr", lo, hi)

    # ---- CovProp-MC + Share (identical to R3) ------------------------------
    rng2 = np.random.RandomState(seed + 1)
    res_signed = res_A_clean * np.sign(rng2.normal(size=res_A_clean.shape))
    pool = eps_ca[A]
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
    lam_s, _, _ = binned_lambda(sc_B, sc_T, rho_B, rho_T, ts_te, ALPHA)
    ev("covprop_share", mid_T - lam_s * half_T, mid_T + lam_s * half_T,
       endo_w)

    # ---- Hybrid A: MC propagation through the CQR heads --------------------
    ws_lo, ws_hi = train_cqr_heads(Xh_tr, Xc_tr, Y_tr, ts_tr)
    rng3 = np.random.RandomState(seed + 2)
    samp_caB = lambda k: pool[rng3.randint(0, len(pool), len(Xc_ca_obs[B]))]
    samp_teA = lambda k: pool[rng3.randint(0, len(pool), len(Xc_te_obs))]
    loA_B, hiA_B = mc_cqr(ws_lo, ws_hi, Xh_ca[B], Xc_ca_obs[B], ts_ca[B],
                          samp_caB, rng3)
    loA_T, hiA_T = mc_cqr(ws_lo, ws_hi, Xh_te, Xc_te_obs, ts_te,
                          samp_teA, rng3)
    scA_B = np.maximum(loA_B - Y_ca[B], Y_ca[B] - hiA_B) / sd_ca[B]
    scA_T = np.maximum(loA_T - Y_te, Y_te - hiA_T) / sd_te
    qA = online_q(scA_B, scA_T, np.asarray(ts_te), ALPHA)
    ev("hybrid_mc_cqr", loA_T - qA * sd_te, hiA_T + qA * sd_te, endo_w)

    # ---- Hybrid B: CQR scores normalized by the CovProp half-width ---------
    loC_B, hiC_B = cqr_pred(ws_lo, ws_hi, Xh_ca[B], Xc_ca_obs[B], ts_ca[B])
    loC_T, hiC_T = cqr_pred(ws_lo, ws_hi, Xh_te, Xc_te_obs, ts_te)
    scB_B = np.maximum(loC_B - Y_ca[B], Y_ca[B] - hiC_B) / half_B
    scB_T = np.maximum(loC_T - Y_te, Y_te - hiC_T) / half_T
    qB = online_q(scB_B, scB_T, np.asarray(ts_te), ALPHA)
    ev("hybrid_cqr_covwidth", loC_T - qB * half_T, hiC_T + qB * half_T,
       endo_w)

    # ---- label-free calibration-split backtest between the two hybrids -----
    # (same selector construction as SS Policy: calibrate on first half of B,
    #  score on the labelled second half; no test data touched)
    n2 = len(Y_ca[B]) // 2
    B1, B2 = slice(0, n2), slice(n2, len(Y_ca[B]))
    ts_B = np.asarray(ts_ca[B])
    bt = {}
    for name, (scb, sct_none, mk) in {
        "hybrid_mc_cqr": (scA_B, None,
                          lambda q: (loA_B[B2] - q * sd_ca[B][B2],
                                     hiA_B[B2] + q * sd_ca[B][B2])),
        "hybrid_cqr_covwidth": (scB_B, None,
                                lambda q: (loC_B[B2] - q * half_B[B2],
                                           hiC_B[B2] + q * half_B[B2]))}.items():
        qbt = online_q(scb[B1], scb[B2], ts_B[B2], ALPHA)
        lo_b, hi_b = mk(qbt)
        bt[name] = float(winkler_pw(Y_ca[B][B2], lo_b, hi_b).mean())
    pick = min(bt, key=bt.get)
    out["backtest_hybrid"] = {"scores": bt, "picked": pick,
                              "picked_test_winkler":
                                  out[pick]["winkler"]}
    per_win["hybrid_backtestpick"] = per_win[pick]
    out["hybrid_backtestpick"] = out[pick]
    return out, per_win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--streams", nargs="+", default=ALL_STREAMS)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    for ds in args.streams:
        data, source = load_any_r3(ds)
        runs, pws = [], []
        for s in args.seeds:
            o, pw = run_once_l2(data, s, source)
            runs.append(o); pws.append(pw)
            print(f"  [l2 {ds} seed {s}] cqr={o['cqr']['winkler']:.3f} "
                  f"hybA={o['hybrid_mc_cqr']['winkler']:.3f} "
                  f"hybB={o['hybrid_cqr_covwidth']['winkler']:.3f} "
                  f"pick={o['backtest_hybrid']['picked']} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        metric_keys = [m for m in runs[0] if m != "backtest_hybrid"]
        agg_runs = [{m: r[m] for m in metric_keys} for r in runs]
        rep = {"dataset": ds, "mode": "real", "seeds": list(args.seeds),
               "aggregate": CF.agg(agg_runs), "runs": runs}
        seed_metrics = [{m: r[m]["winkler"] for m in metric_keys}
                        for r in runs]
        rep["significance_winkler_vs_cqr"] = \
            significance_r2(pws, seed_metrics, ref="cqr")
        rep["significance_winkler_vs_covprop_share"] = \
            significance_r2(pws, seed_metrics, ref="covprop_share")
        with open(f"{OUT}/l2_{ds}.json", "w") as f:
            json.dump(rep, f, indent=1)
        np.savez_compressed(f"{OUT}/perwin_l2_{ds}.npz",
                            **{f"s{si}_{m}": pw[m]
                               for si, pw in enumerate(pws) for m in pw})
        print(f"== l2 {ds} done", flush=True)
    print(f"\ndone in {time.time()-t0:.0f}s -> {OUT}/", flush=True)


if __name__ == "__main__":
    main()
