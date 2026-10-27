"""E4: H4 backtest-selector validation-domain extension (preregistered in
PREREG_FIX.md).  Reruns the frozen 63-point degradation grid of
icde2027_v2_prep/experiments/covprop_sweep.py with two additions and no other
protocol change:
  (E4a) an *early* backtest whose decision window is moved half a B2 segment
        earlier (doubling the temporal gap to deployment);
  (E4c) 5 seeds instead of 3 (seeds 0-2 reproduce the frozen results_v2 runs
        bit-for-bit; read-only cross-check done in the analysis script).
Outputs results/sweep5_<stream>.json; frozen results_v2/ untouched.
"""
import json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "core"))

import covprop_core as CF                                        # noqa: E402
from covprop_core import (ALPHA, build_windows, train_base, split_q,
                          online_q, evaluate)                    # noqa: E402
from covprop_core import winkler as winkler_fn                   # noqa: E402
from adaptive_baselines import winkler_pw                                # noqa: E402
from upstream import (upstream_forecast, mc_quantiles_any,
                        preds_any)                               # noqa: E402
from share import (load_any_r3, share_variable, binned_lambda,
                        cov_by_bins)                             # noqa: E402

STREAMS = ["NP", "FR", "ETTh1", "weather", "ETTh2", "ETTm2", "electricity"]
POINTS = ([("noise", g) for g in (0.0, 0.5, 1.0, 2.0, 4.0)]
          + [("blend", l) for l in (0.25, 0.5, 0.75, 1.0)])
SEEDS = [0, 1, 2, 3, 4]


def backtest_on(Y, mid, half, sc, rho, p_obs, res_noisy, sd, ts, i0, i1, i2):
    """Backtest the 3-intervention menu: calibrate on [i0:i1], score on the
    labeled window [i1:i2].  Identical machinery to covprop_sweep.run_once_v2."""
    bt = {}
    q_rec = online_q(res_noisy[i0:i1],
                     np.abs(Y[i1:i2] - p_obs[i1:i2]) / sd[i1:i2],
                     ts[i1:i2], ALPHA)
    bt["recal"] = winkler_fn(Y[i1:i2], p_obs[i1:i2] - q_rec * sd[i1:i2],
                             p_obs[i1:i2] + q_rec * sd[i1:i2], ALPHA)
    lam = online_q(sc[i0:i1], sc[i1:i2], ts[i1:i2], ALPHA)
    bt["covprop_mc"] = winkler_fn(
        Y[i1:i2], mid[i1:i2] - lam * half[i1:i2],
        mid[i1:i2] + lam * half[i1:i2], ALPHA)
    lam_s, _, _ = binned_lambda(sc[i0:i1], sc[i1:i2], rho[i0:i1],
                                rho[i1:i2], ts[i1:i2], ALPHA)
    bt["covprop_share"] = winkler_fn(
        Y[i1:i2], mid[i1:i2] - lam_s * half[i1:i2],
        mid[i1:i2] + lam_s * half[i1:i2], ALPHA)
    return bt


def run_once_v3(data, seed, mech, level):
    T = len(data)
    tr_end, cal_end = int(T * 0.6), int(T * 0.8)
    Xh_tr, Xc_tr, Y_tr, ts_tr = build_windows(data, 0, tr_end, 4000)
    Xh_ca, Xc_ca, Y_ca, ts_ca = build_windows(data, tr_end, cal_end, 1600)
    Xh_te, Xc_te, Y_te, ts_te = build_windows(data, cal_end, T, 1500)

    cov = data[:, :-1]
    Hh = Xc_ca.shape[1]
    pers_ca = np.stack([np.repeat(cov[t - 1:t, :], Hh, axis=0)
                        for t in ts_ca])
    pers_te = np.stack([np.repeat(cov[t - 1:t, :], Hh, axis=0)
                        for t in ts_te])

    up_ca = upstream_forecast(data, ts_ca, tr_end)
    up_te = upstream_forecast(data, ts_te, tr_end)
    if mech == "noise":
        scale_j = np.sqrt(((up_ca - Xc_ca) ** 2).mean(axis=(0, 1))) + 1e-12
        rng_n = np.random.RandomState(10000 + seed)
        up_ca = up_ca + level * scale_j * rng_n.normal(size=up_ca.shape)
        up_te = up_te + level * scale_j * rng_n.normal(size=up_te.shape)
    elif mech == "blend":
        up_ca = (1 - level) * up_ca + level * pers_ca
        up_te = (1 - level) * up_te + level * pers_te
    else:
        raise ValueError(mech)

    eps_ca, eps_te = up_ca - Xc_ca, up_te - Xc_te
    Xc_ca_obs, Xc_te_obs = Xc_ca + eps_ca, Xc_te + eps_te
    mag_te = np.abs(eps_te).mean(axis=(1, 2))

    skill = float(np.abs(eps_ca).mean() / np.abs(pers_ca - Xc_ca).mean())

    sd_ca = Xh_ca.std(axis=1, keepdims=True) + 1e-5
    sd_te = Xh_te.std(axis=1, keepdims=True) + 1e-5
    ws = train_base(Xh_tr, Xc_tr, Y_tr, ts_tr)

    nA = len(Y_ca) // 2
    A, B = slice(0, nA), slice(nA, len(Y_ca))

    p_ca_clean = preds_any(ws, Xh_ca, Xc_ca, ts_ca)
    p_caB_obs = preds_any(ws, Xh_ca[B], Xc_ca_obs[B], ts_ca[B])
    p_te_clean = preds_any(ws, Xh_te, Xc_te, ts_te)
    p_te_obs = preds_any(ws, Xh_te, Xc_te_obs, ts_te)

    res_A_clean = np.abs(Y_ca[A] - p_ca_clean[A]) / sd_ca[A]
    res_B_noisy = np.abs(Y_ca[B] - p_caB_obs) / sd_ca[B]
    sc_clean_all = np.abs(Y_ca - p_ca_clean) / sd_ca
    endo_w = float((2 * split_q(sc_clean_all, ALPHA) * sd_te).mean())
    q_clean_static = split_q(sc_clean_all, ALPHA)

    q_orc = online_q(sc_clean_all, np.abs(Y_te - p_te_clean) / sd_te,
                     ts_te, ALPHA)
    q_rec = online_q(res_B_noisy, np.abs(Y_te - p_te_obs) / sd_te,
                     ts_te, ALPHA)

    out, per_win, intervals = {}, {}, {}

    def ev(name, lo, hi, endo=None):
        out[name] = evaluate(Y_te, lo, hi, mag_te, endo_width=endo)
        per_win[name] = winkler_pw(Y_te, lo, hi)
        intervals[name] = (lo, hi)

    ev("oracle", p_te_clean - q_orc * sd_te, p_te_clean + q_orc * sd_te)
    ev("naive", p_te_obs - q_clean_static * sd_te,
       p_te_obs + q_clean_static * sd_te, endo_w)
    ev("recal", p_te_obs - q_rec * sd_te, p_te_obs + q_rec * sd_te, endo_w)

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
    lam_s, bins_T, edges = binned_lambda(sc_B, sc_T, rho_B, rho_T, ts_te,
                                         ALPHA)
    ev("covprop_share", mid_T - lam_s * half_T, mid_T + lam_s * half_T,
       endo_w)

    # backtests: late (identical to frozen H4) and early (E4a) ---------------
    nB = len(Y_ca[B])
    h2 = nB // 2
    q1 = h2 // 2
    ts_B = np.asarray(ts_ca[B])
    Y_B, sd_B = Y_ca[B], sd_ca[B]
    # late (frozen H4): calibrate on B[0:h2], score on B[h2:nB]; decision
    # information ends adjacent to the test segment.
    bt_late = backtest_on(Y_B, mid_B, half_B, sc_B, rho_B, p_caB_obs,
                          res_B_noisy, sd_B, ts_B, 0, h2, nB)
    # early (E4a): calibrate on B[0:q1], score on B[q1:h2]; all decision
    # information ends at index h2, leaving a gap of half the B segment
    # before deployment (temporal gap doubled relative to late).
    bt_early = backtest_on(Y_B, mid_B, half_B, sc_B, rho_B, p_caB_obs,
                           res_B_noisy, sd_B, ts_B, 0, q1, h2)

    share_cov = {m: cov_by_bins(Y_te, lo, hi, bins_T)
                 for m, (lo, hi) in intervals.items()}
    diag = {"mech": mech, "level": level,
            "backtest_winkler": bt_late,
            "backtest_winkler_early": bt_early,
            "skill_vs_persistence": skill,
            "prop_share_cal_unclipped": float(
                1.0 - endo_half_B.mean() / half_B.mean()),
            "rho_edges": [float(e) for e in edges],
            "cov_by_share_tercile": share_cov}
    return out, per_win, diag


def main():
    out_dir = os.path.join(ROOT, "results", "policy")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    for ds in STREAMS:
        data, _ = load_any_r3(ds)
        rep = {"dataset": ds, "seeds": SEEDS, "points": []}
        for mech, lv in POINTS:
            runs, diags = [], []
            for s in SEEDS:
                o, pw, dg = run_once_v3(data, s, mech, lv)
                runs.append(o); diags.append(dg)
                print(f"  [{ds} {mech}={lv} seed {s}] "
                      f"recal={o['recal']['winkler']:.3f} "
                      f"mc={o['covprop_mc']['winkler']:.3f} "
                      f"share_w={o['covprop_share']['winkler']:.3f} "
                      f"btL={min(dg['backtest_winkler'], key=dg['backtest_winkler'].get)} "
                      f"btE={min(dg['backtest_winkler_early'], key=dg['backtest_winkler_early'].get)} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            rep["points"].append({
                "mech": mech, "level": lv,
                "aggregate": CF.agg(runs), "runs": runs,
                "diagnostics": diags,
                "backtest_winkler": {
                    m: float(np.mean([d["backtest_winkler"][m]
                                      for d in diags]))
                    for m in diags[0]["backtest_winkler"]},
                "backtest_winkler_early": {
                    m: float(np.mean([d["backtest_winkler_early"][m]
                                      for d in diags]))
                    for m in diags[0]["backtest_winkler_early"]},
                "skill_vs_persistence": float(np.mean(
                    [d["skill_vs_persistence"] for d in diags])),
                "prop_share_cal_unclipped": float(np.mean(
                    [d["prop_share_cal_unclipped"] for d in diags]))})
        with open(os.path.join(out_dir, f"sweep5_{ds}.json"), "w") as f:
            json.dump(rep, f, indent=1)
        print(f"== {ds} done ({time.time()-t0:.0f}s)", flush=True)
    print(f"\nall done in {time.time()-t0:.0f}s -> {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
