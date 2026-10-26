"""CovProp R3 experiments (substantive strengthening round).

D1  CovProp-Share: the width decomposition upgraded from a diagnostic to an
    operational corrector.  Per test window a label-free propagation share
        rho_t = max(0, 1 - endo_half_t / half_t)
    (endo_half_t = static clean-calibration quantile x sd_t, half_t = the MC
    half-width; both computable at prediction time, no test labels) assigns
    the window to one of n_bins Mondrian strata; the second-stage lambda is
    then calibrated per stratum with the same matured online protocol as the
    uniform wrapper.  Guarantee: share-conditional validity under within-bin
    exchangeability (Mondrian); efficiency: a uniform lambda must inflate to
    the largest per-bin quantile to match worst-bin coverage, the binned
    allocation attains it at bin-specific quantiles.
D2  Three new real upstream forecast--realization error streams (ETTh2,
    ETTm2, electricity) under the R2 ridge upstream-forecaster protocol --
    out-of-corpus validation of the applicability-domain criterion.
D3  Head-to-head CQR / SPCI / normalized CP on the new streams (full R2
    method roster inherited).
AD  Predictive power of the calibration-estimated propagation share for the
    realized CovProp benefit, across all seven real streams.

Outputs: results_r3/   (CPU-only).
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "core"))

import covprop_full as CF                                        # noqa: E402
from covprop_full import (ALPHA, H, build_windows, train_base, split_q,
                          online_q, gauss_noise, make_hetero_sigma,
                          make_ar_error, evaluate)               # noqa: E402
from baselines import ACI, NexCP                                 # noqa: E402
from covprop_r1 import (load_any, run_cqr, run_norcp, run_spci,
                        winkler_pw)                              # noqa: E402
from covprop_r2 import (upstream_forecast, mc_quantiles_any, preds_any,
                        significance_r2)                         # noqa: E402

OUT = os.path.join(HERE, "results_r3")
NEW_STREAMS = ["ETTh2", "ETTm2", "electricity"]
OLD_STREAMS = ["NP", "FR", "ETTh1", "weather"]
ALL_STREAMS = OLD_STREAMS + NEW_STREAMS
N_BINS = 3


def load_any_r3(name):
    """Extends R1 load_any with ETTh2 / ETTm2 (same LTSF loader)."""
    import covprop_r1 as R1
    if name in ("ETTh2", "ETTm2"):
        a = R1.load_ltsf(name)
        mu = a[:int(len(a) * .6)].mean(0)
        sd = a[:int(len(a) * .6)].std(0) + 1e-8
        return (a - mu) / sd, "nonparam"
    return load_any(name)


# --------------------------------------------- D1: share-binned lambda ------
def share_variable(half, endo_half):
    """Label-free per-window propagation-share statistic (unclipped:
    1 - endo/half; may be negative when the MC half-width is below the
    endogenous one, which preserves the ordering needed for binning)."""
    return 1.0 - endo_half.mean(1) / np.maximum(half.mean(1), 1e-8)


def binned_lambda(sc_B, sc_T, rho_B, rho_T, ts_te, alpha, n_bins=N_BINS):
    """Mondrian second stage: per-share-bin matured online lambda."""
    edges = np.quantile(rho_B, np.arange(1, n_bins) / n_bins)
    bins_B = np.digitize(rho_B, edges)
    bins_T = np.digitize(rho_T, edges)
    lam = np.zeros((len(sc_T), sc_T.shape[1]))
    for b in range(n_bins):
        mB, mT = bins_B == b, bins_T == b
        if mT.sum() == 0:
            continue
        if mB.sum() < 30:                       # tiny stratum: fall back
            mB = np.ones(len(sc_B), bool)
        lam[mT] = online_q(sc_B[mB], sc_T[mT], np.asarray(ts_te)[mT], alpha)
    return lam, bins_T, edges


def binned_lambda_fixed(sc_B, sc_T, rho_B, rho_T, ts_te, alpha,
                        n_bins=N_BINS):
    """Strict Mondrian variant: bin edges estimated on the first half of the
    calibration split (B1), per-bin lambdas calibrated only on the second
    half (B2) -- the taxonomy is independent of the calibration scores it
    stratifies, matching the exact premise of Prop. share-conditional
    validity."""
    n1 = len(sc_B) // 2
    edges = np.quantile(rho_B[:n1], np.arange(1, n_bins) / n_bins)
    bins_B2 = np.digitize(rho_B[n1:], edges)
    bins_T = np.digitize(rho_T, edges)
    sc_B2 = sc_B[n1:]
    lam = np.zeros((len(sc_T), sc_T.shape[1]))
    for b in range(n_bins):
        mB, mT = bins_B2 == b, bins_T == b
        if mT.sum() == 0:
            continue
        if mB.sum() < 30:
            mB = np.ones(len(sc_B2), bool)
        lam[mT] = online_q(sc_B2[mB], sc_T[mT], np.asarray(ts_te)[mT], alpha)
    return lam, bins_T, edges


def cov_by_bins(Y, lo, hi, bins, n_bins=N_BINS):
    inside = (Y >= lo) & (Y <= hi)
    return [float(inside[bins == b].mean()) if (bins == b).any() else None
            for b in range(n_bins)]


# ---------------------------------------------------------- one R3 run ------
def run_once_r3(data, seed, source, mode="real", sigma_base=0.5, scale_b=0.3):
    rng = np.random.RandomState(seed)
    T = len(data)
    tr_end, cal_end = int(T * 0.6), int(T * 0.8)
    Xh_tr, Xc_tr, Y_tr, ts_tr = build_windows(data, 0, tr_end, 4000)
    Xh_ca, Xc_ca, Y_ca, ts_ca = build_windows(data, tr_end, cal_end, 1600)
    Xh_te, Xc_te, Y_te, ts_te = build_windows(data, cal_end, T, 1500)
    ncov = Xc_tr.shape[2]

    if mode == "real":
        eps_ca = upstream_forecast(data, ts_ca, tr_end) - Xc_ca
        eps_te = upstream_forecast(data, ts_te, tr_end) - Xc_te
    elif source == "gauss":
        sig_ca = make_hetero_sigma(len(Y_ca), sigma_base, rng)
        sig_te = make_hetero_sigma(len(Y_te), sigma_base, rng)
        eps_ca = gauss_noise(Xc_ca.shape, sig_ca[:, None, None], rng)
        eps_te = gauss_noise(Xc_te.shape, sig_te[:, None, None], rng)
    else:
        cov_sd = Xc_tr.reshape(-1, ncov).std(0)
        eps_ca = make_ar_error(len(Y_ca), H, ncov, scale_b * cov_sd, rng)
        eps_te = make_ar_error(len(Y_te), H, ncov, scale_b * cov_sd, rng)
    Xc_ca_obs, Xc_te_obs = Xc_ca + eps_ca, Xc_te + eps_te
    mag_te = np.abs(eps_te).mean(axis=(1, 2))

    sd_ca = Xh_ca.std(axis=1, keepdims=True) + 1e-5
    sd_te = Xh_te.std(axis=1, keepdims=True) + 1e-5

    ws = train_base(Xh_tr, Xc_tr, Y_tr, ts_tr)
    ws_drop = train_base(Xh_tr, Xc_tr, Y_tr, ts_tr, drop=0.3,
                         rng=np.random.RandomState(seed + 7))

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

    pB_d = preds_any(ws_drop, Xh_ca[B], Xc_ca_obs[B], ts_ca[B])
    pT_d = preds_any(ws_drop, Xh_te, Xc_te_obs, ts_te)
    q_d = online_q(np.abs(Y_ca[B] - pB_d) / sd_ca[B],
                   np.abs(Y_te - pT_d) / sd_te, ts_te, ALPHA)
    ev("exodrop", pT_d - q_d * sd_te, pT_d + q_d * sd_te)

    resB = (Y_ca[B] - p_caB_obs) / sd_ca[B]
    resT = (Y_te - p_te_obs) / sd_te
    for M in (ACI(ALPHA), NexCP(ALPHA)):
        c, q = M.run(resB, resT, list(ts_ca[B]), list(ts_te), H)
        ev(M.name, p_te_obs + (c - q) * sd_te, p_te_obs + (c + q) * sd_te)

    lo, hi = run_cqr(Xh_tr, Xc_tr, Y_tr, ts_tr,
                     (Xh_ca[B], Xc_ca_obs[B], Y_ca[B], ts_ca[B], sd_ca[B]),
                     (Xh_te, Xc_te_obs, Y_te, ts_te, sd_te), seed)
    ev("cqr", lo, hi)
    lo, hi = run_norcp(ws, (Xh_ca[A], Xc_ca_obs[A], Y_ca[A], ts_ca[A]),
                       (Xh_ca[B], Xc_ca_obs[B], Y_ca[B], ts_ca[B], sd_ca[B]),
                       (Xh_te, Xc_te_obs, Y_te, ts_te, sd_te))
    ev("norcp", lo, hi)
    lo, hi = run_spci(p_caB_obs, Y_ca[B], sd_ca[B], p_te_obs, Y_te, sd_te,
                      ts_te)
    ev("spci", lo, hi)

    # ---- CovProp-MC (uniform lambda) ----------------------------------------
    rng2 = np.random.RandomState(seed + 1)
    res_signed = res_A_clean * np.sign(rng2.normal(size=res_A_clean.shape))
    if mode != "real" and source == "gauss":
        samp_ca = lambda k: gauss_noise(Xc_ca_obs[B].shape,
                                        sig_ca[B][:, None, None], rng2)
        samp_te = lambda k: gauss_noise(Xc_te_obs.shape,
                                        sig_te[:, None, None], rng2)
    else:
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

    # ---- D1: CovProp-Share (propagation-share-binned lambda) ----------------
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

    # ACI composition on both wrappers
    rB_cp = (Y_ca[B] - mid_B) / half_B
    rT_cp = (Y_te - mid_T) / half_T
    c, q = ACI(ALPHA).run(rB_cp, rT_cp, list(ts_ca[B]), list(ts_te), H)
    ev("aci_covprop", mid_T + (c - q) * half_T, mid_T + (c + q) * half_T,
       endo_w)

    # share-conditional coverage for every method (evaluation only; rho is
    # label-free so this is a legitimate conditional-coverage axis)
    share_cov = {m: cov_by_bins(Y_te, lo, hi, bins_T)
                 for m, (lo, hi) in intervals.items()}
    diag = {"rho_edges": [float(e) for e in edges],
            "rho_B_mean": float(rho_B.mean()), "rho_T_mean": float(rho_T.mean()),
            "rho_T_by_bin": [float(rho_T[bins_T == b].mean())
                             if (bins_T == b).any() else None
                             for b in range(N_BINS)],
            "lam_uniform_mean": float(lam_u.mean()),
            "lam_share_by_bin": [float(lam_s[bins_T == b].mean())
                                 if (bins_T == b).any() else None
                                 for b in range(N_BINS)],
            "cov_by_share_tercile": share_cov,
            "prop_share_cal": float(np.maximum(
                0.0, 1.0 - endo_half_B.mean() / half_B.mean())),
            "prop_share_cal_unclipped": float(
                1.0 - endo_half_B.mean() / half_B.mean()),
            "rho_B_frac_pos": float((rho_B > 0).mean())}
    if mode == "real":
        # upstream sanity statistic: mean |upstream error| relative to the
        # covariate scale on the calibration stream (label-free, pre-deploy)
        diag["upstream_relerr_cal"] = float(
            np.abs(eps_ca).mean() / (Xc_ca.std() + 1e-8))

    # strict-Mondrian ablation: edges from B1, lambdas from B2 only
    lam_f, bins_Tf, _ = binned_lambda_fixed(sc_B, sc_T, rho_B, rho_T,
                                            ts_te, ALPHA)
    ev("covprop_share_fixededges", mid_T - lam_f * half_T,
       mid_T + lam_f * half_T, endo_w)
    diag["cov_by_share_tercile"]["covprop_share_fixededges"] = cov_by_bins(
        Y_te, *intervals["covprop_share_fixededges"], bins_T)
    diag["cov_by_fixededge_tercile"] = {
        m: cov_by_bins(Y_te, lo, hi, bins_Tf)
        for m, (lo, hi) in intervals.items()}
    diag["_ribbon"] = {
        "Y": np.asarray(Y_te), "ts": np.asarray(ts_te),
        "pred": np.asarray(p_te_obs), "mag": np.asarray(mag_te),
        "rho": np.asarray(rho_T), "sd": np.asarray(sd_te),
        "mid": np.asarray(mid_T), "half": np.asarray(half_T),
        **{f"iv_{m}": np.stack(v) for m, v in intervals.items()}}
    return out, per_win, diag


# ------------------------------------------------------------ drivers -------
def run_block(tag, datasets, mode, seeds):
    for ds in datasets:
        data, source = load_any_r3(ds)
        runs, pws, diags = [], [], []
        t0 = time.time()
        for s in seeds:
            o, pw, dg = run_once_r3(data, s, source, mode=mode)
            runs.append(o); pws.append(pw); diags.append(dg)
            print(f"  [{tag} {ds} seed {s}] covprop wink="
                  f"{o['covprop_mc']['winkler']:.3f} "
                  f"share wink={o['covprop_share']['winkler']:.3f} "
                  f"share worst-terc={min(x for x in dg['cov_by_share_tercile']['covprop_share'] if x is not None):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        rep = {"dataset": ds, "mode": mode, "seeds": list(seeds),
               "aggregate": CF.agg(runs), "runs": runs,
               "diagnostics": diags}
        seed_metrics = [{m: r[m]["winkler"] for m in r} for r in runs]
        rep["significance_winkler_vs_covprop_mc"] = \
            significance_r2(pws, seed_metrics, ref="covprop_mc")
        rep["significance_winkler_vs_covprop_share"] = \
            significance_r2(pws, seed_metrics, ref="covprop_share")
        # mean share-conditional coverage across seeds
        rep["cov_by_share_tercile"] = {
            m: [float(np.mean([d["cov_by_share_tercile"][m][b]
                               for d in diags
                               if d["cov_by_share_tercile"][m][b] is not None]))
                for b in range(N_BINS)]
            for m in diags[0]["cov_by_share_tercile"]}
        with open(f"{OUT}/{tag}_{ds}.json", "w") as f:
            json.dump(rep, f, indent=1)
        np.savez_compressed(f"{OUT}/perwin_{tag}_{ds}.npz",
                            **{f"s{si}_{m}": pw[m]
                               for si, pw in enumerate(pws) for m in pw})
        print(f"== {tag} {ds} done", flush=True)


def upstream_skill(ds):
    """Label-free pre-deployment upstream sanity check: mean |OOS error| of
    the ridge upstream forecaster on the calibration stream, relative to a
    persistence forecast (last observed covariate value held over the
    horizon).  >= 1 means the upstream forecaster is no better than
    persistence, i.e. uninformative."""
    data, _ = load_any_r3(ds)
    T = len(data)
    tr_end, cal_end = int(T * 0.6), int(T * 0.8)
    Xh, Xc, Y, ts = build_windows(data, tr_end, cal_end, 1600)
    cov = data[:, :-1]
    up = upstream_forecast(data, ts, tr_end)
    pers = np.stack([np.repeat(cov[t - 1:t, :], Xc.shape[1], axis=0)
                     for t in ts])
    return float(np.abs(up - Xc).mean() / np.abs(pers - Xc).mean())


def ad_analysis(seeds):
    """AD: does the calibration-estimated propagation share predict the
    realized CovProp benefit across the seven real streams?"""
    from scipy import stats as sps
    rows = []
    for ds in ALL_STREAMS:
        fp = f"{OUT}/real_{ds}.json"
        if not os.path.exists(fp):
            print(f"[ad] missing {fp}, skip"); continue
        rep = json.load(open(fp))
        share = float(np.mean([d.get("prop_share_cal_unclipped",
                                     d["prop_share_cal"])
                               for d in rep["diagnostics"]]))
        relerr = float(np.mean([d["upstream_relerr_cal"]
                                for d in rep["diagnostics"]])) \
            if "upstream_relerr_cal" in rep["diagnostics"][0] else None
        wk = rep["aggregate"]
        share_w = wk["covprop_mc"]["width_prop"] / max(
            wk["covprop_mc"]["width"]["mean"], 1e-12)
        gain_recal = (wk["recal"]["winkler"]["mean"]
                      - wk["covprop_share"]["winkler"]["mean"]) \
            / wk["recal"]["winkler"]["mean"]
        gain_mc = (wk["recal"]["winkler"]["mean"]
                   - wk["covprop_mc"]["winkler"]["mean"]) \
            / wk["recal"]["winkler"]["mean"]
        wt_gain = (wk["covprop_share"]["worst_tercile_cov"]["mean"]
                   - wk["recal"]["worst_tercile_cov"]["mean"])
        cu = rep["cov_by_share_tercile"]["covprop_mc"]
        cs = rep["cov_by_share_tercile"]["covprop_share"]
        rows.append({"dataset": ds, "prop_share_cal_unclipped": share,
                     "upstream_relerr_cal": relerr,
                     "upstream_skill_vs_persistence": upstream_skill(ds),
                     "prop_share_width": share_w,
                     "sharebin_cov_spread_uniform": max(cu) - min(cu),
                     "sharebin_cov_spread_share": max(cs) - min(cs),
                     "winkler_gain_share_vs_recal": gain_recal,
                     "winkler_gain_mc_vs_recal": gain_mc,
                     "worst_terc_gain_share_vs_recal": wt_gain})
        print(f"[ad {ds}] share={share:.3f} sharew={share_w:.3f} "
              f"gain(share)={gain_recal:+.3f} "
              f"gain(mc)={gain_mc:+.3f} wt_gain={wt_gain:+.3f}", flush=True)
    if len(rows) >= 4:
        res = {}
        for xkey in ("prop_share_cal_unclipped", "prop_share_width"):
            x = [r[xkey] for r in rows]
            for key in ("winkler_gain_share_vs_recal",
                        "winkler_gain_mc_vs_recal",
                        "worst_terc_gain_share_vs_recal"):
                y = [r[key] for r in rows]
                sp = sps.spearmanr(x, y)
                pe = sps.pearsonr(x, y)
                res[f"{xkey}__{key}"] = {
                    "spearman_rho": float(sp.statistic),
                    "spearman_p": float(sp.pvalue),
                    "pearson_r": float(pe.statistic),
                    "pearson_p": float(pe.pvalue)}
                print(f"[ad corr {xkey}~{key}] spearman={sp.statistic:.3f} "
                      f"(p={sp.pvalue:.3f}) pearson={pe.statistic:.3f} "
                      f"(p={pe.pvalue:.3f})", flush=True)
        out = {"per_dataset": rows, "correlations": res, "seeds": list(seeds)}
    else:
        out = {"per_dataset": rows, "seeds": list(seeds)}
    with open(f"{OUT}/ad_analysis.json", "w") as f:
        json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--exp", default="all",
                    choices=["all", "syndeg", "real_new", "real_old", "ad"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    if args.exp in ("all", "syndeg"):
        run_block("syndeg", ["SynDeg"], "replay", args.seeds)
    if args.exp in ("all", "real_new"):
        run_block("real", NEW_STREAMS, "real", args.seeds)
    if args.exp in ("all", "real_old"):
        run_block("real", OLD_STREAMS, "real", args.seeds)
    if args.exp in ("all", "ad"):
        ad_analysis(args.seeds)
    print(f"\ndone in {time.time()-t0:.0f}s -> {OUT}/", flush=True)


if __name__ == "__main__":
    main()
