"""CovProp R1 experiments: 10 datasets, 3 extra strong baselines, 5 seeds,
paired significance tests (Wilcoxon per-window + t-test per-seed, Holm).

Datasets
--------
SynDeg-hetero (gauss source, self-reported sigma_t) and 9 real datasets
(nonparam source, upstream error replayed as AR(1)+regime-switching noise,
CovProp estimates the error distribution from historical residuals only):
  EPF electricity markets: NP, PJM, BE, FR, DE   (2 exogenous forecasts)
  LTSF suite: ETTh1, ETTm1 (hourly-subsampled), weather (hourly-subsampled),
              electricity (top-4 |corr| client channels as covariates, OT target)

Extra baselines (head-to-head, same backbone features, same matured-feedback
online recalibration protocol as every other method; hyperparameters fixed
across all datasets and disclosed in the paper):
  cqr    Conformalized Quantile Regression (Romano et al., 2019): linear
         pinball heads at alpha/2, 1-alpha/2 per horizon, CQR conformal
         correction on the noisy stream.
  norcp  Normalized (locally-weighted) conformal (Papadopoulos et al., 2011;
         Lei et al., 2018): linear difficulty model g(x_obs) fit on noisy
         split-A residual magnitudes; scores |y-f|/g.
  spci   SPCI-style residual-quantile regression (Xu & Xie, 2023 flavor):
         linear quantile regression of the normalized residual magnitude on
         a window of recent matured residual statistics (fit once on the
         calibration stream; the conformal wrapper adapts online).

Outputs: results_r1/*.json/.csv/.npz/.png  (CPU-only)
"""
import argparse, csv, json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = HERE
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "core"))

import covprop_full as CF                      # noqa: E402
from covprop_full import (ALPHA, H, L_HIST, build_windows, features,     # noqa
                          train_base, preds, split_q, online_q, gauss_noise,
                          make_hetero_sigma, make_ar_error, covprop_mc,
                          covprop_nested, evaluate, load_dataset)
from baselines import ACI, NexCP, winkler      # noqa: E402
from rc_cp import MondrianCP                   # noqa: E402
from scipy.stats import norm                   # noqa: E402

OUT = os.path.join(HERE, "results_r1")
DATA = os.environ.get("LTSF_DIR", os.path.join(HERE, "datasets"))
REAL = ["NP", "PJM", "BE", "FR", "DE", "ETTh1", "ETTm1", "weather", "electricity"]
BETA = 0.05


# ---------------------------------------------------------------- data ------
def load_ltsf(name):
    """Return [T, ncov+1] array, target last. Hourly resolution."""
    fp = os.path.join(DATA, f"{name}.csv")
    rows = []
    with open(fp) as f:
        r = csv.reader(f); next(r)
        for row in r:
            rows.append([float(x) for x in row[1:]])
    a = np.asarray(rows, dtype=np.float32)
    if name == "ETTm1":
        a = a[-20000:]            # native 15-min resolution (H=24 steps = 6h);
                                  # hourly subsampling would duplicate ETTh1
    if name == "weather":
        a = a[::6]                # 10-min -> hourly
    y = a[:, -1:]                 # OT target
    Xall = a[:, :-1]
    ntr = int(len(a) * 0.6)
    if Xall.shape[1] > 4:         # pick top-4 |corr(train)| channels
        yc = y[:ntr, 0] - y[:ntr, 0].mean()
        c = np.array([np.abs(np.corrcoef(Xall[:ntr, j], y[:ntr, 0])[0, 1])
                      if Xall[:ntr, j].std() > 1e-8 else 0.0
                      for j in range(Xall.shape[1])])
        keep = np.argsort(-c)[:4]
        Xall = Xall[:, keep]
    return np.concatenate([Xall, y], axis=1)


def load_any(name):
    if name == "SynDeg":
        return load_dataset("SynDeg"), "gauss"
    if name in ("NP", "PJM", "BE", "FR", "DE"):
        a = load_dataset(name)
    else:
        a = load_ltsf(name)
    mu = a[:int(len(a) * .6)].mean(0)
    sd = a[:int(len(a) * .6)].std(0) + 1e-8
    return (a - mu) / sd, "nonparam"


# ------------------------------------------------------- pinball ridge ------
def fit_pinball(X, y, tau, lam=1e-2, iters=250, lr=None):
    """Linear quantile regression by full-batch subgradient descent."""
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    w = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ y)  # warm start at ridge mean
    if lr is None:
        lr = 0.5 / (np.abs(Xb).mean() ** 2 * Xb.shape[1] + 1e-8)
    for _ in range(iters):
        r = y - Xb @ w
        g = Xb.T @ (np.where(r > 0, -tau, 1 - tau)) / len(y) + lam * w / len(y)
        w -= lr * g
    return w


def pred_lin(w, X):
    return np.concatenate([X, np.ones((len(X), 1))], 1) @ w


# ------------------------------------------------------- new baselines ------
def run_cqr(Xh_tr, Xc_tr, Y_tr, ts_tr, calB, test, seed):
    """CQR: per-horizon linear pinball heads trained on realized covariates,
    CQR-score conformal correction on the noisy stream (matured feedback)."""
    (XhB, XcB_obs, YB, tsB, sdB) = calB
    (XhT, XcT_obs, YT, tsT, sdT) = test
    lo_B = np.zeros_like(YB); hi_B = np.zeros_like(YB)
    lo_T = np.zeros_like(YT); hi_T = np.zeros_like(YT)
    for h in range(H):
        Ftr = features(Xh_tr, Xc_tr, ts_tr, h)
        w_lo = fit_pinball(Ftr, Y_tr[:, h], ALPHA / 2)
        w_hi = fit_pinball(Ftr, Y_tr[:, h], 1 - ALPHA / 2)
        FB = features(XhB, XcB_obs, tsB, h); FT = features(XhT, XcT_obs, tsT, h)
        lo_B[:, h], hi_B[:, h] = pred_lin(w_lo, FB), pred_lin(w_hi, FB)
        lo_T[:, h], hi_T[:, h] = pred_lin(w_lo, FT), pred_lin(w_hi, FT)
    sc_B = np.maximum(lo_B - YB, YB - hi_B) / sdB
    sc_T = np.maximum(lo_T - YT, YT - hi_T) / sdT
    q = online_q(sc_B, sc_T, np.asarray(tsT), ALPHA)
    return lo_T - q * sdT, hi_T + q * sdT


def run_norcp(ws, calA_obs, calB, test):
    """Normalized conformal: difficulty g(x_obs) = ridge on log|residual| of
    noisy split A; scores |y - f(x_obs)| / g; matured online quantile."""
    (XhA, XcA_obs, YA, tsA) = calA_obs
    (XhB, XcB_obs, YB, tsB, sdB) = calB
    (XhT, XcT_obs, YT, tsT, sdT) = test
    pA = preds(ws, XhA, XcA_obs, tsA)
    gws, gmin = [], []
    for h in range(H):
        FA = features(XhA, XcA_obs, tsA, h)
        r = np.log(np.abs(YA[:, h] - pA[:, h]) + 1e-3)
        gws.append(CF.fit_ridge(FA, r))
    gA = np.stack([np.exp(CF.predict(gws[h], features(XhA, XcA_obs, tsA, h)))
                   for h in range(H)], 1)
    flo, fhi = np.quantile(gA, 0.1), 5 * np.quantile(gA, 0.99)
    def g(Xh, Xc, ts):
        out = np.stack([np.exp(CF.predict(gws[h], features(Xh, Xc, ts, h)))
                        for h in range(H)], 1)
        return np.clip(out, flo, fhi)          # floor 10%-q, cap 5x 99%-q (cal)
    pB = preds(ws, XhB, XcB_obs, tsB); pT = preds(ws, XhT, XcT_obs, tsT)
    gB, gT = g(XhB, XcB_obs, tsB), g(XhT, XcT_obs, tsT)
    sc_B = np.abs(YB - pB) / gB
    sc_T = np.abs(YT - pT) / gT
    q = online_q(sc_B, sc_T, np.asarray(tsT), ALPHA)
    return pT - q * gT, pT + q * gT


def run_spci(pB, YB, sdB, pT, YT, sdT, tsT, w_hist=48, refit=10**9):
    """SPCI-style: linear quantile regression of the normalized |residual| on
    recent matured residual statistics; refit every `refit` steps."""
    rB = np.abs(YB - pB) / sdB                 # [nB, H]
    rT = np.abs(YT - pT) / sdT
    sB, sT = rB.mean(1), rT.mean(1)            # per-window scalar difficulty
    shapeB = rB / (sB[:, None] + 1e-8)         # horizon profile
    prof = np.quantile(shapeB, 1 - ALPHA, axis=0)  # fixed profile from cal
    def feats(series, i):
        h = series[max(0, i - w_hist):i]
        if len(h) == 0:
            h = series[:1]
        return np.array([h.mean(), h[-1], h.max(), np.quantile(h, .9), np.log(h.std() + 1e-3)])
    stream = np.concatenate([sB, np.zeros(len(sT))])
    Xall, yall = [], []
    for i in range(1, len(sB)):
        Xall.append(feats(sB, i)); yall.append(sB[i])
    w = fit_pinball(np.asarray(Xall), np.asarray(yall), 1 - ALPHA)
    w0 = w.copy()
    qhat = np.zeros(len(sT))
    tsT = np.asarray(tsT)
    matured = 0
    for i in range(len(sT)):
        j = np.searchsorted(tsT + H, tsT[i], side="right")
        if j > matured + refit:
            matured = j
            for t in range(matured - refit, matured):
                Xall.append(feats(np.concatenate([sB, sT[:t]]), len(sB) + t))
                yall.append(sT[t])
            w = fit_pinball(np.asarray(Xall), np.asarray(yall), 1 - ALPHA)
        qhat[i] = pred_lin(w, feats(np.concatenate([sB, sT[:j]]), len(sB) + j)[None])[0]
    flo, fhi = np.quantile(sB, 0.25), 5 * np.quantile(sB, 0.95)
    qhat = np.clip(qhat, flo, fhi)
    # conformalize the SPCI difficulty scale: score = |res| / (qhat * profile),
    # matured online quantile (same protocol as every other method)
    gB = np.clip(pred_lin(w0, np.asarray([feats(sB, i) for i in range(1, len(sB))])), flo, fhi)
    gB_full = np.concatenate([[gB[0]], gB])[:, None] * prof[None, :]
    gT = qhat[:, None] * prof[None, :]
    q = online_q(rB / (gB_full + 1e-8), rT / (gT + 1e-8), np.asarray(tsT), ALPHA)
    half = q * gT * sdT
    return pT - half, pT + half


# ------------------------------------------------------------ one run -------
def run_once(data, seed, source, sigma_base=0.5, hetero=True, scale_b=0.3):
    rng = np.random.RandomState(seed)
    T = len(data)
    tr_end, cal_end = int(T * 0.6), int(T * 0.8)
    Xh_tr, Xc_tr, Y_tr, ts_tr = build_windows(data, 0, tr_end, 4000)
    Xh_ca, Xc_ca, Y_ca, ts_ca = build_windows(data, tr_end, cal_end, 1600)
    Xh_te, Xc_te, Y_te, ts_te = build_windows(data, cal_end, T, 1500)
    ncov = Xc_tr.shape[2]

    if source == "gauss":
        sig_ca = (make_hetero_sigma(len(Y_ca), sigma_base, rng) if hetero
                  else np.full(len(Y_ca), sigma_base))
        sig_te = (make_hetero_sigma(len(Y_te), sigma_base, rng) if hetero
                  else np.full(len(Y_te), sigma_base))
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

    p_ca_clean = preds(ws, Xh_ca, Xc_ca, ts_ca)
    p_caB_obs = preds(ws, Xh_ca[B], Xc_ca_obs[B], ts_ca[B])
    p_te_clean = preds(ws, Xh_te, Xc_te, ts_te)
    p_te_obs = preds(ws, Xh_te, Xc_te_obs, ts_te)

    res_A_clean = np.abs(Y_ca[A] - p_ca_clean[A]) / sd_ca[A]
    res_B_noisy = np.abs(Y_ca[B] - p_caB_obs) / sd_ca[B]
    sc_clean_all = np.abs(Y_ca - p_ca_clean) / sd_ca
    endo_w = float((2 * split_q(sc_clean_all, ALPHA) * sd_te).mean())

    q_clean_static = split_q(sc_clean_all, ALPHA)
    q_orc = online_q(sc_clean_all, np.abs(Y_te - p_te_clean) / sd_te, ts_te, ALPHA)
    q_rec = online_q(res_B_noisy, np.abs(Y_te - p_te_obs) / sd_te, ts_te, ALPHA)

    out, per_win = {}, {}
    def ev(name, lo, hi, endo=None):
        out[name] = evaluate(Y_te, lo, hi, mag_te, endo_width=endo)
        per_win[name] = winkler_pw(Y_te, lo, hi)

    ev("oracle", p_te_clean - q_orc * sd_te, p_te_clean + q_orc * sd_te)
    ev("naive", p_te_obs - q_clean_static * sd_te,
       p_te_obs + q_clean_static * sd_te, endo_w)
    ev("recal", p_te_obs - q_rec * sd_te, p_te_obs + q_rec * sd_te, endo_w)

    pB_d = preds(ws_drop, Xh_ca[B], Xc_ca_obs[B], ts_ca[B])
    pT_d = preds(ws_drop, Xh_te, Xc_te_obs, ts_te)
    q_d = online_q(np.abs(Y_ca[B] - pB_d) / sd_ca[B],
                   np.abs(Y_te - pT_d) / sd_te, ts_te, ALPHA)
    ev("exodrop", pT_d - q_d * sd_te, pT_d + q_d * sd_te)

    resB = (Y_ca[B] - p_caB_obs) / sd_ca[B]
    resT = (Y_te - p_te_obs) / sd_te
    for M in (ACI(ALPHA), NexCP(ALPHA)):
        c, q = M.run(resB, resT, list(ts_ca[B]), list(ts_te), H)
        ev(M.name, p_te_obs + (c - q) * sd_te, p_te_obs + (c + q) * sd_te)

    if source == "gauss":
        mcp = MondrianCP(ALPHA, n_bins=3)
        mcp.calibrate(res_B_noisy, -sig_ca[B])
        qm = mcp.qhat[np.digitize(-sig_te, mcp.edges)]
        ev("rc_cp", p_te_obs - qm * sd_te, p_te_obs + qm * sd_te)

    # ---- new strong baselines ----------------------------------------------
    lo, hi = run_cqr(Xh_tr, Xc_tr, Y_tr, ts_tr,
                     (Xh_ca[B], Xc_ca_obs[B], Y_ca[B], ts_ca[B], sd_ca[B]),
                     (Xh_te, Xc_te_obs, Y_te, ts_te, sd_te), seed)
    ev("cqr", lo, hi)
    lo, hi = run_norcp(ws, (Xh_ca[A], Xc_ca_obs[A], Y_ca[A], ts_ca[A]),
                       (Xh_ca[B], Xc_ca_obs[B], Y_ca[B], ts_ca[B], sd_ca[B]),
                       (Xh_te, Xc_te_obs, Y_te, ts_te, sd_te))
    ev("norcp", lo, hi)
    lo, hi = run_spci(p_caB_obs, Y_ca[B], sd_ca[B], p_te_obs, Y_te, sd_te, ts_te)
    ev("spci", lo, hi)

    # ---- CovProp-MC ---------------------------------------------------------
    rng2 = np.random.RandomState(seed + 1)
    res_signed = res_A_clean * np.sign(rng2.normal(size=res_A_clean.shape))
    if source == "gauss":
        samp_ca = lambda k: gauss_noise(Xc_ca_obs[B].shape, sig_ca[B][:, None, None], rng2)
        samp_te = lambda k: gauss_noise(Xc_te_obs.shape, sig_te[:, None, None], rng2)
    else:
        pool = eps_ca[A]
        def _draw(n):
            return pool[rng2.randint(0, len(pool), n)]
        samp_ca = lambda k: _draw(len(Xc_ca_obs[B]))
        samp_te = lambda k: _draw(len(Xc_te_obs))
    lo, hi, lam = covprop_mc(ws, (Xh_ca[B], Xc_ca_obs[B], Y_ca[B], ts_ca[B], sd_ca[B]),
                             (Xh_te, Xc_te_obs, ts_te, sd_te, Y_te),
                             samp_ca, samp_te, res_signed, rng2)
    ev("covprop_mc", lo, hi, endo_w)
    half_cp = (hi - lo) / 2      # for the teaser figure

    # ---- CovProp-Nested (beta = 0.05 only) ----------------------------------
    qab = split_q(res_A_clean, ALPHA - BETA)
    if source == "gauss":
        rad = (norm.ppf(1 - BETA / 2) * sig_te)[:, None, None]
    else:
        rad = np.quantile(np.abs(eps_ca[A]).reshape(-1, ncov), 1 - BETA,
                          axis=0)[None, None, :]
        rad = np.broadcast_to(rad, Xc_te_obs.shape)
    lo, hi = covprop_nested(ws, Xh_te, Xc_te_obs, ts_te, rad, qab * sd_te, rng2)
    ev(f"covprop_nest_b{BETA}", lo, hi, float((2 * qab * sd_te).mean()))

    teaser = {"mag": mag_te,
              "w_naive": (2 * q_clean_static * sd_te).mean(1),
              "w_recal": (2 * q_rec * sd_te).mean(1),
              "w_covprop": (2 * half_cp).mean(1)}
    return out, per_win, teaser


def winkler_pw(Y, lo, hi, alpha=ALPHA):
    w = (hi - lo) + (2 / alpha) * np.maximum(lo - Y, 0) + \
        (2 / alpha) * np.maximum(Y - hi, 0)
    return w.mean(1)                                    # [m] per-window


# --------------------------------------------------------------- stats ------
def significance(per_win_by_seed, seed_metrics, ref="covprop_mc"):
    """Paired tests of ref vs every other method.
    (i) Wilcoxon signed-rank on per-window Winkler, paired per (seed, window);
    (ii) paired t-test across per-seed mean Winkler. Holm within dataset."""
    methods = [m for m in per_win_by_seed[0] if m not in (ref, "oracle")]
    res = {}
    raw_p = []
    for m in methods:
        d_ref = np.concatenate([pw[ref] for pw in per_win_by_seed])
        d_m = np.concatenate([pw[m] for pw in per_win_by_seed])
        try:
            wstat, p_w = sps.wilcoxon(d_ref, d_m)
        except ValueError:
            p_w = 1.0
        a = np.array([sm[ref] for sm in seed_metrics])
        b = np.array([sm[m] for sm in seed_metrics])
        t, p_t = sps.ttest_rel(a, b)
        res[m] = {"winkler_ref": float(d_ref.mean()), "winkler_m": float(d_m.mean()),
                  "p_wilcoxon_perwindow": float(p_w), "p_ttest_seeds": float(p_t),
                  "ref_better": bool(d_ref.mean() < d_m.mean())}
        raw_p.append(p_w)
    order = np.argsort(raw_p)
    k = len(raw_p)
    holm = [None] * k
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (k - rank) * raw_p[idx])
        prev = max(prev, adj)
        holm[idx] = prev
    for i, m in enumerate(methods):
        res[m]["p_holm"] = float(holm[i])
        res[m]["significant_holm05"] = bool(holm[i] < 0.05)
    return res


# ---------------------------------------------------------------- main ------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--datasets", nargs="+", default=["SynDeg"] + REAL)
    ap.add_argument("--exp", default="all", choices=["all", "main", "sensitivity", "figures"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    if args.exp in ("all", "main"):
        for ds in args.datasets:
            data, source = load_any(ds)
            runs, pws, teasers = [], [], []
            for s in args.seeds:
                o, pw, tz = run_once(data, s, source)
                runs.append(o); pws.append(pw); teasers.append(tz)
                print(f"  [{ds} seed {s}] covprop wink="
                      f"{o['covprop_mc']['winkler']:.3f} naive-worst="
                      f"{min(o['naive']['cov_by_noise_tercile']):.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            rep = {"dataset": ds, "source": source, "seeds": args.seeds,
                   "aggregate": CF.agg(runs), "runs": runs}
            seed_metrics = [{m: r[m]["winkler"] for m in r} for r in runs]
            rep["significance_winkler_vs_covprop_mc"] = significance(pws, seed_metrics)
            with open(f"{OUT}/main_{ds}.json", "w") as f:
                json.dump(rep, f, indent=1)
            np.savez_compressed(f"{OUT}/perwin_{ds}.npz",
                                **{f"s{si}_{m}": pw[m]
                                   for si, pw in enumerate(pws) for m in pw})
            if ds == "SynDeg":
                np.savez_compressed(f"{OUT}/teaser_syndeg.npz", **teasers[0])
            print(f"== {ds} done [{time.time()-t0:.0f}s]", flush=True)

    if args.exp in ("all", "sensitivity"):
        data, _ = load_any("SynDeg")
        sens = {}
        for sigma in (0.1, 0.25, 0.5, 1.0):
            runs = [CF.run_once(data, s, "gauss", sigma_base=sigma,
                                hetero=False, methods_all=False)
                    for s in args.seeds]
            sens[str(sigma)] = CF.agg(runs)
            print(f"[sens sigma={sigma}] ({time.time()-t0:.0f}s)", flush=True)
        with open(f"{OUT}/sensitivity_syndeg.json", "w") as f:
            json.dump(sens, f, indent=1)

    if args.exp in ("all", "figures"):
        make_figures()

    write_csv()
    print(f"\ndone in {time.time()-t0:.0f}s -> {OUT}/", flush=True)


def write_csv():
    rows = []
    for ds in ["SynDeg"] + REAL:
        fp = f"{OUT}/main_{ds}.json"
        if not os.path.exists(fp):
            continue
        rep = json.load(open(fp))
        sig = rep.get("significance_winkler_vs_covprop_mc", {})
        for m, v in rep["aggregate"].items():
            s = sig.get(m, {})
            rows.append([rep["dataset"], m,
                         f"{v['coverage']['mean']:.4f}", f"{v['coverage']['std']:.4f}",
                         f"{v['worst_tercile_cov']['mean']:.4f}",
                         f"{v['worst_tercile_cov']['std']:.4f}",
                         f"{v['width']['mean']:.4f}", f"{v['winkler']['mean']:.4f}",
                         f"{v['winkler']['std']:.4f}",
                         f"{v.get('width_endo', ''):}", f"{v.get('width_prop', ''):}",
                         f"{s.get('p_wilcoxon_perwindow', ''):}",
                         f"{s.get('p_holm', ''):}",
                         f"{s.get('p_ttest_seeds', ''):}",
                         s.get("ref_better", "")])
    with open(f"{OUT}/summary_r1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "cov_mean", "cov_std",
                    "worst_terc_mean", "worst_terc_std", "width",
                    "winkler_mean", "winkler_std", "width_endo", "width_prop",
                    "p_wilcoxon", "p_holm", "p_ttest_seeds", "covprop_better"])
        w.writerows(rows)


# ------------------------------------------------------------- figures ------
def make_figures():
    C = {"oracle": "gray", "naive": "#4878d0", "recal": "#ee854a",
         "exodrop": "#dc7ec0", "aci": "#6acc64", "nexcp": "#d65f5f",
         "rc_cp": "#8c613c", "cqr": "#e8c93e", "norcp": "#4dbdb0",
         "spci": "#7a7a2e", "covprop_mc": "#956cb4",
         "covprop_nest_b0.05": "#82c6e2"}
    # ---- fig 0: teaser ------------------------------------------------------
    tz = np.load(f"{OUT}/teaser_syndeg.npz")
    sens = json.load(open(f"{OUT}/sensitivity_syndeg.json"))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    ax = axes[0]
    mag = tz["mag"]; order = np.argsort(mag)
    nb = 20
    bins = np.array_split(order, nb)
    xs = [mag[b].mean() for b in bins]
    for key, lab in (("w_naive", "naive (uniform, collapses)"),
                     ("w_recal", "recalibration (uniform, inflated)"),
                     ("w_covprop", "CovProp-MC (adaptive)")):
        ax.plot(xs, [tz[key][b].mean() for b in bins], marker="o", ms=3,
                label=lab, color={"w_naive": C["naive"], "w_recal": C["recal"],
                                  "w_covprop": C["covprop_mc"]}[key])
    ax.set_xlabel("upstream covariate error magnitude $|\\eta|$ (binned)")
    ax.set_ylabel("mean interval width")
    ax.set_title("Width should follow input uncertainty")
    ax.legend(fontsize=7)
    ax = axes[1]
    sigmas = sorted(float(s) for s in sens)
    for m in ("naive", "recal", "covprop_mc"):
        ax.plot(sigmas, [sens[str(s)][m]["coverage"]["mean"] for s in sigmas],
                marker="o", label=m, color=C.get(m))
    ax.axhline(0.9, color="k", ls="--", lw=1)
    ax.set_xlabel("upstream forecast error scale $\\sigma$")
    ax.set_ylabel("marginal coverage")
    ax.set_title("Naive interval collapses; CovProp holds")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig0_teaser.png", dpi=170)
    plt.close(fig)
    # ---- fig 1: collapse curve (5 seeds) ------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in ("oracle", "naive", "recal", "covprop_mc", "covprop_nest_b0.02"):
        if m in sens[str(sigmas[0])]:
            ax.plot(sigmas, [sens[str(s)][m]["coverage"]["mean"] for s in sigmas],
                    marker="o", label=m, color=C.get(m, "#82c6e2"))
    ax.axhline(0.9, color="k", ls="--", lw=1)
    ax.set_xlabel("upstream forecast error sigma")
    ax.set_ylabel("marginal coverage")
    ax.set_title("Coverage collapse and recovery (SynDeg, 5 seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_collapse_curve.png", dpi=170)
    plt.close(fig)
    # ---- fig 2: conditional coverage (SynDeg) -------------------------------
    rep = json.load(open(f"{OUT}/main_SynDeg.json"))["aggregate"]
    ms = [m for m in ("naive", "recal", "exodrop", "aci", "nexcp", "cqr",
                      "norcp", "spci", "rc_cp", "covprop_mc") if m in rep]
    x = np.arange(3); wd = 0.8 / len(ms)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for i, m in enumerate(ms):
        ax.bar(x + (i - (len(ms) - 1) / 2) * wd, rep[m]["cov_by_noise_tercile"],
               wd, label=m, color=C.get(m), edgecolor="black", linewidth=0.3)
    ax.axhline(0.9, color="k", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(["low |eps|", "mid |eps|", "high |eps|"])
    ax.set_ylabel("conditional coverage"); ax.set_ylim(0.35, 1.02)
    ax.set_title("Conditional coverage by covariate-error tercile (SynDeg-hetero, 5 seeds)")
    ax.legend(fontsize=7, ncol=5)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_conditional_coverage.png", dpi=170)
    plt.close(fig)
    # ---- fig 3: width decomposition across datasets --------------------------
    tags, endo, prop = [], [], []
    for ds in ["SynDeg"] + REAL:
        fp = f"{OUT}/main_{ds}.json"
        if os.path.exists(fp):
            v = json.load(open(fp))["aggregate"]["covprop_mc"]
            if "width_endo" in v:
                tags.append(ds); endo.append(v["width_endo"]); prop.append(v["width_prop"])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(tags))
    e, p = np.array(endo), np.array(prop)
    share = p / (e + p) * 100
    ax.bar(x, e / (e + p), 0.55, label="endogenous residual", color="#4878d0")
    ax.bar(x, p / (e + p), 0.55, bottom=e / (e + p), label="covariate propagation", color="#ee854a")
    for i in range(len(tags)):
        ax.text(x[i], 1.01, f"{share[i]:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=20)
    ax.set_ylabel("width share")
    ax.set_title("CovProp-MC width decomposition (propagation share on top)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_width_decomposition.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
