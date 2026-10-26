"""CovProp R2 experiments (reviewer round 2).

M1  Replace the per-window Wilcoxon (invalid under overlapping-window serial
    correlation) by a seed-clustered moving-block bootstrap on per-window
    Winkler differences (block length 48 > H = 24, B = 4000), Holm-corrected
    within dataset; recompute all tab:real stars from the archived R1
    per-window scores.  Seed-level paired t-tests kept alongside.
S1  Real upstream forecast--realization error streams: a genuine upstream
    forecaster (per-channel, per-horizon ridge on 24 covariate lags +
    hour-of-day, trained on the train split only) issues day-ahead covariate
    forecasts; eta = its real out-of-sample errors.  No synthetic noise.
    Datasets: NP, FR (markets), ETTh1, weather.
    (Note: epftoolbox CSVs ship the published day-ahead exogenous forecasts
    but NOT the realizations of load/wind, so on the markets the "realized"
    covariate is the published series the backbone consumes and the upstream
    task is forecasting it 24 h ahead -- a real operational upstream model
    with real errors; disclosed in the paper.)
S2  Nonlinear backbone: per-horizon 1-hidden-layer MLP (tanh, 32 units,
    Adam 400 steps) replacing the ridge backbone; SynDeg + NP + ETTh1.
S3  ACI x CovProp composition: ACI level adaptation run on the CovProp-MC
    normalized scores (Y - mid)/half instead of the static matured quantile;
    added to every R2 run and to replay-mode PJM / weather.
N1  K (MC samples) sensitivity on SynDeg.
N3  Per-hour conditional coverage on the market real-stream runs.

Outputs: results_r2/  (CPU-only).
"""
import argparse, csv, json, os, sys, time
import numpy as np
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = HERE
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "core"))
sys.path.insert(0, HERE)

import covprop_full as CF                                        # noqa: E402
from covprop_full import (ALPHA, H, L_HIST, K_MC, build_windows, features,
                          train_base, preds, split_q, online_q, gauss_noise,
                          make_hetero_sigma, make_ar_error, covprop_nested,
                          evaluate, load_dataset)                # noqa: E402
from baselines import ACI, NexCP, winkler                        # noqa: E402
from rc_cp import MondrianCP                                     # noqa: E402
from scipy.stats import norm                                     # noqa: E402
import covprop_r1 as R1                                          # noqa: E402
from covprop_r1 import (load_any, run_cqr, run_norcp, run_spci,
                        winkler_pw, BETA)                        # noqa: E402

OUT = os.path.join(HERE, "results_r2")
R1OUT = os.environ.get("R1_RESULTS", os.path.join(HERE, "..", "results_r1"))
REAL_STREAM = ["NP", "FR", "ETTh1", "weather"]
MLP_DS = ["SynDeg", "NP", "ETTh1"]
COMPOSE_DS = ["PJM", "weather"]


# ------------------------------------------------- M1: block bootstrap ------
def block_boot_p(diffs_by_seed, L=48, B=4000, rng=None):
    """Two-sided p for H0: mean per-window Winkler difference = 0.
    Moving-block bootstrap within each seed (blocks of length L preserve the
    overlap-induced serial correlation), seeds resampled independently."""
    rng = rng or np.random.RandomState(0)
    obs = np.mean([d.mean() for d in diffs_by_seed])
    cent = [d - d.mean() for d in diffs_by_seed]          # null-centered
    starts = [len(d) - L for d in cent]
    boots = np.zeros(B)
    for b in range(B):
        tot, cnt = 0.0, 0
        for d, smax in zip(cent, starts):
            if smax <= 0:
                tot += d.sum(); cnt += len(d); continue
            nblk = int(np.ceil(len(d) / L))
            idx = rng.randint(0, smax + 1, nblk)
            take = np.concatenate([d[i:i + L] for i in idx])[:len(d)]
            tot += take.sum(); cnt += len(take)
        boots[b] = tot / cnt
    p = (np.sum(np.abs(boots) >= abs(obs)) + 1) / (B + 1)
    return float(p), float(obs)


def holm(pvals):
    order = np.argsort(pvals)
    k = len(pvals)
    out = [None] * k
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (k - rank) * pvals[idx])
        prev = max(prev, adj)
        out[idx] = prev
    return out


def significance_r2(per_win_by_seed, seed_metrics, ref="covprop_mc"):
    methods = [m for m in per_win_by_seed[0] if m not in (ref, "oracle")]
    res, raw = {}, []
    for m in methods:
        diffs = [pw[ref] - pw[m] for pw in per_win_by_seed]
        p_bb, obs = block_boot_p(diffs, rng=np.random.RandomState(hash(m) % 2**31))
        a = np.array([sm[ref] for sm in seed_metrics])
        b = np.array([sm[m] for sm in seed_metrics])
        _, p_t = sps.ttest_rel(a, b)
        res[m] = {"mean_diff_ref_minus_m": obs, "p_blockboot": p_bb,
                  "p_ttest_seeds": float(p_t),
                  "ref_better": bool(obs < 0)}
        raw.append(p_bb)
    hp = holm(raw)
    for i, m in enumerate(methods):
        res[m]["p_holm_blockboot"] = float(hp[i])
        res[m]["significant_holm05"] = bool(hp[i] < 0.05)
    return res


def restat_r1():
    """Recompute significance for all 10 R1 datasets from archived per-window
    scores; write per-dataset JSON + LaTeX star map."""
    out = {}
    for ds in ["SynDeg"] + R1.REAL:
        npz = np.load(f"{R1OUT}/perwin_{ds}.npz")
        rep = json.load(open(f"{R1OUT}/main_{ds}.json"))
        seeds = rep["seeds"]
        methods = sorted({k.split("_", 1)[1] for k in npz.files})
        pws = [{m: npz[f"s{si}_{m}"] for m in methods if f"s{si}_{m}" in npz}
               for si in range(len(seeds))]
        seed_metrics = [{m: r[m]["winkler"] for m in r} for r in rep["runs"]]
        out[ds] = significance_r2(pws, seed_metrics)
        print(f"[restat {ds}] "
              + " ".join(f"{m}:{'*' if v['significant_holm05'] else '-'}"
                         f"{'+' if v['ref_better'] else '-'}"
                         for m, v in out[ds].items()), flush=True)
    with open(f"{OUT}/significance_blockboot_r1data.json", "w") as f:
        json.dump(out, f, indent=1)
    return out


# --------------------------------------------- S1: real upstream errors -----
def upstream_forecast(a, ts, tr_end, n_tr=4000):
    """Per-channel per-horizon ridge upstream forecaster: predicts covariate
    channel j at issue-time+h from its own 24 lags + hour-of-day, trained on
    the train split only.  Returns xhat [len(ts), H, ncov]."""
    cov = a[:, :-1]
    ncov = cov.shape[1]
    t_tr = np.arange(L_HIST, tr_end - H)
    if len(t_tr) > n_tr:
        t_tr = t_tr[np.linspace(0, len(t_tr) - 1, n_tr).astype(int)]
    xhat = np.zeros((len(ts), H, ncov))
    for j in range(ncov):
        lag_tr = np.stack([cov[t - L_HIST:t, j] for t in t_tr])
        lag_te = np.stack([cov[t - L_HIST:t, j] for t in ts])
        for h in range(H):
            F = np.concatenate([lag_tr,
                                np.sin(2 * np.pi * (t_tr + h) / 24)[:, None],
                                np.cos(2 * np.pi * (t_tr + h) / 24)[:, None]], 1)
            w = CF.fit_ridge(F, cov[t_tr + h, j])
            Fte = np.concatenate([lag_te,
                                  np.sin(2 * np.pi * (ts + h) / 24)[:, None],
                                  np.cos(2 * np.pi * (ts + h) / 24)[:, None]], 1)
            xhat[:, h, j] = CF.predict(w, Fte)
    return xhat


# ------------------------------------------------- S2: MLP backbone ---------
def train_mlp(Xh, Xc, Y, ts, seed, hid=32, iters=400, lr=1e-2, drop=0.0):
    """Per-horizon 1-hidden-layer tanh MLP trained with Adam on MSE."""
    rng = np.random.RandomState(seed + 101)
    ws = []
    for h in range(H):
        Xc_h = Xc
        if drop > 0:
            keep = (rng.random((len(Xc), 1, Xc.shape[2])) > drop)
            Xc_h = Xc * keep
        F = features(Xh, Xc_h, ts, h)
        y = Y[:, h]
        d = F.shape[1]
        W1 = rng.normal(0, 1 / np.sqrt(d), (d, hid)); b1 = np.zeros(hid)
        W2 = rng.normal(0, 1 / np.sqrt(hid), (hid, 1)); b2 = np.zeros(1)
        params = [W1, b1, W2, b2]
        mom = [np.zeros_like(p) for p in params]
        vel = [np.zeros_like(p) for p in params]
        for it in range(1, iters + 1):
            Z = np.tanh(F @ W1 + b1)
            pred = (Z @ W2 + b2).ravel()
            e = (pred - y) / len(y)
            gW2 = Z.T @ e[:, None]; gb2 = np.array([e.sum()])
            dZ = e[:, None] @ W2.T * (1 - Z ** 2)
            gW1 = F.T @ dZ; gb1 = dZ.sum(0)
            for p, g, mo, ve in zip(params, [gW1, gb1, gW2, gb2], mom, vel):
                mo *= 0.9; mo += 0.1 * g
                ve *= 0.999; ve += 0.001 * g * g
                p -= lr * (mo / (1 - 0.9 ** it)) / (np.sqrt(ve / (1 - 0.999 ** it)) + 1e-8)
        ws.append(("mlp", W1, b1, W2, b2))
    return ws


def preds_any(ws, Xh, Xc, ts):
    if isinstance(ws[0], tuple) and ws[0][0] == "mlp":
        out = []
        for h in range(H):
            _, W1, b1, W2, b2 = ws[h]
            F = features(Xh, Xc, ts, h)
            out.append((np.tanh(F @ W1 + b1) @ W2 + b2).ravel())
        return np.stack(out, 1)
    return preds(ws, Xh, Xc, ts)


def mc_quantiles_any(ws, Xh, Xc_obs, ts, sampler, res_pool, sd, rng, k_mc=K_MC):
    S = []
    for k in range(k_mc):
        pk = preds_any(ws, Xh, Xc_obs + sampler(k), ts)
        r = res_pool[rng.randint(0, len(res_pool), len(pk))]
        S.append(pk + r * sd)
    S = np.stack(S, 0)
    return (np.quantile(S, ALPHA / 2, axis=0),
            np.quantile(S, 1 - ALPHA / 2, axis=0))


# ---------------------------------------------------------- one R2 run ------
def run_once_r2(data, seed, source, mode="replay", backbone="linear",
                sigma_base=0.5, scale_b=0.3, k_mc=K_MC, record_hours=False):
    """mode: 'replay' (R1 protocol) or 'real' (upstream forecaster errors).
    backbone: 'linear' | 'mlp'.  Adds the aci_covprop composition."""
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

    trainer = train_base if backbone == "linear" else \
        (lambda *a, **kw: train_mlp(*a, seed=seed, **kw))
    ws = trainer(Xh_tr, Xc_tr, Y_tr, ts_tr)
    ws_drop = trainer(Xh_tr, Xc_tr, Y_tr, ts_tr, drop=0.3,
                      **({"rng": np.random.RandomState(seed + 7)}
                         if backbone == "linear" else {}))

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
    q_orc = online_q(sc_clean_all, np.abs(Y_te - p_te_clean) / sd_te, ts_te, ALPHA)
    q_rec = online_q(res_B_noisy, np.abs(Y_te - p_te_obs) / sd_te, ts_te, ALPHA)

    out, per_win, hourcov = {}, {}, {}
    hours = (ts_te[:, None] + np.arange(H)[None, :]) % 24
    def ev(name, lo, hi, endo=None):
        out[name] = evaluate(Y_te, lo, hi, mag_te, endo_width=endo)
        per_win[name] = winkler_pw(Y_te, lo, hi)
        if record_hours:
            inside = ((Y_te >= lo) & (Y_te <= hi)).astype(float)
            hourcov[name] = [float(inside[hours == hh].mean()) for hh in range(24)]

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
                       (Xh_te, Xc_te_obs, Y_te, ts_te, sd_te)) \
        if backbone == "linear" else run_norcp_mlp(
            ws, (Xh_ca[A], Xc_ca_obs[A], Y_ca[A], ts_ca[A]),
            (Xh_ca[B], Xc_ca_obs[B], Y_ca[B], ts_ca[B], sd_ca[B]),
            (Xh_te, Xc_te_obs, Y_te, ts_te, sd_te))
    ev("norcp", lo, hi)
    lo, hi = run_spci(p_caB_obs, Y_ca[B], sd_ca[B], p_te_obs, Y_te, sd_te, ts_te)
    ev("spci", lo, hi)

    # ---- CovProp-MC (+ ACI composition on the same MC quantiles) -----------
    rng2 = np.random.RandomState(seed + 1)
    res_signed = res_A_clean * np.sign(rng2.normal(size=res_A_clean.shape))
    if mode != "real" and source == "gauss":
        samp_ca = lambda k: gauss_noise(Xc_ca_obs[B].shape, sig_ca[B][:, None, None], rng2)
        samp_te = lambda k: gauss_noise(Xc_te_obs.shape, sig_te[:, None, None], rng2)
        pool = None
    else:
        pool = eps_ca[A]
        samp_ca = lambda k: pool[rng2.randint(0, len(pool), len(Xc_ca_obs[B]))]
        samp_te = lambda k: pool[rng2.randint(0, len(pool), len(Xc_te_obs))]
    lo_B, hi_B = mc_quantiles_any(ws, Xh_ca[B], Xc_ca_obs[B], ts_ca[B],
                                  samp_ca, res_signed, sd_ca[B], rng2, k_mc)
    mid_B, half_B = (lo_B + hi_B) / 2, np.maximum((hi_B - lo_B) / 2, 1e-8)
    lo_T, hi_T = mc_quantiles_any(ws, Xh_te, Xc_te_obs, ts_te,
                                  samp_te, res_signed, sd_te, rng2, k_mc)
    mid_T, half_T = (lo_T + hi_T) / 2, np.maximum((hi_T - lo_T) / 2, 1e-8)
    sc_B = np.abs(Y_ca[B] - mid_B) / half_B
    sc_T = np.abs(Y_te - mid_T) / half_T
    lam = online_q(sc_B, sc_T, np.asarray(ts_te), ALPHA)
    ev("covprop_mc", mid_T - lam * half_T, mid_T + lam * half_T, endo_w)

    # S3: ACI on CovProp normalized scores (composition)
    rB_cp = (Y_ca[B] - mid_B) / half_B
    rT_cp = (Y_te - mid_T) / half_T
    c, q = ACI(ALPHA).run(rB_cp, rT_cp, list(ts_ca[B]), list(ts_te), H)
    ev("aci_covprop", mid_T + (c - q) * half_T, mid_T + (c + q) * half_T, endo_w)

    # Nested (beta = 0.05)
    qab = split_q(res_A_clean, ALPHA - BETA)
    if mode != "real" and source == "gauss":
        rad = (norm.ppf(1 - BETA / 2) * sig_te)[:, None, None]
        rad = np.broadcast_to(rad, Xc_te_obs.shape)
    else:
        rad = np.quantile(np.abs(eps_ca[A]).reshape(-1, ncov), 1 - BETA,
                          axis=0)[None, None, :]
        rad = np.broadcast_to(rad, Xc_te_obs.shape)
    if backbone == "linear":
        lo, hi = covprop_nested(ws, Xh_te, Xc_te_obs, ts_te, rad,
                                qab * sd_te, rng2)
        ev(f"covprop_nest_b{BETA}", lo, hi, float((2 * qab * sd_te).mean()))
    return out, per_win, hourcov


def run_norcp_mlp(ws, calA_obs, calB, test):
    """NorCP with the MLP backbone: same as run_norcp but predictions via
    preds_any (difficulty model stays ridge, as disclosed)."""
    (XhA, XcA_obs, YA, tsA) = calA_obs
    (XhB, XcB_obs, YB, tsB, sdB) = calB
    (XhT, XcT_obs, YT, tsT, sdT) = test
    pA = preds_any(ws, XhA, XcA_obs, tsA)
    gws = []
    for h in range(H):
        FA = features(XhA, XcA_obs, tsA, h)
        gws.append(CF.fit_ridge(FA, np.log(np.abs(YA[:, h] - pA[:, h]) + 1e-3)))
    gA = np.stack([np.exp(CF.predict(gws[h], features(XhA, XcA_obs, tsA, h)))
                   for h in range(H)], 1)
    flo, fhi = np.quantile(gA, 0.1), 5 * np.quantile(gA, 0.99)
    def g(Xh, Xc, ts):
        o = np.stack([np.exp(CF.predict(gws[h], features(Xh, Xc, ts, h)))
                      for h in range(H)], 1)
        return np.clip(o, flo, fhi)
    pB = preds_any(ws, XhB, XcB_obs, tsB); pT = preds_any(ws, XhT, XcT_obs, tsT)
    gB, gT = g(XhB, XcB_obs, tsB), g(XhT, XcT_obs, tsT)
    q = online_q(np.abs(YB - pB) / gB, np.abs(YT - pT) / gT, np.asarray(tsT), ALPHA)
    return pT - q * gT, pT + q * gT


# ------------------------------------------------------------ drivers -------
def run_block(tag, datasets, mode, backbone, seeds, record_hours=False):
    for ds in datasets:
        data, source = load_any(ds)
        runs, pws, hcs = [], [], []
        t0 = time.time()
        for s in seeds:
            o, pw, hc = run_once_r2(data, s, source, mode=mode,
                                    backbone=backbone, record_hours=record_hours)
            runs.append(o); pws.append(pw); hcs.append(hc)
            print(f"  [{tag} {ds} seed {s}] covprop wink="
                  f"{o['covprop_mc']['winkler']:.3f} "
                  f"acixcp wink={o['aci_covprop']['winkler']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        rep = {"dataset": ds, "mode": mode, "backbone": backbone,
               "seeds": list(seeds), "aggregate": CF.agg(runs), "runs": runs}
        seed_metrics = [{m: r[m]["winkler"] for m in r} for r in runs]
        rep["significance_winkler_vs_covprop_mc"] = significance_r2(pws, seed_metrics)
        if record_hours and hcs[0]:
            rep["cov_by_hour"] = {m: [float(np.mean([h[m][hh] for h in hcs]))
                                      for hh in range(24)] for m in hcs[0]}
        with open(f"{OUT}/{tag}_{ds}.json", "w") as f:
            json.dump(rep, f, indent=1)
        np.savez_compressed(f"{OUT}/perwin_{tag}_{ds}.npz",
                            **{f"s{si}_{m}": pw[m]
                               for si, pw in enumerate(pws) for m in pw})
        print(f"== {tag} {ds} done", flush=True)


def k_sensitivity(seeds=(0, 1, 2)):
    data, source = load_any("SynDeg")
    res = {}
    for k in (8, 16, 32, 64, 128):
        runs = []
        for s in seeds:
            o, _, _ = run_once_r2(data, s, source, mode="replay",
                                  backbone="linear", k_mc=k)
            runs.append({m: o[m] for m in ("covprop_mc",)})
        res[str(k)] = CF.agg(runs)
        print(f"[K={k}] wink={res[str(k)]['covprop_mc']['winkler']['mean']:.3f} "
              f"cov={res[str(k)]['covprop_mc']['coverage']['mean']:.3f}", flush=True)
    with open(f"{OUT}/k_sensitivity_syndeg.json", "w") as f:
        json.dump(res, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--exp", default="all",
                    choices=["all", "restat", "real", "mlp", "compose", "ksens"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    if args.exp in ("all", "restat"):
        restat_r1()
    if args.exp in ("all", "real"):
        run_block("real", REAL_STREAM, "real", "linear", args.seeds,
                  record_hours=True)
    if args.exp in ("all", "mlp"):
        run_block("mlp", MLP_DS, "replay", "mlp", args.seeds)
    if args.exp in ("all", "compose"):
        run_block("compose", COMPOSE_DS, "replay", "linear", args.seeds)
    if args.exp in ("all", "ksens"):
        k_sensitivity()
    print(f"\ndone in {time.time()-t0:.0f}s -> {OUT}/", flush=True)


if __name__ == "__main__":
    main()
