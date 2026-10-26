"""CovProp full experiments — propagating future-covariate uncertainty into
conformal time-series forecasts (T2, targeting ICDE 2027 R2 / ICML 2027).

Reuses the T1 harness (data.py / syndeg.py / baselines.py / rc_cp.py from
t1_rc_cp) via sys.path; nothing in t1_rc_cp is modified.

Setting
-------
Base forecaster f trained with TRUE future covariates. At deployment the
covariates are an upstream forecast x_obs = x_true + eps_t. Two uncertainty
sources for the upstream error:
  (a) "known-sigma": upstream self-reports a per-window Gaussian scale
      sigma_t (heteroscedastic on SynDeg-hetero; fixed on the collapse grid);
  (b) "nonparam": nothing self-reported; CovProp estimates the error
      distribution nonparametrically from a historical window of
      forecast-vs-actual residuals (conformalized; used on NP/PJM).

Methods (alpha = 0.1)
---------------------
oracle          split-CP with true covariates (undeployable upper bound)
naive           split-CP calibrated clean, deployed on x_obs
recal           split-CP recalibrated on the noisy stream (blind widening)
exodrop         covariate-dropout-trained ridge (point-robust) + noisy recal
aci / nexcp     time-adaptive CP on the noisy stream (covariate-blind)
rc_cp           T1 Mondrian-RC conditioning on the reliability signal
                (-sigma_t, source (a) only — needs an interpretable score)
covprop_mc      MC convolution {f(x'_k) + r_k} of the upstream distribution
                with calibration residuals, quantiles conformalized by a
                second-stage scale wrapper lambda (finite-sample valid)
covprop_nest    nested conformal: stage-1 (1-beta) covariate set from
                self-reported sigma or historical residual quantiles;
                stage-2 union of y-intervals at level 1-alpha+beta
                (Bonferroni-valid); beta in {0.01, 0.02, 0.05}

Metrics: marginal coverage, conditional coverage by |eps| terciles, mean
width, Winkler, and the width decomposition endo + propagation.

Outputs: results_covprop/*.json/.csv/.png.  CPU-only.
Usage: bash run_covprop_full.sh   (or python covprop_full.py --exp all)
"""
import argparse, csv, json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T1 = os.environ.get("T1_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))
sys.path.insert(0, T1)
from data import load_market  # noqa: E402
from syndeg import generate  # noqa: E402
from baselines import ACI, NexCP, winkler  # noqa: E402
from rc_cp import MondrianCP  # noqa: E402

ALPHA = 0.10
BETAS = (0.01, 0.02, 0.05)
L_HIST = 24
H = 24
K_MC = 64
RIDGE = 1e-2
OUT = "results_covprop"


# ----------------------------- base forecaster ------------------------------
def build_windows(a, start, end, n_max):
    cov, y = a[:, :-1], a[:, -1]
    lo = max(start, L_HIST)
    ts = np.arange(lo, end - H)
    if len(ts) > n_max:
        ts = ts[np.linspace(0, len(ts) - 1, n_max).astype(int)]
    Xh = np.stack([y[t - L_HIST:t] for t in ts])
    Xc = np.stack([cov[t:t + H] for t in ts])
    Y = np.stack([y[t:t + H] for t in ts])
    return Xh, Xc, Y, ts


def features(Xh, Xc, ts, h):
    tt = ts + h
    cal = np.stack([np.sin(2 * np.pi * tt / 24), np.cos(2 * np.pi * tt / 24)], 1)
    return np.concatenate([Xh, Xc[:, h, :], cal], axis=1)


def fit_ridge(X, y):
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
    return np.linalg.solve(Xb.T @ Xb + RIDGE * np.eye(Xb.shape[1]), Xb.T @ y)


def predict(w, X):
    return np.concatenate([X, np.ones((len(X), 1))], 1) @ w


def train_base(Xh, Xc, Y, ts, drop=0.0, rng=None):
    ws = []
    for h in range(H):
        Xc_h = Xc
        if drop > 0:
            keep = (rng.random((len(Xc), 1, Xc.shape[2])) > drop)
            Xc_h = Xc * keep
        ws.append(fit_ridge(features(Xh, Xc_h, ts, h), Y[:, h]))
    return ws


def preds(ws, Xh, Xc, ts):
    return np.stack([predict(ws[h], features(Xh, Xc, ts, h)) for h in range(H)], 1)


def split_q(res, alpha):
    n = len(res)
    k = min(n - 1, int(np.ceil((n + 1) * (1 - alpha))))
    return np.sort(res, axis=0)[k]


def online_q(scores_cal, scores_te, ts_te, alpha, every=25):
    """Per-test-window split-CP quantile with matured feedback: window i uses
    calibration scores plus test scores j with ts_j + H <= ts_i (T1
    convention; makes split-based methods fair against ACI/NexCP on
    nonstationary streams). Recomputed every `every` steps. [m, Hdim]."""
    m = len(scores_te)
    q = np.zeros((m, scores_cal.shape[1]))
    cur = split_q(scores_cal, alpha)
    last_j = 0
    for i in range(m):
        j = np.searchsorted(ts_te + H, ts_te[i], side="right")
        if j > last_j and (i % every == 0):
            cur = split_q(np.concatenate([scores_cal, scores_te[:j]]), alpha)
            last_j = j
        q[i] = cur
    return q


# ----------------------------- upstream error -------------------------------
def gauss_noise(shape, sigma, rng):
    return rng.normal(0, 1, shape) * sigma


def make_hetero_sigma(n, base, rng):
    """Per-window self-reported scale sigma_t in {0.5, 1, 2} x base."""
    return base * rng.choice([0.5, 1.0, 2.0], size=n, p=[0.4, 0.35, 0.25])


def make_ar_error(n, H_, ncov, scale, rng):
    """AR(1)+regime-switching heteroscedastic upstream error for real data
    (source (b)): temporally correlated, volatility switches between calm and
    turbulent regimes — no self-reported sigma."""
    T = n + H_
    vol = np.empty(T)
    state = 1.0
    for t in range(T):
        if rng.random() < 0.02:
            state = rng.choice([0.5, 1.0, 2.5])
        vol[t] = state
    e = np.zeros((T, ncov))
    z = rng.normal(0, 1, (T, ncov))
    for t in range(1, T):
        e[t] = 0.9 * e[t - 1] + np.sqrt(1 - 0.9 ** 2) * z[t] * vol[t]
    e = e * scale
    return np.stack([e[i:i + H_] for i in range(n)])  # [n, H, ncov]


# ----------------------------- CovProp core ---------------------------------
def mc_quantiles(ws, Xh, Xc_obs, ts, sampler, res_pool, sd, rng, k_mc=K_MC):
    """Quantiles of the MC convolution {f(x'_k) + r_k * sd_t} (residual pool
    is in volatility-normalized units, rescaled per window)."""
    S = []
    for k in range(k_mc):
        pk = preds(ws, Xh, Xc_obs + sampler(k), ts)
        r = res_pool[rng.randint(0, len(res_pool), len(pk))]
        S.append(pk + r * sd)
    S = np.stack(S, 0)
    return (np.quantile(S, ALPHA / 2, axis=0),
            np.quantile(S, 1 - ALPHA / 2, axis=0))


def covprop_mc(ws, cal, test, sampler_cal, sampler_te, res_A_signed, rng):
    """Returns test (lo, hi). Second-stage lambda wrapper on noisy split B."""
    (XhB, XcB_obs, YB, tsB, sdB) = cal
    (XhT, XcT_obs, tsT, sdT) = test[:4]
    lo_B, hi_B = mc_quantiles(ws, XhB, XcB_obs, tsB, sampler_cal, res_A_signed, sdB, rng)
    mid_B, half_B = (lo_B + hi_B) / 2, np.maximum((hi_B - lo_B) / 2, 1e-8)
    sc_B = np.abs(YB - mid_B) / half_B
    lo_T, hi_T = mc_quantiles(ws, XhT, XcT_obs, tsT, sampler_te, res_A_signed, sdT, rng)
    mid_T, half_T = (lo_T + hi_T) / 2, (hi_T - lo_T) / 2
    YT, tsT_arr = test[4], np.asarray(tsT)
    sc_T = np.abs(YT - mid_T) / np.maximum(half_T, 1e-8)
    lam = online_q(sc_B, sc_T, tsT_arr, ALPHA)             # [m, H] matured
    return mid_T - lam * half_T, mid_T + lam * half_T, lam


def covprop_nested(ws, XhT, XcT_obs, tsT, radius_te, q_a_minus_b, rng, k=K_MC // 2):
    """Bonferroni nested: union of predictions over the stage-1 covariate box
    (|x' - x_obs| <= radius, per entry), approximated by k uniform vertices."""
    Pmin = None
    Pmax = None
    for _ in range(k):
        u = rng.uniform(-1, 1, XcT_obs.shape)
        pk = preds(ws, XhT, XcT_obs + radius_te * u, tsT)
        Pmin = pk if Pmin is None else np.minimum(Pmin, pk)
        Pmax = pk if Pmax is None else np.maximum(Pmax, pk)
    return Pmin - q_a_minus_b, Pmax + q_a_minus_b


# ----------------------------- evaluation -----------------------------------
def evaluate(Y, lo, hi, eps_mag, endo_width=None):
    inside = (Y >= lo) & (Y <= hi)
    qs = np.quantile(eps_mag, [1 / 3, 2 / 3])
    bins = np.digitize(eps_mag, qs)
    width = float((hi - lo).mean())
    rep = {"coverage": float(inside.mean()),
           "width": width,
           "winkler": winkler(Y, lo, hi, ALPHA),
           "cov_by_noise_tercile": [float(inside[bins == b].mean())
                                    for b in range(3)]}
    if endo_width is not None:
        rep["width_endo"] = float(endo_width)
        rep["width_prop"] = float(max(width - endo_width, 0.0))
    return rep


# ----------------------------- one run --------------------------------------
def run_once(data, seed, source, sigma_base=0.5, hetero=True, scale_b=0.3,
             n_tr=4000, n_ca=1600, n_te=1500, methods_all=True):
    """source: 'gauss' (a) or 'nonparam' (b)."""
    rng = np.random.RandomState(seed)
    T = len(data)
    tr_end, cal_end = int(T * 0.6), int(T * 0.8)
    Xh_tr, Xc_tr, Y_tr, ts_tr = build_windows(data, 0, tr_end, n_tr)
    Xh_ca, Xc_ca, Y_ca, ts_ca = build_windows(data, tr_end, cal_end, n_ca)
    Xh_te, Xc_te, Y_te, ts_te = build_windows(data, cal_end, T, n_te)
    ncov = Xc_tr.shape[2]

    # ---- upstream error ----------------------------------------------------
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

    # local-history volatility for score normalization (T1 convention;
    # essential on nonstationary market data)
    sd_ca = Xh_ca.std(axis=1, keepdims=True) + 1e-5
    sd_te = Xh_te.std(axis=1, keepdims=True) + 1e-5

    # ---- base + exodrop forecasters ---------------------------------------
    ws = train_base(Xh_tr, Xc_tr, Y_tr, ts_tr)
    ws_drop = train_base(Xh_tr, Xc_tr, Y_tr, ts_tr, drop=0.3,
                         rng=np.random.RandomState(seed + 7)) if methods_all else None

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

    # online matured quantiles (fair vs ACI/NexCP on nonstationary streams);
    # naive stays frozen-clean-calibration on purpose: it IS the failure mode
    q_clean_static = split_q(sc_clean_all, ALPHA)
    q_orc = online_q(sc_clean_all, np.abs(Y_te - p_te_clean) / sd_te,
                     ts_te, ALPHA)
    q_rec = online_q(res_B_noisy, np.abs(Y_te - p_te_obs) / sd_te, ts_te, ALPHA)

    out = {}
    out["oracle"] = evaluate(Y_te, p_te_clean - q_orc * sd_te,
                             p_te_clean + q_orc * sd_te, mag_te)
    out["naive"] = evaluate(Y_te, p_te_obs - q_clean_static * sd_te,
                            p_te_obs + q_clean_static * sd_te, mag_te,
                            endo_width=endo_w)
    out["recal"] = evaluate(Y_te, p_te_obs - q_rec * sd_te,
                            p_te_obs + q_rec * sd_te, mag_te, endo_width=endo_w)

    if methods_all:
        # exodrop: point-robust model, recalibrated on the noisy stream
        pB_d = preds(ws_drop, Xh_ca[B], Xc_ca_obs[B], ts_ca[B])
        pT_d = preds(ws_drop, Xh_te, Xc_te_obs, ts_te)
        q_d = online_q(np.abs(Y_ca[B] - pB_d) / sd_ca[B],
                       np.abs(Y_te - pT_d) / sd_te, ts_te, ALPHA)
        out["exodrop"] = evaluate(Y_te, pT_d - q_d * sd_te, pT_d + q_d * sd_te,
                                  mag_te)

        # ACI / NexCP on the noisy residual stream (covariate-blind)
        resB = (Y_ca[B] - p_caB_obs) / sd_ca[B]
        resT = (Y_te - p_te_obs) / sd_te
        for M in (ACI(ALPHA), NexCP(ALPHA)):
            c, q = M.run(resB, resT, list(ts_ca[B]), list(ts_te), H)
            out[M.name] = evaluate(Y_te, p_te_obs + (c - q) * sd_te,
                                   p_te_obs + (c + q) * sd_te, mag_te)

        # T1 RC-CP (Mondrian on reliability = -sigma_t), source (a) only
        if source == "gauss":
            mcp = MondrianCP(ALPHA, n_bins=3)
            mcp.calibrate(res_B_noisy, -sig_ca[B])
            qm = mcp.qhat[np.digitize(-sig_te, mcp.edges)]
            out["rc_cp"] = evaluate(Y_te, p_te_obs - qm * sd_te,
                                    p_te_obs + qm * sd_te, mag_te)

    # ---- CovProp-MC ---------------------------------------------------------
    rng2 = np.random.RandomState(seed + 1)
    res_signed = res_A_clean * np.sign(rng2.normal(size=res_A_clean.shape))
    if source == "gauss":
        samp_ca = lambda k: gauss_noise(Xc_ca_obs[B].shape, sig_ca[B][:, None, None], rng2)
        samp_te = lambda k: gauss_noise(Xc_te_obs.shape, sig_te[:, None, None], rng2)
    else:
        # nonparametric: block-bootstrap whole H-length historical
        # forecast-vs-actual error trajectories (preserves the temporal
        # correlation of upstream errors within a window)
        pool = eps_ca[A]                              # [nA, H, ncov]
        def _draw(n):
            return pool[rng2.randint(0, len(pool), n)]
        samp_ca = lambda k: _draw(len(Xc_ca_obs[B]))
        samp_te = lambda k: _draw(len(Xc_te_obs))
    lo, hi, lam = covprop_mc(ws, (Xh_ca[B], Xc_ca_obs[B], Y_ca[B], ts_ca[B], sd_ca[B]),
                             (Xh_te, Xc_te_obs, ts_te, sd_te, Y_te),
                             samp_ca, samp_te, res_signed, rng2)
    out["covprop_mc"] = evaluate(Y_te, lo, hi, mag_te, endo_width=endo_w)
    out["covprop_mc"]["lambda_mean"] = float(lam.mean())

    # ---- CovProp-Nested (beta grid) ------------------------------------------
    from scipy.stats import norm
    for beta in BETAS:
        qab = split_q(res_A_clean, ALPHA - beta)
        if source == "gauss":
            rad = (norm.ppf(1 - beta / 2) * sig_te)[:, None, None]
        else:
            # conformalized per-covariate radius from historical residuals
            rad = np.quantile(np.abs(eps_ca[A]).reshape(-1, ncov),
                              1 - beta, axis=0)[None, None, :]
            rad = np.broadcast_to(rad, Xc_te_obs.shape)
        lo, hi = covprop_nested(ws, Xh_te, Xc_te_obs, ts_te, rad, qab * sd_te, rng2)
        out[f"covprop_nest_b{beta}"] = evaluate(Y_te, lo, hi, mag_te,
                                                endo_width=float((2 * qab * sd_te).mean()))
    return out


# ----------------------------- experiments ----------------------------------
def load_dataset(name):
    if name == "SynDeg":
        return generate()
    a = load_market(name)
    return a


def agg(runs):
    keys = runs[0].keys()
    res = {}
    for m in keys:
        rs = [r[m] for r in runs if m in r]
        res[m] = {}
        for fld in ("coverage", "width", "winkler"):
            vals = [x[fld] for x in rs]
            res[m][fld] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        res[m]["worst_tercile_cov"] = {
            "mean": float(np.mean([min(x["cov_by_noise_tercile"]) for x in rs])),
            "std": float(np.std([min(x["cov_by_noise_tercile"]) for x in rs]))}
        res[m]["cov_by_noise_tercile"] = [
            float(np.mean([x["cov_by_noise_tercile"][b] for x in rs]))
            for b in range(3)]
        if "width_prop" in rs[0]:
            res[m]["width_endo"] = float(np.mean([x["width_endo"] for x in rs]))
            res[m]["width_prop"] = float(np.mean([x["width_prop"] for x in rs]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all",
                    choices=["all", "main", "sensitivity", "figures"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    if args.exp in ("all", "main"):
        # --- main table: SynDeg-hetero (a) + NP/PJM (b) ----------------------
        for ds, source, kw in (("SynDeg", "gauss", dict(sigma_base=0.5, hetero=True)),
                               ("NP", "nonparam", dict(scale_b=0.3)),
                               ("PJM", "nonparam", dict(scale_b=0.3))):
            data = load_dataset(ds)
            if ds != "SynDeg":
                mu, sd = data[:int(len(data) * .6)].mean(0), data[:int(len(data) * .6)].std(0) + 1e-8
                data = (data - mu) / sd
            runs = [run_once(data, s, source, **kw) for s in args.seeds]
            rep = {"dataset": ds, "source": source, "seeds": args.seeds,
                   "aggregate": agg(runs), "runs": runs}
            with open(f"{OUT}/main_{ds}.json", "w") as f:
                json.dump(rep, f, indent=1)
            print(f"== {ds} ({source}) [{time.time()-t0:.0f}s]")
            for m, v in rep["aggregate"].items():
                print(f"  {m:20s} cov={v['coverage']['mean']:.3f}±{v['coverage']['std']:.3f} "
                      f"worst-terc={v['worst_tercile_cov']['mean']:.3f} "
                      f"width={v['width']['mean']:.2f} wink={v['winkler']['mean']:.2f}")

    if args.exp in ("all", "sensitivity"):
        # --- SynDeg collapse curve: fixed sigma grid -------------------------
        data = load_dataset("SynDeg")
        sens = {}
        for sigma in (0.1, 0.25, 0.5, 1.0):
            runs = [run_once(data, s, "gauss", sigma_base=sigma, hetero=False,
                             methods_all=False) for s in args.seeds]
            sens[str(sigma)] = agg(runs)
            print(f"[sens sigma={sigma}] naive={sens[str(sigma)]['naive']['coverage']['mean']:.3f} "
                  f"covprop={sens[str(sigma)]['covprop_mc']['coverage']['mean']:.3f} "
                  f"({time.time()-t0:.0f}s)")
        with open(f"{OUT}/sensitivity_syndeg.json", "w") as f:
            json.dump(sens, f, indent=1)

    if args.exp in ("all", "figures"):
        make_figures()

    # CSV summary
    rows = []
    for fp in sorted(os.listdir(OUT)):
        if fp.startswith("main_") and fp.endswith(".json"):
            rep = json.load(open(f"{OUT}/{fp}"))
            for m, v in rep["aggregate"].items():
                rows.append([rep["dataset"], m,
                             f"{v['coverage']['mean']:.4f}", f"{v['coverage']['std']:.4f}",
                             f"{v['worst_tercile_cov']['mean']:.4f}",
                             f"{v['width']['mean']:.4f}", f"{v['winkler']['mean']:.4f}",
                             f"{v.get('width_endo', ''):}", f"{v.get('width_prop', ''):}"])
    with open(f"{OUT}/summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "cov_mean", "cov_std",
                    "worst_tercile_cov", "width", "winkler",
                    "width_endo", "width_prop"])
        w.writerows(rows)
    print(f"\ndone in {time.time()-t0:.0f}s -> {OUT}/")


# ----------------------------- figures --------------------------------------
def make_figures():
    C = {"oracle": "gray", "naive": "#4878d0", "recal": "#ee854a",
         "exodrop": "#dc7ec0", "aci": "#6acc64", "nexcp": "#d65f5f",
         "rc_cp": "#8c613c", "covprop_mc": "#956cb4",
         "covprop_nest_b0.02": "#82c6e2"}
    # fig 1: collapse curve
    if os.path.exists(f"{OUT}/sensitivity_syndeg.json"):
        sens = json.load(open(f"{OUT}/sensitivity_syndeg.json"))
        sigmas = sorted(float(s) for s in sens)
        fig, ax = plt.subplots(figsize=(6, 4))
        for m in ("oracle", "naive", "recal", "covprop_mc", "covprop_nest_b0.02"):
            ax.plot(sigmas, [sens[str(s)][m]["coverage"]["mean"] for s in sigmas],
                    marker="o", label=m, color=C.get(m))
        ax.axhline(0.9, color="k", ls="--", lw=1)
        ax.set_xlabel("upstream forecast error sigma")
        ax.set_ylabel("marginal coverage")
        ax.set_title("Coverage collapse and recovery (SynDeg, 3 seeds)")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{OUT}/fig1_collapse_curve.png", dpi=170)
        plt.close(fig)
    # fig 2: conditional coverage by noise tercile (SynDeg hetero) + fig 3 decomposition
    if os.path.exists(f"{OUT}/main_SynDeg.json"):
        rep = json.load(open(f"{OUT}/main_SynDeg.json"))["aggregate"]
        ms = [m for m in ("naive", "recal", "exodrop", "aci", "nexcp", "rc_cp",
                          "covprop_mc", "covprop_nest_b0.02") if m in rep]
        x = np.arange(3); wd = 0.8 / len(ms)
        fig, ax = plt.subplots(figsize=(7.5, 4))
        for i, m in enumerate(ms):
            ax.bar(x + (i - (len(ms) - 1) / 2) * wd, rep[m]["cov_by_noise_tercile"],
                   wd, label=m, color=C.get(m), edgecolor="black", linewidth=0.3)
        ax.axhline(0.9, color="k", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(["low |eps|", "mid |eps|", "high |eps|"])
        ax.set_ylabel("conditional coverage"); ax.set_ylim(0.35, 1.02)
        ax.set_title("Conditional coverage by covariate-error tercile (SynDeg-hetero)")
        ax.legend(fontsize=7, ncol=4)
        fig.tight_layout(); fig.savefig(f"{OUT}/fig2_conditional_coverage.png", dpi=170)
        plt.close(fig)
    # fig 3: width decomposition across datasets for covprop_mc
    tags, endo, prop = [], [], []
    for ds in ("SynDeg", "NP", "PJM"):
        fp = f"{OUT}/main_{ds}.json"
        if os.path.exists(fp):
            v = json.load(open(fp))["aggregate"]["covprop_mc"]
            if "width_endo" in v:
                tags.append(ds); endo.append(v["width_endo"]); prop.append(v["width_prop"])
    if tags:
        fig, ax = plt.subplots(figsize=(5, 4))
        x = np.arange(len(tags))
        ax.bar(x, endo, 0.5, label="endogenous residual", color="#4878d0")
        ax.bar(x, prop, 0.5, bottom=endo, label="covariate propagation", color="#ee854a")
        ax.set_xticks(x); ax.set_xticklabels(tags)
        ax.set_ylabel("mean interval width")
        ax.set_title("CovProp-MC width decomposition")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(f"{OUT}/fig3_width_decomposition.png", dpi=170)
        plt.close(fig)


if __name__ == "__main__":
    main()
