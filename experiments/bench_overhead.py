"""E3: runtime/storage overhead microbenchmark (preregistered in
PREREG_FIX.md).  Measures, per real upstream-error stream, the batch and
single-window latency of the CovProp-MC pipeline vs K, against recalibration
and CQR, plus memory/storage of the observability primitives and the
incremental maintenance cost of the stage-2 quantile(s).  CPU only; nothing
frozen is modified; outputs results/overhead_<stream>.json.
"""
import json, os, sys, time, tracemalloc
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "core"))

import covprop_core as CF                                        # noqa: E402
from covprop_core import (ALPHA, H, build_windows, train_base, split_q,
                          online_q)                              # noqa: E402
from adaptive_baselines import fit_pinball, pred_lin                     # noqa: E402
from covprop_core import features                                # noqa: E402
from upstream import (upstream_forecast, mc_quantiles_any,
                        preds_any)                               # noqa: E402
from share import (load_any_r3, share_variable,
                        binned_lambda)                           # noqa: E402

STREAMS = ["NP", "FR", "ETTh1", "weather", "ETTh2", "ETTm2", "electricity"]
KS = [8, 16, 32, 64, 128]
REPS = 3
SEED = 0


def med_time(fn, reps=REPS):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def bench_stream(name):
    data, _ = load_any_r3(name)
    T = len(data)
    tr_end, cal_end = int(T * 0.6), int(T * 0.8)
    Xh_tr, Xc_tr, Y_tr, ts_tr = build_windows(data, 0, tr_end, 4000)
    Xh_ca, Xc_ca, Y_ca, ts_ca = build_windows(data, tr_end, cal_end, 1600)
    Xh_te, Xc_te, Y_te, ts_te = build_windows(data, cal_end, T, 1500)

    up_ca = upstream_forecast(data, ts_ca, tr_end)
    up_te = upstream_forecast(data, ts_te, tr_end)
    eps_ca, eps_te = up_ca - Xc_ca, up_te - Xc_te
    Xc_ca_obs, Xc_te_obs = Xc_ca + eps_ca, Xc_te + eps_te

    sd_ca = Xh_ca.std(axis=1, keepdims=True) + 1e-5
    sd_te = Xh_te.std(axis=1, keepdims=True) + 1e-5
    ws = train_base(Xh_tr, Xc_tr, Y_tr, ts_tr)

    nA = len(Y_ca) // 2
    A, B = slice(0, nA), slice(nA, len(Y_ca))
    n_te = len(Y_te)

    p_ca_clean = preds_any(ws, Xh_ca, Xc_ca, ts_ca)
    res_A_clean = np.abs(Y_ca[A] - p_ca_clean[A]) / sd_ca[A]
    rng2 = np.random.RandomState(SEED + 1)
    res_signed = res_A_clean * np.sign(rng2.normal(size=res_A_clean.shape))
    pool = eps_ca[A]
    p_caB_obs = preds_any(ws, Xh_ca[B], Xc_ca_obs[B], ts_ca[B])
    res_B_noisy = np.abs(Y_ca[B] - p_caB_obs) / sd_ca[B]

    out = {"stream": name, "n_test": int(n_te), "H": int(H),
           "n_cov": int(Xc_te.shape[2]), "alpha": ALPHA, "seed": SEED}

    # -- storage of the observability primitives -----------------------------
    out["storage_bytes"] = {
        "upstream_error_pool": int(pool.nbytes),
        "endogenous_residual_pool": int(res_signed.nbytes),
        "stage2_score_buffer": int(res_B_noisy.nbytes)}

    # -- batch test-path latency ---------------------------------------------
    t_fwd = med_time(lambda: preds_any(ws, Xh_te, Xc_te_obs, ts_te))
    sc_te_dummy = np.abs(Y_te - preds_any(ws, Xh_te, Xc_te_obs, ts_te)) / sd_te
    t_recal_q = med_time(lambda: online_q(res_B_noisy, sc_te_dummy,
                                          ts_te, ALPHA))
    out["recal"] = {"batch_s": t_fwd + t_recal_q,
                    "per_window_ms": (t_fwd + t_recal_q) / n_te * 1e3,
                    "forward_s": t_fwd, "onlineq_s": t_recal_q}

    # CQR: 2 pinball heads per horizon (training amortized; report both)
    t_cqr_tr0 = time.perf_counter()
    w_los, w_his = [], []
    for h in range(H):
        Ftr = features(Xh_tr, Xc_tr, ts_tr, h)
        w_los.append(fit_pinball(Ftr, Y_tr[:, h], ALPHA / 2))
        w_his.append(fit_pinball(Ftr, Y_tr[:, h], 1 - ALPHA / 2))
    t_cqr_tr = time.perf_counter() - t_cqr_tr0

    def cqr_fwd():
        lo = np.zeros((n_te, H)); hi = np.zeros((n_te, H))
        for h in range(H):
            FT = features(Xh_te, Xc_te_obs, ts_te, h)
            lo[:, h], hi[:, h] = pred_lin(w_los[h], FT), pred_lin(w_his[h], FT)
        return lo, hi
    t_cqr_fwd = med_time(cqr_fwd)
    lo_c, hi_c = cqr_fwd()
    sc_c = np.maximum(lo_c - Y_te, Y_te - hi_c) / sd_te
    t_cqr_q = med_time(lambda: online_q(res_B_noisy, sc_c, ts_te, ALPHA))
    out["cqr"] = {"train_once_s": t_cqr_tr,
                  "batch_s": t_cqr_fwd + t_cqr_q,
                  "per_window_ms": (t_cqr_fwd + t_cqr_q) / n_te * 1e3}

    # CovProp-MC vs K ---------------------------------------------------------
    out["covprop_mc"] = {}
    for K in KS:
        rng = np.random.RandomState(SEED + 1)
        samp_te = lambda k: pool[rng.randint(0, len(pool), n_te)]
        t_mc = med_time(lambda: mc_quantiles_any(
            ws, Xh_te, Xc_te_obs, ts_te, samp_te, res_signed, sd_te,
            rng, k_mc=K))
        lo_T, hi_T = mc_quantiles_any(ws, Xh_te, Xc_te_obs, ts_te, samp_te,
                                      res_signed, sd_te, rng, k_mc=K)
        mid_T = (lo_T + hi_T) / 2
        half_T = np.maximum((hi_T - lo_T) / 2, 1e-8)
        sc_T = np.abs(Y_te - mid_T) / half_T
        t_q = med_time(lambda: online_q(res_B_noisy, sc_T, ts_te, ALPHA))
        rec = {"batch_s": t_mc + t_q,
               "per_window_ms": (t_mc + t_q) / n_te * 1e3,
               "mc_s": t_mc, "onlineq_s": t_q,
               "x_recal": (t_mc + t_q) / max(out["recal"]["batch_s"], 1e-12)}
        if K == 64:
            tracemalloc.start()
            mc_quantiles_any(ws, Xh_te, Xc_te_obs, ts_te, samp_te,
                             res_signed, sd_te, rng, k_mc=K)
            cur, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rec["mc_buffer_peak_bytes"] = int(peak)
        out["covprop_mc"][f"K{K}"] = rec

    # CovProp-Share increment (share stats + binned lambda) -------------------
    rng = np.random.RandomState(SEED + 1)
    samp_ca = lambda k: pool[rng.randint(0, len(pool), len(Xc_ca_obs[B]))]
    lo_B, hi_B = mc_quantiles_any(ws, Xh_ca[B], Xc_ca_obs[B], ts_ca[B],
                                  samp_ca, res_signed, sd_ca[B], rng)
    mid_B = (lo_B + hi_B) / 2
    half_B = np.maximum((hi_B - lo_B) / 2, 1e-8)
    sc_B = np.abs(Y_ca[B] - mid_B) / half_B
    samp_te = lambda k: pool[rng.randint(0, len(pool), n_te)]
    lo_T, hi_T = mc_quantiles_any(ws, Xh_te, Xc_te_obs, ts_te, samp_te,
                                  res_signed, sd_te, rng)
    mid_T = (lo_T + hi_T) / 2
    half_T = np.maximum((hi_T - lo_T) / 2, 1e-8)
    sc_T = np.abs(Y_te - mid_T) / half_T
    sc_clean_all = np.abs(Y_ca - p_ca_clean) / sd_ca
    q_clean_static = split_q(sc_clean_all, ALPHA)
    endo_half_B = np.broadcast_to(q_clean_static[None, :] * sd_ca[B],
                                  half_B.shape)
    endo_half_T = np.broadcast_to(q_clean_static[None, :] * sd_te,
                                  half_T.shape)
    t_share = med_time(lambda: (share_variable(half_B, endo_half_B),
                                share_variable(half_T, endo_half_T)))
    rho_B = share_variable(half_B, endo_half_B)
    rho_T = share_variable(half_T, endo_half_T)
    t_binned = med_time(lambda: binned_lambda(sc_B, sc_T, rho_B, rho_T,
                                              ts_te, ALPHA))
    out["covprop_share_increment"] = {
        "share_stat_s": t_share, "binned_lambda_s": t_binned,
        "per_window_ms": (t_share + t_binned) / n_te * 1e3}

    # -- single-window online latency ----------------------------------------
    one = slice(0, 1)
    rng = np.random.RandomState(SEED + 1)
    samp1 = lambda k: pool[rng.randint(0, len(pool), 1)]
    t1_mc = med_time(lambda: mc_quantiles_any(
        ws, Xh_te[one], Xc_te_obs[one], ts_te[one], samp1, res_signed,
        sd_te[one], rng, k_mc=64), reps=20)
    t1_fwd = med_time(lambda: preds_any(ws, Xh_te[one], Xc_te_obs[one],
                                        ts_te[one]), reps=20)
    out["single_window_ms"] = {"covprop_mc_K64": t1_mc * 1e3,
                               "recal_forward": t1_fwd * 1e3}

    # -- incremental maintenance ----------------------------------------------
    grown = np.concatenate([res_B_noisy, sc_T[: len(sc_T) // 2]])
    t_upd_u = med_time(lambda: split_q(grown, ALPHA), reps=20)
    edges = np.quantile(rho_B, [1 / 3, 2 / 3])
    bins_g = np.digitize(np.concatenate([rho_B, rho_T[: len(sc_T) // 2]]),
                         edges)
    sc_g = np.concatenate([sc_B, sc_T[: len(sc_T) // 2]])

    def upd_binned():
        for b in range(3):
            m = bins_g == b
            if m.sum() >= 30:
                split_q(sc_g[m], ALPHA)
    t_upd_b = med_time(upd_binned, reps=20)
    out["incremental_update_ms"] = {"uniform_lambda": t_upd_u * 1e3,
                                    "binned_lambda": t_upd_b * 1e3}
    return out


def main():
    out_dir = os.path.join(ROOT, "results", "overhead")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    for ds in STREAMS:
        rep = bench_stream(ds)
        with open(os.path.join(out_dir, f"overhead_{ds}.json"), "w") as f:
            json.dump(rep, f, indent=1)
        k64 = rep["covprop_mc"]["K64"]
        print(f"== {ds}: recal {rep['recal']['per_window_ms']:.3f} ms/win, "
              f"cqr {rep['cqr']['per_window_ms']:.3f}, "
              f"mc K64 {k64['per_window_ms']:.3f} ({k64['x_recal']:.1f}x), "
              f"1-win {rep['single_window_ms']['covprop_mc_K64']:.1f} ms "
              f"({time.time()-t0:.0f}s)", flush=True)
    print(f"all done in {time.time()-t0:.0f}s -> {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
