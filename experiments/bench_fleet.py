"""Fleet-scale end-to-end benchmark for CovProp-MC (preregistered in
oral_push/DESIGN.md, H-S1..H-S4).  Builds N independent per-stream states by
cycling the seven real streams, then measures (i) amortized throughput and
online-latency percentiles of the interleaved single-window matured-feedback
loop, (ii) per-stream persistent memory at N=1000 (tracemalloc, includes
Python object overhead), (iii) multi-process sharding scalability (streams
are share-nothing).  CPU only; reads existing pools only; raw/frozen
untouched; output results_fleet/*.json (<5 MB)."""
import json, os, sys, time, tracemalloc
from multiprocessing import get_context
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "core"))

from covprop_core import (ALPHA, H, build_windows, train_base,  # noqa: E402
                          split_q)
from upstream import (upstream_forecast, mc_quantiles_any,    # noqa: E402
                        preds_any)
from share import load_any_r3                              # noqa: E402

STREAMS = ["NP", "FR", "ETTh1", "weather", "ETTh2", "ETTm2", "electricity"]
K = 64
SEED = 0


def base_state(name):
    data, _ = load_any_r3(name)
    T = len(data)
    tr_end, cal_end = int(T * 0.6), int(T * 0.8)
    Xh_tr, Xc_tr, Y_tr, ts_tr = build_windows(data, 0, tr_end, 4000)
    Xh_ca, Xc_ca, Y_ca, ts_ca = build_windows(data, tr_end, cal_end, 1600)
    Xh_te, Xc_te, Y_te, ts_te = build_windows(data, cal_end, T, 300)
    up_ca = upstream_forecast(data, ts_ca, tr_end)
    up_te = upstream_forecast(data, ts_te, tr_end)
    eps_ca, eps_te = up_ca - Xc_ca, up_te - Xc_te
    Xc_ca_obs, Xc_te_obs = Xc_ca + eps_ca, Xc_te + eps_te
    sd_ca = Xh_ca.std(axis=1, keepdims=True) + 1e-5
    sd_te = Xh_te.std(axis=1, keepdims=True) + 1e-5
    ws = train_base(Xh_tr, Xc_tr, Y_tr, ts_tr)
    nA = len(Y_ca) // 2
    A, B = slice(0, nA), slice(nA, len(Y_ca))
    p_ca_clean = preds_any(ws, Xh_ca, Xc_ca, ts_ca)
    res_A = np.abs(Y_ca[A] - p_ca_clean[A]) / sd_ca[A]
    rng = np.random.RandomState(SEED + 1)
    res_signed = res_A * np.sign(rng.normal(size=res_A.shape))
    pool = eps_ca[A]
    p_caB = preds_any(ws, Xh_ca[B], Xc_ca_obs[B], ts_ca[B])
    score_buf = np.abs(Y_ca[B] - p_caB) / sd_ca[B]
    return {"name": name, "ws": ws, "pool": pool, "res_signed": res_signed,
            "score_buf": score_buf, "Xh_te": Xh_te, "Xc_te_obs": Xc_te_obs,
            "ts_te": ts_te, "Y_te": Y_te, "sd_te": sd_te}


def clone_state(b, i):
    # independent per-stream copies of the persistent primitives (the thing
    # H-S2 measures); read-only test windows are shared views.
    return {"name": "%s#%d" % (b["name"], i), "ws": b["ws"],
            "pool": b["pool"].copy(), "res_signed": b["res_signed"].copy(),
            "score_buf": b["score_buf"].copy(),
            "Xh_te": b["Xh_te"], "Xc_te_obs": b["Xc_te_obs"],
            "ts_te": b["ts_te"], "Y_te": b["Y_te"], "sd_te": b["sd_te"],
            "rng": np.random.RandomState(1000 + i), "cursor": i % 100}


def persistent_nbytes(s):
    return int(s["pool"].nbytes + s["res_signed"].nbytes +
               s["score_buf"].nbytes)


def one_window(s):
    j = s["cursor"] % len(s["Y_te"])
    s["cursor"] += 1
    one = slice(j, j + 1)
    rng = s["rng"]
    pool = s["pool"]
    samp = lambda k: pool[rng.randint(0, len(pool), 1)]
    lo, hi = mc_quantiles_any(s["ws"], s["Xh_te"][one], s["Xc_te_obs"][one],
                              s["ts_te"][one], samp, s["res_signed"],
                              s["sd_te"][one], rng, k_mc=K)
    # stage-2 correction from the maintained score buffer
    lam = split_q(s["score_buf"], ALPHA)
    mid, half = (lo + hi) / 2, np.maximum((hi - lo) / 2, 1e-8)
    # matured-feedback maintenance: append this window score, refresh quantile
    sc = np.abs(s["Y_te"][one] - mid) / np.maximum(half, 1e-8)
    s["score_buf"] = np.concatenate([s["score_buf"][-1599:], sc])
    return lo, hi, lam


def run_interleaved(states, n_windows, record_lat=True):
    lats = []
    t0 = time.perf_counter()
    done = 0
    for w in range(n_windows):
        for s in states:
            if record_lat:
                t1 = time.perf_counter()
                one_window(s)
                lats.append(time.perf_counter() - t1)
            else:
                one_window(s)
            done += 1
    wall = time.perf_counter() - t0
    out = {"windows": done, "wall_s": wall, "win_per_s": done / wall}
    if record_lat:
        a = np.array(lats) * 1e3
        out["lat_ms"] = {"p50": float(np.percentile(a, 50)),
                         "p90": float(np.percentile(a, 90)),
                         "p99": float(np.percentile(a, 99)),
                         "mean": float(a.mean())}
    return out


def worker(args):
    (base_names, n_streams, n_windows, wid) = args
    bases = {n: base_state(n) for n in base_names}
    states = [clone_state(bases[STREAMS[i % 7]], i)
              for i in range(wid * n_streams, (wid + 1) * n_streams)]
    r = run_interleaved(states, n_windows, record_lat=False)
    return r["windows"], r["wall_s"]


def main():
    out_dir = os.path.join(ROOT, "results", "fleet")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    print("building 7 base states...", flush=True)
    bases = {n: base_state(n) for n in STREAMS}

    # ---- memory per stream at N=1000 (H-S2) ----
    tracemalloc.start()
    states = [clone_state(bases[STREAMS[i % 7]], i) for i in range(1000)]
    cur, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    arr_bytes = float(np.mean([persistent_nbytes(s) for s in states]))
    mem = {"n_streams": 1000, "traced_total_mb": cur / 1e6,
           "traced_per_stream_mb": cur / 1e6 / 1000,
           "array_persistent_per_stream_mb": arr_bytes / 1e6}
    print("mem/stream traced %.3f MB, arrays %.3f MB" %
          (mem["traced_per_stream_mb"],
           mem["array_persistent_per_stream_mb"]), flush=True)

    # ---- interleaved online loop, N=100 and N=1000 (H-S1, H-S4) ----
    inter = {}
    for N, nw in [(100, 20), (1000, 6)]:
        sub = states[:N]
        run_interleaved(sub[: min(N, 50)], 1, record_lat=False)  # warmup
        inter["N%d" % N] = run_interleaved(sub, nw)
        r = inter["N%d" % N]
        print("N=%d: %.1f win/s, p50 %.1f p99 %.1f ms (%.0fs)" %
              (N, r["win_per_s"], r["lat_ms"]["p50"], r["lat_ms"]["p99"],
               time.time() - t0), flush=True)

    # derived fleet capacity per core at operational cadences
    wps = inter["N1000"]["win_per_s"]
    capacity = {"streams_per_core_5min_cadence": wps * 300,
                "streams_per_core_1min_cadence": wps * 60,
                "streams_per_core_5s_cadence": wps * 5}

    # ---- multi-process sharding (H-S3) ----
    del states
    scaling = {}
    ctx = get_context("spawn")
    for nproc in [1, 2, 4, 8]:
        per = 125  # streams per worker
        nw = 4
        args = [(STREAMS, per, nw, w) for w in range(nproc)]
        t1 = time.perf_counter()
        with ctx.Pool(nproc) as p:
            res = p.map(worker, args)
        wall = time.perf_counter() - t1
        wins = sum(r[0] for r in res)
        scaling["P%d" % nproc] = {"workers": nproc, "windows": wins,
                                  "wall_s": wall,
                                  "agg_win_per_s": wins / wall}
        print("P=%d: %.1f win/s agg (%.0fs)" %
              (nproc, wins / wall, time.time() - t0), flush=True)
    eff = (scaling["P8"]["agg_win_per_s"] /
           scaling["P1"]["agg_win_per_s"])

    # ---- preregistered verdicts ----
    verdicts = {
        "H-S1_throughput_ge_2000wps_N1000": bool(wps >= 2000),
        "H-S1_measured_wps": wps,
        "H-S2_per_stream_le_1.5MB": bool(mem["traced_per_stream_mb"] <= 1.5),
        "H-S2_measured_mb": mem["traced_per_stream_mb"],
        "H-S3_8worker_speedup_ge_6x": bool(eff >= 6.0),
        "H-S3_measured_speedup": eff,
        "H-S4_p99_le_60ms_N1000": bool(
            inter["N1000"]["lat_ms"]["p99"] <= 60.0),
        "H-S4_measured_p99_ms": inter["N1000"]["lat_ms"]["p99"]}

    out = {"K": K, "alpha": ALPHA, "H": int(H), "seed": SEED,
           "memory": mem, "interleaved": inter, "capacity": capacity,
           "scaling": scaling, "verdicts": verdicts,
           "runtime_s": time.time() - t0}
    with open(os.path.join(out_dir, "fleet_summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(verdicts, indent=1), flush=True)
    print("done in %.0fs -> %s/" % (time.time() - t0, out_dir), flush=True)


if __name__ == "__main__":
    main()
