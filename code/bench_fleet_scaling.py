"""Amendment A1 (logged in DESIGN.md before this run): the first H-S3
measurement timed worker startup (per-process state construction) together
with the online loop, confounding sharding scalability with fixed spawn
cost.  This rerun times the interleaved loop only (max over workers of the
loop wall), and reports both.  Nothing else changes; H-S1/H-S2/H-S4 verdicts
from fleet_summary.json stand."""
import json, os, time
from multiprocessing import get_context

HERE = os.path.dirname(os.path.abspath(__file__))
import bench_fleet as BF  # noqa: E402


def worker(args):
    (base_names, n_streams, n_windows, wid) = args
    bases = {n: BF.base_state(n) for n in base_names}
    states = [BF.clone_state(bases[BF.STREAMS[i % 7]], i)
              for i in range(wid * n_streams, (wid + 1) * n_streams)]
    r = BF.run_interleaved(states, n_windows, record_lat=False)
    return r["windows"], r["wall_s"]


def main():
    out = {}
    ctx = get_context("spawn")
    for nproc in [1, 2, 4, 8]:
        per, nw = 125, 8
        args = [(BF.STREAMS, per, nw, w) for w in range(nproc)]
        t1 = time.perf_counter()
        with ctx.Pool(nproc) as p:
            res = p.map(worker, args)
        outer = time.perf_counter() - t1
        wins = sum(r[0] for r in res)
        loop_wall = max(r[1] for r in res)
        out["P%d" % nproc] = {
            "workers": nproc, "windows": wins,
            "outer_wall_s": outer, "loop_wall_max_s": loop_wall,
            "agg_win_per_s_loop_only": wins / loop_wall,
            "agg_win_per_s_incl_startup": wins / outer}
        print("P=%d: loop-only %.1f win/s (incl startup %.1f)" %
              (nproc, wins / loop_wall, wins / outer), flush=True)
    eff = (out["P8"]["agg_win_per_s_loop_only"] /
           out["P1"]["agg_win_per_s_loop_only"])
    out["H-S3_loop_only_speedup"] = eff
    out["H-S3_pass_ge_6x"] = bool(eff >= 6.0)
    with open(os.path.join(HERE, "results_fleet",
                           "fleet_scaling_a1.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"speedup": eff, "pass": eff >= 6.0}, indent=1))


if __name__ == "__main__":
    main()
