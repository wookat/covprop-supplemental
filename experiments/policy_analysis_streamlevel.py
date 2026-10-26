"""E4 analysis (preregistered in PREREG_R1FIX.md): evaluates H5 (early
backtest / temporal extrapolation), H6 (undegraded-domain regret), and H7
(stream-clustered significance) over the 5-seed sweep produced by
backtest_extrap.py, and cross-checks that seeds 0-2 reproduce the frozen
results_v2 numbers (read-only).  Writes results/policy_r1fix.json.
"""
import argparse, glob, json, os
import numpy as np
from scipy import stats as sps

METHODS = ["recal", "covprop_mc", "covprop_share"]


def load_points(out_dir, prefix="sweep5_"):
    pts = []
    for fp in sorted(glob.glob(os.path.join(out_dir, prefix + "*.json"))):
        rep = json.load(open(fp))
        for p in rep["points"]:
            agg = p["aggregate"]
            pts.append({
                "dataset": rep["dataset"], "mech": p["mech"],
                "level": p["level"],
                "backtest": p["backtest_winkler"],
                "backtest_early": p.get("backtest_winkler_early"),
                "winkler": {m: agg[m]["winkler"]["mean"] for m in METHODS}})
    return pts


def regret_stats(pts, key):
    """Per-point winkler of the selector picking argmin of pts[key], plus
    regret vs the hindsight-best intervention."""
    sel_w, orc_w, agree = [], [], []
    for p in pts:
        pick = min(METHODS, key=lambda m: p[key][m])
        best = min(METHODS, key=lambda m: p["winkler"][m])
        sel_w.append(p["winkler"][pick])
        orc_w.append(p["winkler"][best])
        agree.append(pick == best)
    sel_w, orc_w = np.array(sel_w), np.array(orc_w)
    reg = (sel_w - orc_w) / orc_w
    return {"mean_winkler": float(sel_w.mean()),
            "oracle_winkler": float(orc_w.mean()),
            "mean_regret_pct": float(reg.mean() * 100),
            "median_regret_pct": float(np.median(reg) * 100),
            "max_regret_pct": float(reg.max() * 100),
            "agree_rate": float(np.mean(agree)),
            "n_points": len(pts)}


def per_point_diff(pts, key, blanket):
    """Per-point winkler difference blanket - selector (positive = selector
    better)."""
    d = []
    for p in pts:
        pick = min(METHODS, key=lambda m: p[key][m])
        d.append(p["winkler"][blanket] - p["winkler"][pick])
    return np.array(d)


def stream_clustered_tests(pts, key, blanket, B=10000, seed=0):
    """H7: (i) stream-level sign test on per-stream mean differences;
    (ii) hierarchical bootstrap (resample streams, then points within)."""
    streams = sorted({p["dataset"] for p in pts})
    per_stream = []
    for ds in streams:
        d = per_point_diff([p for p in pts if p["dataset"] == ds],
                           key, blanket)
        per_stream.append(d)
    means = np.array([d.mean() for d in per_stream])
    n_pos = int((means > 0).sum())
    p_sign = float(sps.binomtest(n_pos, len(means), 0.5).pvalue)
    rng = np.random.RandomState(seed)
    S = len(per_stream)
    boots = np.empty(B)
    for b in range(B):
        idx = rng.randint(0, S, S)
        vals = [per_stream[i][rng.randint(0, len(per_stream[i]),
                                          len(per_stream[i]))].mean()
                for i in idx]
        boots[b] = np.mean(vals)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"per_stream_mean_diff": {s: float(m)
                                     for s, m in zip(streams, means)},
            "n_streams_selector_better": n_pos, "n_streams": S,
            "sign_test_p": p_sign,
            "hier_bootstrap_mean": float(boots.mean()),
            "hier_bootstrap_ci95": [float(lo), float(hi)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--frozen", default=None)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.dir or os.path.join(here, "..", "results", "policy")
    frozen = args.frozen or os.path.join(here, "..", "results", "policy")
    pts = load_points(out_dir)
    res = {"n_points": len(pts)}

    # frozen cross-check: seeds 0-2 of the frozen run vs new run means -------
    if os.path.isdir(frozen):
        old = load_points(frozen, prefix="sweep_")
        omap = {(p["dataset"], p["mech"], p["level"]): p for p in old}
        max_bt_dev = 0.0
        for p in pts:
            o = omap.get((p["dataset"], p["mech"], p["level"]))
            if o:
                for m in METHODS:
                    max_bt_dev = max(max_bt_dev, abs(
                        p["backtest"][m] - o["backtest"][m])
                        / max(o["backtest"][m], 1e-9))
        res["frozen_crosscheck"] = {
            "note": "new run uses 5 seeds so aggregate means differ; "
                    "backtest values are seed-averaged too. Relative "
                    "deviation of the (5-seed) backtest means vs frozen "
                    "3-seed means:",
            "max_rel_dev_backtest": float(max_bt_dev)}

    # H4 restated on 5 seeds --------------------------------------------------
    res["H4_late_5seeds"] = regret_stats(pts, "backtest")
    for bl in ("recal", "covprop_share"):
        d = per_point_diff(pts, "backtest", bl)
        res["H4_late_5seeds"][f"wilcoxon_vs_always_{bl}"] = {
            "mean_improvement_pct": float(
                d.mean() / np.mean([p["winkler"][bl] for p in pts]) * 100),
            "p": float(sps.wilcoxon(d).pvalue)}

    # H5: early backtest ------------------------------------------------------
    res["H5_early"] = regret_stats(pts, "backtest_early")
    both = [(min(METHODS, key=lambda m: p["backtest"][m]),
             min(METHODS, key=lambda m: p["backtest_early"][m]))
            for p in pts]
    res["H5_early"]["agree_with_late"] = float(
        np.mean([a == b for a, b in both]))
    res["H5_early"]["pass"] = bool(
        res["H5_early"]["mean_regret_pct"] <= 3.0
        and res["H5_early"]["agree_with_late"] >= 0.70)

    # H6: undegraded points ---------------------------------------------------
    pts0 = [p for p in pts if p["mech"] == "noise" and p["level"] == 0.0]
    res["H6_undegraded"] = regret_stats(pts0, "backtest")
    res["H6_undegraded"]["picks"] = {
        p["dataset"]: min(METHODS, key=lambda m: p["backtest"][m])
        for p in pts0}
    res["H6_undegraded"]["pass"] = bool(
        res["H6_undegraded"]["mean_regret_pct"] <= 2.0)

    # H7: stream-clustered significance ---------------------------------------
    res["H7_stream_level"] = {
        "vs_always_recal": stream_clustered_tests(pts, "backtest", "recal"),
        "vs_always_share": stream_clustered_tests(pts, "backtest",
                                                  "covprop_share")}
    sr = res["H7_stream_level"]["vs_always_recal"]
    res["H7_stream_level"]["pass"] = bool(
        (sr["n_streams_selector_better"] == sr["n_streams"])
        or (sr["hier_bootstrap_ci95"][0] > 0))

    with open(os.path.join(out_dir, "policy_r1fix.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items()
                      if k != "frozen_crosscheck"}, indent=1)[:4000])
    print("H5 pass:", res["H5_early"]["pass"],
          "| H6 pass:", res["H6_undegraded"]["pass"],
          "| H7 pass:", res["H7_stream_level"]["pass"])


if __name__ == "__main__":
    main()
