"""E2: policy analysis over the E1 degradation sweep (preregistered).

Selectors choose one intervention in {recal, covprop_mc, covprop_share} per
operating point (stream x mech x level) using only calibration-side
information.  Evaluates the preregistered hypotheses H1-H4 exactly as written
in PREREGISTRATION.md and writes policy_analysis.json.
"""
import argparse, glob, json, os
import numpy as np
from scipy import stats as sps

METHODS = ["recal", "covprop_mc", "covprop_share"]
TAU_SKILL = 1.0
TAU_SHARE = 0.05


def load_points(out_dir):
    pts = []
    for fp in sorted(glob.glob(os.path.join(out_dir, "sweep_*.json"))):
        rep = json.load(open(fp))
        for p in rep["points"]:
            agg = p["aggregate"]
            pts.append({
                "dataset": rep["dataset"], "mech": p["mech"],
                "level": p["level"],
                "skill": p["skill_vs_persistence"],
                "share": p["prop_share_cal_unclipped"],
                "backtest": p["backtest_winkler"],
                "winkler": {m: agg[m]["winkler"]["mean"] for m in METHODS},
                "wtc": {m: agg[m]["worst_tercile_cov"]["mean"]
                        for m in METHODS},
                "cov": {m: agg[m]["coverage"]["mean"] for m in METHODS}})
    return pts


def pick(policy, p):
    if policy == "pi_skill":
        return "recal" if p["skill"] >= TAU_SKILL else "covprop_share"
    if policy == "pi_skill_share":
        return "recal" if (p["skill"] >= TAU_SKILL
                           or p["share"] < TAU_SHARE) else "covprop_share"
    if policy == "pi_backtest":
        return min(METHODS, key=lambda m: p["backtest"][m])
    if policy == "oracle":
        return min(METHODS, key=lambda m: p["winkler"][m])
    return policy  # always-<method>


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.dir or os.path.join(here, "..", "results", "policy")
    pts = load_points(out_dir)
    n = len(pts)
    res = {"n_points": n, "tau_skill": TAU_SKILL, "tau_share": TAU_SHARE,
           "points": pts}

    # H1: sign accuracy of 1{skill<1} predicting share-wrapper beats recal
    pred = np.array([p["skill"] < TAU_SKILL for p in pts])
    act = np.array([p["winkler"]["covprop_share"] < p["winkler"]["recal"]
                    for p in pts])
    acc = float((pred == act).mean())
    x = np.array([-p["skill"] for p in pts])
    n1, n0 = act.sum(), (~act).sum()
    auc = float((sps.rankdata(x)[act].sum() - n1 * (n1 + 1) / 2)
                / max(n1 * n0, 1))
    res["H1"] = {"sign_accuracy": acc, "n_true_gain": int(n1),
                 "auc_neg_skill": auc, "pass": bool(acc >= 0.80),
                 "confusion": {
                     "tp": int((pred & act).sum()),
                     "fp": int((pred & ~act).sum()),
                     "fn": int((~pred & act).sum()),
                     "tn": int((~pred & ~act).sum())}}

    # H2 / H4: decision gain / regret
    policies = ["pi_skill", "pi_skill_share", "pi_backtest", "recal",
                "covprop_mc", "covprop_share", "oracle"]
    wk = {pol: np.array([p["winkler"][pick(pol, p)] for p in pts])
          for pol in policies}
    wk["random"] = np.mean(
        [np.array([p["winkler"][m] for p in pts]) for m in METHODS], axis=0)
    wtc = {pol: np.array([p["wtc"][pick(pol, p)] for p in pts])
           for pol in policies}
    per_pol = {pol: {"mean_winkler": float(wk[pol].mean()),
                     "mean_worst_terc": float(wtc[pol].mean())
                     if pol in wtc else None,
                     "mean_regret_vs_oracle_pct": float(
                         ((wk[pol] - wk["oracle"]) / wk["oracle"]).mean()
                         * 100)}
               for pol in list(policies) + ["random"] if pol in wk}

    def wilcox(a, b):
        if np.allclose(a, b):
            return 1.0
        return float(sps.wilcoxon(a, b).pvalue)

    def gain_block(pol):
        return {
            "vs_always_recal": {
                "mean_improvement_pct": float(
                    ((wk["recal"] - wk[pol]) / wk["recal"]).mean() * 100),
                "wilcoxon_p": wilcox(wk[pol], wk["recal"])},
            "vs_always_share": {
                "mean_improvement_pct": float(
                    ((wk["covprop_share"] - wk[pol])
                     / wk["covprop_share"]).mean() * 100),
                "wilcoxon_p": wilcox(wk[pol], wk["covprop_share"])},
            "regret_pct": per_pol[pol]["mean_regret_vs_oracle_pct"]}

    h2 = gain_block("pi_skill")
    h2["per_policy"] = per_pol
    h2["pass"] = bool(h2["regret_pct"] <= 3.0
                      and h2["vs_always_recal"]["mean_improvement_pct"] > 0
                      and h2["vs_always_recal"]["wilcoxon_p"] < .05
                      and h2["vs_always_share"]["mean_improvement_pct"] > 0
                      and h2["vs_always_share"]["wilcoxon_p"] < .05)
    res["H2"] = h2

    h4 = gain_block("pi_backtest")
    agree = np.array([pick("pi_backtest", p) == pick("oracle", p)
                      for p in pts])
    h4["winner_agreement"] = float(agree.mean())
    h4["pass"] = bool(h4["regret_pct"] <= 3.0
                      and h4["vs_always_recal"]["mean_improvement_pct"] > 0
                      and h4["vs_always_recal"]["wilcoxon_p"] < .05
                      and h4["vs_always_share"]["mean_improvement_pct"] > 0
                      and h4["vs_always_share"]["wilcoxon_p"] < .05
                      and h4["winner_agreement"] >= 0.80)
    res["H4"] = h4

    # H3: within-stream, per-axis monotonicity + crossover location
    per_stream, signs, cross = {}, [], []
    for ds in sorted(set(p["dataset"] for p in pts)):
        for mech in ("noise", "blend"):
            sub = [p for p in pts if p["dataset"] == ds
                   and (p["mech"] == mech
                        or (mech == "blend" and p["mech"] == "noise"
                            and p["level"] == 0.0))]
            sub = sorted(sub, key=lambda p: (p["level"]
                                             if p["mech"] == mech else -1))
            if len(sub) < 3:
                continue
            gain = [(p["winkler"]["recal"] - p["winkler"]["covprop_share"])
                    / p["winkler"]["recal"] for p in sub]
            lev = list(range(len(sub)))
            rho = float(sps.spearmanr(lev, gain).statistic)
            signs.append(rho <= 0)
            row = {"levels": [(p["mech"], p["level"]) for p in sub],
                   "gain_share_vs_recal": gain,
                   "skill": [p["skill"] for p in sub],
                   "spearman_level_gain": rho}
            flips = [i for i in range(1, len(sub))
                     if (gain[i - 1] > 0) != (gain[i] > 0)]
            if flips and gain[0] > 0:
                i = flips[0]
                row["crossover_skill_interval"] = [sub[i - 1]["skill"],
                                                   sub[i]["skill"]]
                cross.append(row["crossover_skill_interval"])
            per_stream[f"{ds}__{mech}"] = row
    k = sum(signs)
    p_sign = float(sps.binomtest(k, len(signs), 0.5,
                                 alternative="greater").pvalue) \
        if signs else 1.0
    in_band = [c for c in cross if not (max(c) < 0.9 or min(c) > 1.1)]
    res["H3"] = {"per_stream_axis": per_stream,
                 "n_monotone_nonincreasing": k, "n_axes": len(signs),
                 "sign_test_p": p_sign,
                 "crossover_intervals": cross,
                 "crossovers_intersect_0.9_1.1": len(in_band),
                 "pass": bool(p_sign < 0.05
                              and len(in_band) == len(cross))}

    res["all_pass"] = bool(res["H1"]["pass"] and res["H2"]["pass"]
                           and res["H3"]["pass"] and res["H4"]["pass"])
    with open(os.path.join(out_dir, "policy_analysis.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"n_points={n}")
    print(f"H1 sign_acc={res['H1']['sign_accuracy']:.3f} "
          f"auc={res['H1']['auc_neg_skill']:.3f} pass={res['H1']['pass']}")
    for tag, h in (("H2 pi_skill", h2), ("H4 pi_backtest", h4)):
        print(f"{tag} regret={h['regret_pct']:.2f}% "
              f"vs_recal=+{h['vs_always_recal']['mean_improvement_pct']:.2f}% "
              f"(p={h['vs_always_recal']['wilcoxon_p']:.4f}) "
              f"vs_share=+{h['vs_always_share']['mean_improvement_pct']:.2f}% "
              f"(p={h['vs_always_share']['wilcoxon_p']:.4f}) "
              + (f"agree={h['winner_agreement']:.3f} "
                 if "winner_agreement" in h else "")
              + f"pass={h['pass']}")
    print(f"H3 monotone={k}/{len(signs)} (p={p_sign:.4f}) "
          f"crossovers={cross} pass={res['H3']['pass']}")
    print(f"ALL PASS: {res['all_pass']}")


if __name__ == "__main__":
    main()
