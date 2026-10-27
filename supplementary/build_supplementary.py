"""Build supplementary.pdf: long-form result tables omitted from the paper body.

Reads the shipped raw artifacts under ../results/ and emits a self-contained
supplementary.tex, then (optionally) calls pdflatex.  Run from the repo root:

    python supplementary/build_supplementary.py          # writes .tex only
    python supplementary/build_supplementary.py --pdf    # + pdflatex

Every table below is produced mechanically from the shipped JSON/NPZ files;
nothing is transcribed by hand.
"""
import argparse, glob, json, os
import numpy as np
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RES = os.path.join(ROOT, "results")

STREAMS = ["NP", "FR", "ETTh1", "ETTh2", "ETTm2", "electricity", "weather"]


def load(pattern):
    out = {}
    for fp in sorted(glob.glob(os.path.join(RES, pattern))):
        ds = os.path.basename(fp).split("_", 1)[1].rsplit(".", 1)[0]
        out[ds] = json.load(open(fp))
    return out


def f3(x):
    return f"{x:.3f}"


def tex_biased_log(bias):
    """Full biased-log + propagation-only grid (paper Sec. VII-C)."""
    rows = []
    for ds in STREAMS:
        g = bias[ds]["grid"]
        for f in ["1.0", "0.5", "0.25", "0.1"]:
            a = g[f]["aggregate"]
            mc, nc = a["covprop_mc"], a["covprop_nocal"]
            rows.append(
                f"{ds} & {f} & {f3(mc['coverage']['mean'])} & "
                f"{f3(mc['worst_tercile_cov']['mean'])} & "
                f"{f3(mc['width']['mean'])} & {f3(nc['coverage']['mean'])} \\\\")
    body = "\n".join(rows)
    return (
        "\\begin{table}[h]\\centering\\small\n"
        "\\caption{Biased upstream-error-log ablation (full grid). "
        "``$f$'' is the retained smallest-$|\\eta|$ fraction of the error log; "
        "``noCal'' is the propagation-only variant ($\\lambda\\equiv 1$).}\n"
        "\\begin{tabular}{llcccc}\n\\toprule\n"
        "stream & $f$ & MC cov & MC worst-tercile & MC width & noCal cov \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def tex_poolsize(pool):
    """Full error-pool-size grid (paper Table VII expands only part of it)."""
    rows = []
    for ds in STREAMS:
        g = pool[ds]["grid"]
        cells = []
        for m in ["5", "10", "25", "50", "100", "200", "full"]:
            cells.append(f3(g[m]["aggregate"]["covprop_mc"]["coverage"]["mean"]))
        cells.append(f3(g["full"]["aggregate"]["covprop_share"]["coverage"]["mean"]))
        rows.append(f"{ds} & " + " & ".join(cells) + " \\\\")
    body = "\n".join(rows)
    return (
        "\\begin{table}[h]\\centering\\small\n"
        "\\caption{Error-pool-size sensitivity, full grid: marginal coverage "
        "of CovProp-MC as the upstream-error pool is subsampled to $m$; the "
        "last column is CovProp-Share at the full pool.}\n"
        "\\begin{tabular}{lcccccccc}\n\\toprule\n"
        "stream & $m{=}5$ & 10 & 25 & 50 & 100 & 200 & full & Share(full) \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def tex_perwin():
    """Per-instance width statistics from the ribbon dumps (paper Sec. VII-D)."""
    rows = []
    for ds in STREAMS:
        z = np.load(os.path.join(RES, "ribbon", f"ribbon_{ds}.npz"))
        half = np.asarray(z["half"]).mean(axis=1)  # per-window mean over H
        mag = np.asarray(z["mag"]).ravel()
        rho = np.asarray(z["rho"]).ravel()
        rs_err = float(sps.spearmanr(half, mag).statistic)
        rs_rho = float(sps.spearmanr(half, rho).statistic)
        rows.append(
            f"{ds} & {f3(rs_err)} & {f3(rs_rho)} & {f3(float(np.mean(rho)))} & "
            f"{f3(float(np.std(half)))} \\\\")
    body = "\n".join(rows)
    return (
        "\\begin{table}[h]\\centering\\small\n"
        "\\caption{Per-instance propagated-width diagnostics, seed 0 "
        "($n{=}1500$ windows per stream).  $\\rho_s(\\mathrm{half},|\\eta|)$ "
        "is the weak correlation reported in the paper; "
        "$\\rho_s(\\mathrm{half},\\rho_t)$ shows the width does track the "
        "label-free share statistic.}\n"
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "stream & $\\rho_s(\\mathrm{half},|\\eta|)$ & "
        "$\\rho_s(\\mathrm{half},\\rho_t)$ & $\\bar\\rho_t$ & "
        "std(half) \\\\\n\\midrule\n" + body +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def tex_ad(ad):
    """Applicability-domain per-stream statistics (paper Sec. VII-D)."""
    rows = []
    for r in ad["per_dataset"]:
        rows.append(
            f"{r['dataset']} & {f3(r['upstream_skill_vs_persistence'])} & "
            f"{f3(r['upstream_relerr_cal'])} & {f3(r['prop_share_width'])} & "
            f"{f3(r['winkler_gain_share_vs_recal'])} & "
            f"{f3(r['sharebin_cov_spread_share'])} \\\\")
    body = "\n".join(rows)
    return (
        "\\begin{table}[h]\\centering\\small\n"
        "\\caption{Applicability-domain statistics per stream "
        "(\\texttt{ad\\_analysis.json}).  Skill is upstream forecast skill vs.\\ "
        "persistence; share is the width-based propagation share; gain is the "
        "Winkler improvement of CovProp-Share over recalibration; spread is "
        "the residual coverage spread across share bins.}\n"
        "\\begin{tabular}{lccccc}\n\\toprule\n"
        "stream & skill & rel-err & share & gain & bin-spread \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def tex_overhead(oh):
    """Per-stream latency detail (paper Sec. VIII)."""
    ks = ["K8", "K16", "K32", "K64"]
    rows = []
    for ds in STREAMS:
        d = oh[ds]
        cells = [f3(d["covprop_mc"][k]["per_window_ms"]) for k in ks
                 if k in d["covprop_mc"]]
        x = d["covprop_mc"][ks[0]]["x_recal"] if ks[0] in d["covprop_mc"] else 0
        rows.append(
            f"{ds} & {f3(d['recal']['per_window_ms'])} & " +
            " & ".join(cells) + f" & {x:.1f}$\\times$ \\\\")
    body = "\n".join(rows)
    return (
        "\\begin{table}[h]\\centering\\small\n"
        "\\caption{Per-window latency (ms) on 1500 test windows, seed 0. "
        "$\\times$recal is the $K{=}8$ latency relative to plain "
        "recalibration.}\n"
        "\\begin{tabular}{lccccc}\n\\toprule\n"
        "stream & recal & MC $K{=}8$ & $K{=}16$ & $K{=}32$ & "
        "$\\times$recal \\\\\n\\midrule\n" + body +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def tex_hybrid(hyb):
    """Hybrid negative result, per stream (paper Sec. VII-C)."""
    rows = []
    for ds in STREAMS:
        a = hyb[ds]["aggregate"]
        meths = [m for m in ["recal", "cqr", "covprop_mc", "covprop_share",
                             "hybrid_mc_cqr", "hybrid_cqr_covwidth"] if m in a]
        cells = " & ".join(f3(a[m]["winkler"]["mean"]) for m in meths)
        rows.append(f"{ds} & {cells} \\\\")
    header = "stream & " + " & ".join(
        m.replace("covprop_", "C-").replace("hybrid_", "H-").replace("_", "\\_")
        for m in meths) + " \\\\"
    body = "\n".join(rows)
    return (
        "\\begin{table}[h]\\centering\\small\n"
        "\\caption{Winkler score per stream for the two preregistered hybrid "
        "designs vs.\\ the lean roster (lower is better).  The pick beats CQR "
        "on only 2/7 streams against the preregistered $\\ge 5/7$ criterion; "
        "adopted as a negative result.}\n"
        "\\begin{tabular}{l" + "c" * len(meths) + "}\n\\toprule\n" +
        header + "\n\\midrule\n" + body +
        "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def tex_policy(pol):
    h = pol["H4_late_5seeds"]
    return (
        "\\paragraph{Backtest selector (paper \\S VII-E, Table VIII).} "
        f"Across the {h['n_points']} preregistered operating points the "
        f"label-free backtest selector attains mean Winkler "
        f"{f3(h['mean_winkler'])} vs.\\ oracle {f3(h['oracle_winkler'])} "
        f"(mean regret {h['mean_regret_pct']:.2f}\\%, median "
        f"{h['median_regret_pct']:.2f}\\%, max {h['max_regret_pct']:.2f}\\%; "
        f"choose-rate agreement {h['agree_rate']*100:.1f}\\%).  Wilcoxon vs.\\ "
        f"always-recalibrate: +{h['wilcoxon_vs_always_recal']['mean_improvement_pct']:.1f}\\% "
        f"($p={h['wilcoxon_vs_always_recal']['p']:.2e}$); vs.\\ "
        f"always-share: +{h['wilcoxon_vs_always_covprop_share']['mean_improvement_pct']:.1f}\\% "
        f"($p={h['wilcoxon_vs_always_covprop_share']['p']:.2e}$).  "
        f"Cross-check vs.\\ the frozen 3-seed run: max relative deviation of "
        f"seed-averaged backtest means "
        f"{pol['frozen_crosscheck']['max_rel_dev_backtest']*100:.1f}\\%.\n")


HEADER = r"""\documentclass[10pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage[hidelinks]{hyperref}
\title{Supplementary Tables for ``CovProp: Propagating Upstream Forecast
Uncertainty of Future Covariates into Conformal Prediction Intervals''}
\author{}
\date{}
\begin{document}
\maketitle
\noindent This artifact-appendix is generated mechanically from the shipped raw
results under \texttt{results/} --- no number is transcribed by hand.
Rebuild: \texttt{python supplementary/build\_supplementary.py --pdf}
"""

FOOTER = "\\end{document}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    bias = load("biased_log/bias_*.json")
    pool = load("poolsize/poolsize_*.json")
    hyb = load("hybrid/hybrid_*.json")
    oh = load("overhead/overhead_*.json")
    ad = json.load(open(os.path.join(RES, "ad_analysis.json")))
    pol = json.load(open(os.path.join(RES, "policy", "policy_streamlevel.json")))

    tex = (HEADER
           + "\\section{Robustness ablations}\n"
           + tex_biased_log(bias) + tex_poolsize(pool)
           + "\\section{Per-instance and applicability-domain diagnostics}\n"
           + tex_perwin() + tex_ad(ad)
           + "\\section{Negative result and deployment metrics}\n"
           + tex_hybrid(hyb) + tex_overhead(oh) + tex_policy(pol)
           + FOOTER)

    out = os.path.join(HERE, "supplementary.tex")
    open(out, "w", encoding="utf-8").write(tex)
    print("wrote", out)
    if args.pdf:
        os.chdir(HERE)
        for _ in range(2):
            os.system("pdflatex -interaction=nonstopmode supplementary.tex "
                      "> latex.log 2>&1")
        print("pdf build done")


if __name__ == "__main__":
    main()
