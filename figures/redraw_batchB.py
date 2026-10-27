"""P2 CovProp (ICDE 2027) figure remediation, batch B.

Redraws the four data figures at final print size so that all in-figure
text is >= 6pt after inclusion (guide v2 AP-2), adds line-style + marker
double coding (8.2.3), keeps the figstyle.py semantic palette.

Raw inputs (read-only):
  r1_results/teaser_syndeg.npz, r1_results/sensitivity_syndeg.json,
  r1_results/main_*.json, r3_results/real_*.json
Outputs -> oral_loop/r3_fix/paper_v4/figures/*_remed.pdf
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # tsf-2027
T2 = os.path.abspath(os.path.join(HERE, "..", ".."))
R1 = os.path.join(T2, "r1_results")
R3 = os.path.join(T2, "r3_results")
OUT = os.path.join(T2, "oral_loop", "r3_fix", "paper_v4", "figures")

sys.path.insert(0, ROOT)
from figstyle import (PALETTE, FigureStyle, apply_publication_style,
                      create_subplots, finalize_figure)

C = {
    "oracle": "#767676",
    "naive": PALETTE["red_strong"],
    "recal": PALETTE["red_2"],
    "exodrop": "#E9A6A1",
    "aci": PALETTE["teal"],
    "nexcp": "#7AB8C4",
    "cqr": PALETTE["violet"],
    "norcp": "#C08AB8",
    "spci": "#8A8A46",
    "rc_cp": PALETTE["green_3"],
    "covprop_mc": PALETTE["blue_main"],
    "covprop_nest_b0.02": PALETTE["blue_secondary"],
}
LS = {"oracle": ":", "naive": "-", "recal": "--", "covprop_mc": "-",
      "covprop_nest_b0.02": "-."}
MK = {"oracle": "s", "naive": "o", "recal": "^", "covprop_mc": "D",
      "covprop_nest_b0.02": "v"}
LBL = {
    "oracle": "oracle", "naive": "naive", "recal": "recal",
    "exodrop": "ExoDrop", "aci": "ACI", "nexcp": "NexCP", "cqr": "CQR",
    "norcp": "NorCP", "spci": "SPCI", "rc_cp": "RC-CP",
    "covprop_mc": "CovProp-MC (ours)",
    "covprop_nest_b0.02": "CovProp-Nest (ours)",
}
REAL = ["NP", "PJM", "BE", "FR", "DE", "ETTh1", "ETTm1", "weather",
        "electricity"]
STREAMS = ["NP", "FR", "ETTh1", "weather", "ETTh2", "ETTm2", "electricity"]


def fig0_teaser():
    tz = np.load(os.path.join(R1, "teaser_syndeg.npz"))
    sens = json.load(open(os.path.join(R1, "sensitivity_syndeg.json")))
    fig, axes = create_subplots(1, 2, figsize=(3.45, 1.32))
    ax = axes[0]
    mag = tz["mag"]
    order = np.argsort(mag)
    bins = np.array_split(order, 20)
    xs = [mag[b].mean() for b in bins]
    for key, m, lab in (("w_naive", "naive", "naive"),
                        ("w_recal", "recal", "recal"),
                        ("w_covprop", "covprop_mc", "CovProp-MC")):
        ax.plot(xs, [tz[key][b].mean() for b in bins], marker=MK[m], ms=2.4,
                lw=1.1, ls=LS[m], label=lab, color=C[m], markevery=3)
    ax.set_xlabel("upstream error $|\\eta|$ (binned)")
    ax.set_ylabel("interval width")
    
    ax = axes[1]
    sigmas = sorted(float(s) for s in sens)
    for m in ("naive", "recal", "covprop_mc"):
        ax.plot(sigmas, [sens[str(s)][m]["coverage"]["mean"] for s in sigmas],
                marker=MK[m], ms=2.8, lw=1.1, ls=LS[m], label=LBL[m],
                color=C[m])
    ax.axhline(0.9, color="black", ls="--", lw=0.8)
    ax.set_xlabel("upstream error scale $\\sigma$")
    ax.set_ylabel("marginal coverage")
    ax.legend(fontsize=6.5, handlelength=1.5, handletextpad=0.4,
              loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, frameon=False, columnspacing=0.7)
    fig.tight_layout(pad=0.4)
    finalize_figure(fig, os.path.join(OUT, "fig0_teaser_remed"),
                    formats=["pdf"])


def fig1_collapse():
    sens = json.load(open(os.path.join(R1, "sensitivity_syndeg.json")))
    sigmas = sorted(float(s) for s in sens)
    fig, axes = create_subplots(1, 1, figsize=(2.35, 1.4))
    ax = axes[0]
    for m in ("oracle", "naive", "recal", "covprop_mc",
              "covprop_nest_b0.02"):
        if m in sens[str(sigmas[0])]:
            ax.plot(sigmas,
                    [sens[str(s)][m]["coverage"]["mean"] for s in sigmas],
                    marker=MK[m], ms=2.6, lw=1.1, ls=LS[m],
                    label=LBL.get(m, m), color=C.get(m))
    ax.axhline(0.9, color="black", ls="--", lw=0.8)
    ax.set_xlabel("upstream error scale $\\sigma$")
    ax.set_ylabel("marginal coverage")
    ax.set_xticks(sigmas)
    ax.legend(fontsize=6.3, handlelength=1.5, labelspacing=0.3,
              loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=2, frameon=False, columnspacing=0.9,
              handletextpad=0.4)
    fig.tight_layout(pad=0.4)
    finalize_figure(fig, os.path.join(OUT, "fig1_collapse_curve_remed"),
                    formats=["pdf"])


def fig2_conditional():
    rep = json.load(open(os.path.join(R1, "main_SynDeg.json")))["aggregate"]
    ms = [m for m in ("naive", "recal", "exodrop", "aci", "nexcp", "cqr",
                      "norcp", "spci", "rc_cp", "covprop_mc") if m in rep]
    fig, axes = create_subplots(1, 1, figsize=(1.9, 1.3))
    ax = axes[0]
    y = np.arange(len(ms))[::-1]
    terc = [(0, "low $|\eta|$", PALETTE["green_3"], "o"),
            (1, "mid $|\eta|$", PALETTE["teal"], "^"),
            (2, "high $|\eta|$", PALETTE["red_strong"], "D")]
    for j, lab, col, mk in terc:
        ax.scatter([rep[m]["cov_by_noise_tercile"][j] for m in ms], y,
                   s=11, color=col, marker=mk, label=lab, zorder=3,
                   edgecolors="black", linewidths=0.3)
    ax.axvline(0.9, color="black", ls="--", lw=0.8)
    ax.set_yticks(y)
    names = [LBL[m] for m in ms]
    names = [n.replace(" (ours)", "*") for n in names]
    ax.set_yticklabels(names, fontsize=6.0)
    ax.set_xlabel("conditional coverage", fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_xlim(0.35, 1.03)
    ax.legend(fontsize=6.0, loc="lower center", ncol=3,
              bbox_to_anchor=(0.5, 1.0), frameon=False,
              handlelength=1.0, handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout(pad=0.3)
    finalize_figure(fig, os.path.join(OUT, "fig2_conditional_coverage_remed"),
                    formats=["pdf"])


def fig3_width_decomp():
    tags, endo, prop = [], [], []
    for ds in ["SynDeg"] + REAL:
        fp = os.path.join(R1, f"main_{ds}.json")
        if os.path.exists(fp):
            v = json.load(open(fp))["aggregate"]["covprop_mc"]
            if "width_endo" in v:
                tags.append(ds)
                endo.append(v["width_endo"])
                prop.append(v["width_prop"])
    fig, axes = create_subplots(1, 1, figsize=(1.5, 1.3))
    ax = axes[0]
    x = np.arange(len(tags))
    e, p = np.array(endo), np.array(prop)
    share = p / (e + p) * 100
    ax.bar(x, e / (e + p), 0.6, label="endogenous residual",
           color=PALETTE["blue_secondary"], edgecolor="black", linewidth=0.35)
    ax.bar(x, p / (e + p), 0.6, bottom=e / (e + p),
           label="covariate propagation", color=PALETTE["green_3"],
           edgecolor="black", linewidth=0.35)
    for i in range(len(tags)):
        ax.text(x[i], 1.03, f"{share[i]:.0f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=90, fontsize=7)
    ax.set_ylabel("width share")
    ax.set_ylim(0, 1.0)
    fig.tight_layout(pad=0.4)
    finalize_figure(fig, os.path.join(OUT, "fig3_width_decomposition_remed"),
                    formats=["pdf"])


def fig_r3_share():
    fig, axes = create_subplots(1, len(STREAMS), figsize=(7.0, 1.22),
                                sharey=True)
    x = np.arange(3)
    for ax, ds in zip(axes, STREAMS):
        r = json.load(open(os.path.join(R3, f"real_{ds}.json")))
        cu = r["cov_by_share_tercile"]["covprop_mc"]
        cs = r["cov_by_share_tercile"]["covprop_share"]
        ax.bar(x - 0.19, cu, 0.38, color="#9A9A9A",
               label="uniform $\\lambda$", edgecolor="black", linewidth=0.3)
        ax.bar(x + 0.19, cs, 0.38, color=PALETTE["blue_main"],
               label="share-binned $\\lambda$", edgecolor="black",
               linewidth=0.3)
        ax.axhline(0.9, ls="--", lw=0.8, color="black")
        ax.set_title(ds, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["low", "mid", "high"], fontsize=7)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_ylim(0.6, 1.0)
    axes[0].set_ylabel("coverage", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", bbox_to_anchor=(1.0, 1.13),
               ncol=2, fontsize=7, frameon=False, handlelength=1.4)
    fig.supxlabel("propagation-share tercile", fontsize=8)
    fig.tight_layout(pad=0.4)
    finalize_figure(fig, os.path.join(OUT,
                    "fig_r3_share_conditional_coverage_remed"),
                    formats=["pdf"])


if __name__ == "__main__":
    apply_publication_style(FigureStyle(font_size=8, axes_linewidth=0.75))
    fig0_teaser()
    fig1_collapse()
    fig2_conditional()
    fig3_width_decomp()
    fig_r3_share()
    print("done ->", OUT)
