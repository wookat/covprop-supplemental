"""Deployment architecture figure for the system section (fig_arch.pdf)."""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(3.5, 2.35))
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")


def box(x, y, w, h, text, fc="#eef3fb", ec="#3b6ea5", fs=7.2, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs)


def arrow(p, q, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8,
                                 color="#444444", lw=0.9,
                                 shrinkA=1, shrinkB=1, **kw))


# top row: data path
box(0.1, 5.3, 2.6, 1.3, "upstream\nforecaster")
box(3.7, 5.3, 2.6, 1.3, "downstream\nforecaster $\\hat f$ (frozen)")
box(7.3, 5.3, 2.6, 1.3, "interval\n$[\\mathrm{mid}\\pm\\lambda\\,\\mathrm{half}]$",
    fc="#e8f6e8", ec="#2e7d32")
arrow((2.7, 5.95), (3.7, 5.95))
ax.text(3.2, 6.28, "$x^{\\mathrm{obs}}_t$", fontsize=7.0, ha="center")
arrow((6.3, 5.95), (7.3, 5.95))
ax.text(6.8, 6.28, "$K$ samples", fontsize=7.0, ha="center")

# middle row: metadata path
box(0.1, 2.9, 2.6, 1.3, "error log\n(forecast$-$realization)", fc="#fdf3e3",
    ec="#b26a00")
box(3.7, 2.9, 2.6, 1.3, "error pool $\\hat F_t$\n($\\sim$0.3--0.6 MB)",
    fc="#fdf3e3", ec="#b26a00")
box(7.3, 2.9, 2.6, 1.3, "stage-2 $\\lambda$ / $\\lambda_b$\n(matured feedback)",
    fc="#fdf3e3", ec="#b26a00")
arrow((1.4, 5.3), (1.4, 4.2))
arrow((2.7, 3.55), (3.7, 3.55))
arrow((5.0, 4.2), (5.0, 5.3))
ax.text(5.55, 4.6, "MC draws", fontsize=7.0, ha="center")
arrow((8.6, 4.2), (8.6, 5.3))

# bottom row: monitoring / decision
box(2.0, 0.4, 6.0, 1.3,
    "observability: share $\\rho_t$, width decomposition,\n"
    "calibration-split backtest $\\Rightarrow$ deploy CovProp vs.\\ recal",
    fc="#f3e9f7", ec="#6a3d9a")
arrow((8.6, 2.9), (7.4, 1.7))
arrow((4.2, 2.9), (4.2, 1.7))
fig.tight_layout(pad=0.2)
fig.savefig("fig_arch.pdf")
print("wrote fig_arch.pdf")
