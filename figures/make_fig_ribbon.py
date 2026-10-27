# Qualitative interval-ribbon showcase: real upstream-error stream (weather),
# 200-window segment where naive coverage collapses and CovProp-MC restores it.
# Data: ribbon_weather.npz dumped by experiments/dump_ribbon.py (seed 0,
# real-stream protocol). Top: realized series + naive/CovProp bands + naive
# misses (red dots); bottom strip: |eta_t| (unlabeled upstream error) with the
# top-tercile threshold -- misses cluster where upstream error is large.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F"}
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})

d = np.load("../pagecheck/ribbon_weather.npz")
Y, ts, mag = d["Y"], d["ts"], d["mag"]
H = 23                      # 24-step-ahead forecast (matured horizon)
i0, W = 1290, 200           # segment maximizing (naive miss & hi-|eta| &
                            # CovProp cover) events
sl = slice(i0, i0 + W)
x = ts[sl] + H              # realized timestamp of the h=24 prediction
y = Y[sl, H]
nlo, nhi = d["iv_naive"][0][sl, H], d["iv_naive"][1][sl, H]
clo, chi = d["iv_covprop_mc"][0][sl, H], d["iv_covprop_mc"][1][sl, H]
m = mag[sl]
ter = np.quantile(mag, 2/3)  # top-tercile |eta| threshold (calibrated globally)

fig, (ax, axm) = plt.subplots(
    2, 1, figsize=(3.45, 1.10), sharex=True,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})

ax.fill_between(x, clo, chi, color=OI["blue"], alpha=0.30, lw=0,
                label="CovProp $C_t$", zorder=1)
ax.fill_between(x, nlo, nhi, color=OI["grey"], alpha=0.45, lw=0,
                label="naive $C_t$", zorder=2)
ax.plot(x, nlo, color=OI["grey"], lw=0.5, ls="--", zorder=2)
ax.plot(x, nhi, color=OI["grey"], lw=0.5, ls="--", zorder=2)
ax.plot(x, y, color="black", lw=1.0, label="realized $y$", zorder=3)
miss = (y < nlo) | (y > nhi)
ax.scatter(x[miss], y[miss], s=5, color=OI["red"], zorder=4,
           label="naive miss")
ax.set_ylabel("$y_{t+24}$")
ax.legend(ncol=4, loc="upper left", frameon=False,
          handlelength=1.1, columnspacing=0.8, borderaxespad=0.1)
ax.set_xlim(x[0], x[-1])
ax.text(0.995, 0.03, f"segment coverage: naive {1 - miss.mean():.2f}, "
        f"CovProp {((y >= clo) & (y <= chi)).mean():.2f}",
        transform=ax.transAxes, fontsize=7, ha="right", va="bottom")

axm.fill_between(x, 0, m, color=OI["orange"], alpha=0.55, lw=0)
axm.axhline(ter, color=OI["red"], ls="--", lw=0.8)
axm.set_ylabel("$|\\eta_t|$", labelpad=2)
axm.set_xlabel("time $t$")
axm.set_ylim(0, np.quantile(m, 0.98) * 1.15)
axm.annotate("top tercile", (x[2], ter * 1.12), fontsize=7, color=OI["red"])

fig.align_ylabels()
fig.savefig("../paper/figures/fig_ribbon.pdf", bbox_inches="tight")
fig.savefig("../paper/figures/fig_ribbon.png", dpi=300, bbox_inches="tight")

print(f"seg naive cov={1 - miss.mean():.2f} "
      f"cp cov={((y >= clo) & (y <= chi)).mean():.2f} | "
      f"|eta|>tercile frac={(m > ter).mean():.2f} mean|eta|={m.mean():.3f}")
