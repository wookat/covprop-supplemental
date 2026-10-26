"""T1 RC-CP: Reliability-Conditional Conformal Prediction — core routines.

Split-CP and Mondrian-CP (binned by a covariate-reliability score) prediction
intervals for multi-horizon forecasting.

Conformity score: per-step absolute residual |y_{t+h} - yhat_{t+h}|.
Quantiles are computed per horizon step h (24 quantiles for a 24h horizon),
with the standard finite-sample correction ceil((n+1)(1-alpha))/n.

Mondrian-CP: calibration windows are partitioned into K bins by the
reliability score r (bin edges = quantiles of calibration scores); each bin
gets its own per-step quantiles; a test window uses the quantiles of its bin.
"""
import numpy as np


def _corrected_quantile(scores, alpha):
    """Finite-sample-valid conformal quantile along axis 0. scores: [n, H]."""
    n = scores.shape[0]
    q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(scores, q, axis=0, method="higher")


class SplitCP:
    """Marginal split conformal: one set of per-step quantiles for everyone."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.qhat = None

    def calibrate(self, cal_scores):
        self.qhat = _corrected_quantile(cal_scores, self.alpha)  # [H]

    def interval(self, pred):
        """pred: [m, H] point forecasts -> (lo, hi) each [m, H]."""
        return pred - self.qhat[None, :], pred + self.qhat[None, :]


class MondrianCP:
    """Reliability-conditional conformal: separate quantiles per r-bin."""

    def __init__(self, alpha=0.1, n_bins=3, min_bin=50):
        self.alpha = alpha
        self.n_bins = n_bins
        self.min_bin = min_bin
        self.edges = None
        self.qhat = None  # [K, H]

    def calibrate(self, cal_scores, cal_r):
        probs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges = np.quantile(cal_r, probs)
        bins = np.digitize(cal_r, self.edges)
        H = cal_scores.shape[1]
        self.qhat = np.zeros((self.n_bins, H))
        fallback = _corrected_quantile(cal_scores, self.alpha)
        for k in range(self.n_bins):
            sk = cal_scores[bins == k]
            self.qhat[k] = _corrected_quantile(sk, self.alpha) if len(sk) >= self.min_bin else fallback

    def interval(self, pred, r):
        """pred: [m, H], r: [m] reliability scores."""
        q = self.qhat[np.digitize(r, self.edges)]  # [m, H]
        return pred - q, pred + q


def _weighted_quantile(scores, weights, level):
    """Weighted conformal quantile along axis 0 with an extra +inf point-mass
    of weight 1/(sum w + 1) for the test point (Barber et al. 2023). Returns
    the smallest score whose cumulative normalized weight reaches `level`;
    if the test point mass is needed, returns +inf (interval = whole line).
    scores: [n, H], weights: [n]."""
    n, H = scores.shape
    order = np.argsort(scores, axis=0)
    sw = weights[order]                           # [n, H]
    total = weights.sum() + 1.0
    cw = np.cumsum(sw, axis=0) / total
    idx = np.argmax(cw >= level, axis=0)          # first index reaching level
    reached = cw[-1] >= level
    s_sorted = np.take_along_axis(scores, order, 0)
    out = s_sorted[idx, np.arange(H)]
    return np.where(reached, out, np.inf)


class KernelCP:
    """Kernel-RC: reliability-conditional conformal via kernel-weighted
    quantiles (Barber et al. weighted-CP with a reliability kernel).

    Weights w_i = K((r_t - r_i) / h); Gaussian or Epanechnikov. Smooth
    alternative to Mondrian binning: no bin-edge artifacts, better behaved in
    sparse reliability regions. Bandwidth defaults to Silverman's rule on the
    calibration reliability scores.
    """

    def __init__(self, alpha=0.1, kernel="gaussian", bandwidth=None):
        self.alpha = alpha
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.cal_scores = None
        self.cal_r = None
        self.h = None

    def calibrate(self, cal_scores, cal_r):
        self.cal_scores = cal_scores
        self.cal_r = cal_r
        if self.bandwidth is not None:
            self.h = float(self.bandwidth)
        else:
            sd = cal_r.std()
            self.h = max(1.06 * sd * len(cal_r) ** (-1 / 5), 1e-3)  # Silverman

    def _weights(self, r):
        u = (r - self.cal_r) / self.h
        if self.kernel == "epanechnikov":
            w = np.where(np.abs(u) <= 1, 0.75 * (1 - u ** 2), 0.0)
        else:
            w = np.exp(-0.5 * u ** 2)
        return w

    def interval(self, pred, r):
        """pred: [m, H], r: [m]."""
        n = len(self.cal_scores)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        m, H = pred.shape
        q = np.zeros((m, H))
        for i in range(m):
            w = self._weights(r[i])
            if w.sum() < 1e-8:
                w = np.ones_like(w)
            q[i] = _weighted_quantile(self.cal_scores, w, level)
        big = np.isinf(q)
        if big.any():                              # fall back to marginal q
            q_marg = _corrected_quantile(self.cal_scores, self.alpha)
            q = np.where(big, q_marg[None, :], q)
        return pred - q, pred + q


def coverage_report(y, lo, hi, r, edges, label=""):
    """Per-window coverage (fraction of the 24 steps inside the band), overall
    and stratified by reliability bins defined by `edges`."""
    inside = ((y >= lo) & (y <= hi)).mean(axis=1)  # [m]
    width = (hi - lo).mean(axis=1)
    bins = np.digitize(r, edges)
    rep = {
        "method": label,
        "overall_coverage": float(inside.mean()),
        "overall_width": float(width.mean()),
        "by_reliability_bin": [],
    }
    for k in range(len(edges) + 1):
        m = bins == k
        rep["by_reliability_bin"].append({
            "bin": k,
            "n": int(m.sum()),
            "coverage": float(inside[m].mean()) if m.any() else None,
            "width": float(width[m].mean()) if m.any() else None,
        })
    return rep
