"""Calibration baselines for T1 E2 (numpy/scipy only).

All baselines operate on volatility-normalized signed residuals
res = (y - yhat) / sd_hist, shape [n, H], and produce per-window
(center c, radius q) in normalized units; the prediction interval is
    [ yhat + (c - q) * sd,  yhat + (c + q) * sd ].

Online methods stream through test windows in time order and may use test
residuals only through MATURED feedback (a window issued at s matures once
s + PRED is fully observed) — the same no-leakage protocol as the rest of
the harness.

Implemented:
  SplitCPB : static split conformal (per-step quantiles, finite-sample corr.)
  ACI      : Adaptive Conformal Inference (Gibbs & Candes 2021), per-step
             quantile of a sliding pool at level 1 - alpha_t, with alpha_t
             updated from matured miscoverage.
  NexCP    : weighted CP for non-exchangeable data (Barber et al. 2023),
             time-decay weights rho^age over cal + matured test scores.
  BCACI    : bias-corrected ACI — per-step EMA recentering of matured signed
             residuals, ACI radius on the recentered absolute scores.
(Mondrian-RC lives in rc_cp.MondrianCP.)
"""
import numpy as np

from rc_cp import _corrected_quantile


class SplitCPB:
    name = "split_cp"

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def run(self, cal_res, test_res, cal_w, test_w, pred_len, **ctx):
        q = _corrected_quantile(np.abs(cal_res), self.alpha)
        m = len(test_res)
        return np.zeros((m, cal_res.shape[1])), np.tile(q, (m, 1))


class _OnlineBase:
    """Shared matured-feedback streaming machinery."""

    def __init__(self, alpha=0.1, pool=1000, refresh=10):
        self.alpha = alpha
        self.pool = pool
        self.refresh = refresh

    def run(self, cal_res, test_res, cal_w, test_w, pred_len, **ctx):
        H = cal_res.shape[1]
        m = len(test_res)
        centers = np.zeros((m, H))
        radii = np.zeros((m, H))
        self._init(cal_res)
        pending = []          # (mature_time, test_index)
        since = self.refresh  # force quantile computation on first window
        for i, s in enumerate(test_w):
            while pending and pending[0][0] <= s:
                _, j = pending.pop(0)
                self._feedback(test_res[j], centers[j], radii[j])
                since += 1
            if since >= self.refresh:
                self._recompute()
                since = 0
            centers[i], radii[i] = self._predict()
            pending.append((s + pred_len, i))
        return centers, radii

    # hooks ------------------------------------------------------------
    def _init(self, cal_res):
        raise NotImplementedError

    def _feedback(self, res, c, q):
        raise NotImplementedError

    def _recompute(self):
        raise NotImplementedError

    def _predict(self):
        raise NotImplementedError


class ACI(_OnlineBase):
    name = "aci"

    def __init__(self, alpha=0.1, gamma=0.01, **kw):
        super().__init__(alpha, **kw)
        self.gamma = gamma

    def _init(self, cal_res):
        self.scores = list(np.abs(cal_res))
        self.alpha_t = self.alpha
        self.q = None

    def _feedback(self, res, c, q):
        err = float((np.abs(res - c) > q).mean())   # matured miscoverage
        self.alpha_t = float(np.clip(
            self.alpha_t + self.gamma * (self.alpha - err), 1e-3, 1 - 1e-3))
        self.scores.append(np.abs(res))
        if len(self.scores) > self.pool:
            self.scores.pop(0)

    def _recompute(self):
        s = np.asarray(self.scores)
        lvl = min(1.0, np.ceil((len(s) + 1) * (1 - self.alpha_t)) / len(s))
        self.q = np.quantile(s, lvl, axis=0, method="higher")

    def _predict(self):
        if self.q is None:
            self._recompute()
        return np.zeros_like(self.q), self.q


class NexCP(_OnlineBase):
    """Weighted split-CP with exponential time-decay weights."""
    name = "nexcp"

    def __init__(self, alpha=0.1, rho=0.999, **kw):
        super().__init__(alpha, **kw)
        self.rho = rho

    def _init(self, cal_res):
        self.scores = list(np.abs(cal_res))
        self.q = None

    def _feedback(self, res, c, q):
        self.scores.append(np.abs(res))
        if len(self.scores) > self.pool:
            self.scores.pop(0)

    def _recompute(self):
        s = np.asarray(self.scores)                       # [n, H]
        n = len(s)
        w = self.rho ** np.arange(n - 1, -1, -1)          # newest heaviest
        wn = w / (w.sum() + 1.0)                          # +1: test point mass
        order = np.argsort(s, axis=0)
        cw = np.cumsum(wn[order], axis=0)
        idx = np.argmax(cw >= (1 - self.alpha), axis=0)
        hit = cw[-1] >= (1 - self.alpha)
        idx = np.where(hit, idx, n - 1)                   # else max score
        self.q = np.take_along_axis(s, order, 0)[idx, np.arange(s.shape[1])]

    def _predict(self):
        if self.q is None:
            self._recompute()
        return np.zeros_like(self.q), self.q


class BCACI(ACI):
    """ACI + per-step EMA bias recentering of matured signed residuals."""
    name = "bc_aci"

    def __init__(self, alpha=0.1, gamma=0.01, beta=0.99, **kw):
        super().__init__(alpha, gamma, **kw)
        self.beta = beta

    def _init(self, cal_res):
        self.bias = cal_res.mean(axis=0)
        self.scores = list(np.abs(cal_res - self.bias[None, :]))
        self.alpha_t = self.alpha
        self.q = None

    def _feedback(self, res, c, q):
        err = float((np.abs(res - c) > q).mean())
        self.alpha_t = float(np.clip(
            self.alpha_t + self.gamma * (self.alpha - err), 1e-3, 1 - 1e-3))
        self.bias = self.beta * self.bias + (1 - self.beta) * res
        self.scores.append(np.abs(res - self.bias))
        if len(self.scores) > self.pool:
            self.scores.pop(0)

    def _predict(self):
        if self.q is None:
            self._recompute()
        return self.bias.copy(), self.q


class RareCP:
    """Simplified RareCP — error-regime-conditioned CP (Sesia-style regime
    retrieval, arXiv 2605.08857 lineage). At each test window we retrieve the
    k calibration/matured windows with the most similar *observable error
    regime* (a scalar regime key known at prediction time — here the target
    history volatility, which sets the residual scale; no target leakage) and
    take the finite-sample-corrected quantile over their conformity scores.

    This is an OUTPUT-SIDE, a-posteriori conditioner (regime is inferred from
    accumulated errors), included as the strongest "condition after the fact"
    contrast to RC-CP's input-side score.
    """
    name = "rarecp"

    def __init__(self, alpha=0.1, k=200):
        self.alpha = alpha
        self.k = k

    def run(self, cal_res, test_res, cal_w, test_w, pred_len,
            key_c=None, key_t=None, **ctx):
        H = cal_res.shape[1]
        m = len(test_res)
        scores = list(np.abs(cal_res))
        keys = list(np.asarray(key_c, dtype=float))
        cal_scale = np.std(keys) + 1e-6
        radii = np.zeros((m, H))
        pending = []
        for i, s in enumerate(test_w):
            while pending and pending[0][0] <= s:
                _, j = pending.pop(0)
                scores.append(np.abs(test_res[j]))
                keys.append(float(key_t[j]))
            S = np.asarray(scores)
            K = np.asarray(keys)
            d = np.abs(K - key_t[i]) / cal_scale
            k = min(self.k, len(S))
            idx = np.argpartition(d, k - 1)[:k]
            radii[i] = _corrected_quantile(S[idx], self.alpha)
            pending.append((s + pred_len, i))
        return np.zeros((m, H)), radii


def split_cp_from_residuals(cal_res, test_res, alpha=0.1):
    """Split-CP quantiles for an arbitrary residual stream (used by the
    Exogenous-Dropout baseline, whose residuals come from a separately
    dropout-trained point-robust model)."""
    q = _corrected_quantile(np.abs(cal_res), alpha)
    m = len(test_res)
    return np.zeros((m, cal_res.shape[1])), np.tile(q, (m, 1))


def winkler(y, lo, hi, alpha):
    """Mean Winkler (interval) score."""
    w = hi - lo
    w = w + (2.0 / alpha) * np.where(y < lo, lo - y, 0.0)
    w = w + (2.0 / alpha) * np.where(y > hi, y - hi, 0.0)
    return float(w.mean())
