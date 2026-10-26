"""EPF data pipeline + covariate corruption protocols (paper A)."""
import csv, os
import numpy as np

EPF_DIR = os.environ.get("EPF_DIR", os.path.expanduser("~/epf"))
MARKETS = ["NP", "PJM", "BE", "FR", "DE"]

def load_market(name):
    """Returns array [T, 3] with columns [cov1, cov2, price]. Hourly."""
    rows = []
    with open(os.path.join(EPF_DIR, f"{name}.csv")) as f:
        r = csv.reader(f); next(r)
        for row in r:
            rows.append([float(x) for x in row[1:]])
    a = np.asarray(rows, dtype=np.float32)
    return np.concatenate([a[:, 1:], a[:, :1]], axis=1)

def splits(n, tr=0.7, va=0.1):
    ntr = int(n * tr); nva = int(n * (tr + va))
    return ntr, nva, n

def normalize(data, ntr):
    mu = data[:ntr].mean(0); sd = data[:ntr].std(0) + 1e-5
    return (data - mu) / sd, mu, sd

# ---------------- corruption protocols (applied to future covariates cf: [H, n_cov]) --------
def corrupt(cf, mode, strength, rng):
    """mode: none|missing|noise|stale|lag ; strength in [0,1] (or hours for lag)."""
    c = cf.copy()
    n_cov = c.shape[1]
    if mode == "none":
        return c
    if mode == "missing":                      # each covariate dropped w.p. strength
        mask = rng.random(n_cov) < strength
        c[:, mask] = 0.0
        return c
    if mode == "noise":                        # additive gaussian, sigma = strength (in std units)
        return c + rng.normal(0, strength, c.shape).astype(np.float32)
    if mode == "stale":                        # freeze at first value for a strength-fraction of horizon
        h = int(round(len(c) * strength))
        if h > 0:
            c[-h:] = c[len(c) - h - 1 if len(c) > h else 0]
        return c
    if mode == "lag":                          # covariates shifted (delayed) by int(strength) steps
        k = int(strength)
        if k > 0:
            c[k:] = c[:-k]; c[:k] = c[0]
        return c
    if mode == "fcerr":                        # realistic forecast-error replay proxy:
        H = len(c)                             # AR(1)-correlated noise growing with horizon
        phi = 0.9
        for j in range(n_cov):
            e = np.zeros(H, dtype=np.float32); z = rng.normal(0, 1, H)
            for t in range(1, H):
                e[t] = phi * e[t - 1] + np.sqrt(1 - phi ** 2) * z[t]
            scale = strength * (0.3 + 0.7 * np.arange(H) / max(H - 1, 1))
            c[:, j] += (e * scale).astype(np.float32)
        return c
    raise ValueError(mode)

# canonical evaluation grid
CORRUPTIONS = [
    ("none", 0.0),
    ("missing", 0.5), ("missing", 1.0),
    ("noise", 0.5), ("noise", 1.0),
    ("stale", 0.5), ("stale", 1.0),
    ("lag", 6), ("lag", 24),
]

def make_windows(n, seq_len, pred_len, lo, hi):
    return list(range(max(lo, seq_len), hi - pred_len + 1))
