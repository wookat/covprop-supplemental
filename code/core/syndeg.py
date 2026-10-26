"""SynDeg: synthetic controlled-degradation dataset for T1 (E1 main testbed).

1D target + 2 synthetic covariates, hourly, length 20k, same array interface
as data.load_market: [T, 3] with columns [cov1, cov2, price].

Generative model
  cov1 ("load-like")  : daily + weekly seasonality + slow trend + AR(1) noise
  cov2 ("wind-like")  : slowly-mixing AR(2) process, no seasonality
  price               : nonlinear function of (cov1, cov2) + own daily cycle
                        + AR(1) innovation noise
The target genuinely depends on both covariates, so corrupting the observed
future covariates (missing / noise / stale / lag / contamination via
data.corrupt) degrades the forecaster in a controlled way with ground-truth
degradation timing — exactly what E1 needs.
"""
import numpy as np

T_DEFAULT = 20_000


def ar1(n, phi, sigma, rng):
    e = np.zeros(n)
    z = rng.normal(0, sigma, n)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + np.sqrt(1 - phi ** 2) * z[t]
    return e


def generate(T=T_DEFAULT, seed=7):
    rng = np.random.RandomState(seed)
    t = np.arange(T)
    day = 2 * np.pi * t / 24.0
    week = 2 * np.pi * t / (24.0 * 7)

    cov1 = (1.0 * np.sin(day - 0.4) + 0.5 * np.sin(week)
            + 0.3 * np.sin(2 * day + 0.9)
            + 0.0001 * t / 24.0
            + ar1(T, 0.95, 0.35, rng))

    w = np.zeros(T)
    z = rng.normal(0, 1.0, T)
    for i in range(2, T):
        w[i] = 1.6 * w[i - 1] - 0.65 * w[i - 2] + z[i]
    cov2 = 0.4 * w / w.std()

    price = (0.9 * cov1 + 0.6 * np.tanh(1.5 * cov2) - 0.35 * cov1 * cov2
             + 0.5 * np.sin(day + 0.2)
             + ar1(T, 0.8, 0.30, rng))

    a = np.stack([cov1, cov2, price], axis=1).astype(np.float32)
    return a
