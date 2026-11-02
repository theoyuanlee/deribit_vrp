"""Undiscounted Black-76 price and forward delta (vectorised)."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1d2(F, K, T, sigma):
    F, K, T, sigma = (np.asarray(x, dtype=float) for x in (F, K, T, sigma))
    sd = sigma * np.sqrt(np.maximum(T, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sd * sd) / sd
    return d1, d1 - sd, sd


def price(F, K, T, sigma, is_call):
    """Undiscounted Black-76. Falls back to intrinsic value when sigma*sqrt(T) == 0."""
    F, K = np.asarray(F, dtype=float), np.asarray(K, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)
    d1, d2, sd = _d1d2(F, K, T, sigma)
    call = F * norm.cdf(d1) - K * norm.cdf(d2)
    put = K * norm.cdf(-d2) - F * norm.cdf(-d1)
    px = np.where(is_call, call, put)
    intrinsic = np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    return np.where(sd > 0, px, intrinsic)


def delta(F, K, T, sigma, is_call):
    """Forward delta: N(d1) for calls, N(d1) - 1 for puts."""
    is_call = np.asarray(is_call, dtype=bool)
    d1, _, sd = _d1d2(F, K, T, sigma)
    nd1 = np.where(sd > 0, norm.cdf(d1), (np.asarray(F, float) > np.asarray(K, float)).astype(float))
    return np.where(is_call, nd1, nd1 - 1.0)
