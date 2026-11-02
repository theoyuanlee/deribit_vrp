import math

import numpy as np
import pytest

from audit import black76


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def ref_call(F, K, T, s):
    sd = s * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sd * sd) / sd
    return F * ncdf(d1) - K * ncdf(d1 - sd)


rng = np.random.default_rng(0)
N = 500
F = rng.uniform(20_000, 150_000, N)
K = F * np.exp(rng.uniform(-1.0, 1.0, N))
T = rng.uniform(1e-4, 1.5, N)
S = rng.uniform(0.05, 3.0, N)


def test_matches_independent_reference():
    got = black76.price(F, K, T, S, np.ones(N, bool))
    ref = np.array([ref_call(*x) for x in zip(F, K, T, S)])
    assert np.allclose(got, ref, rtol=1e-10, atol=1e-8)


def test_put_call_parity_undiscounted():
    c = black76.price(F, K, T, S, np.ones(N, bool))
    p = black76.price(F, K, T, S, np.zeros(N, bool))
    assert np.allclose(c - p, F - K, rtol=0, atol=1e-7 * F.max())


def test_atm_closed_form():
    f, t, s = 95_000.0, 0.25, 0.6
    expected = f * (2 * ncdf(0.5 * s * math.sqrt(t)) - 1)
    assert black76.price(f, f, t, s, True) == pytest.approx(expected, rel=1e-12)


def test_zero_time_or_vol_is_intrinsic():
    f, k = np.array([100.0, 100.0, 100.0, 100.0]), np.array([90.0, 110.0, 90.0, 110.0])
    is_call = np.array([True, True, False, False])
    for t, s in [(0.0, 0.5), (0.5, 0.0)]:
        assert np.allclose(black76.price(f, k, t, s, is_call), [10.0, 0.0, 0.0, 10.0])


def test_monotone_in_vol_and_bounded():
    c_lo = black76.price(F, K, T, S, np.ones(N, bool))
    c_hi = black76.price(F, K, T, S * 1.1, np.ones(N, bool))
    assert (c_hi >= c_lo - 1e-9).all()
    assert (c_lo >= np.maximum(F - K, 0) - 1e-7).all() and (c_lo <= F + 1e-7).all()


def test_delta_put_call_and_finite_difference():
    dc = black76.delta(F, K, T, S, np.ones(N, bool))
    dp = black76.delta(F, K, T, S, np.zeros(N, bool))
    assert np.allclose(dc - dp, 1.0)
    h = F * 1e-6
    fd = (black76.price(F + h, K, T, S, True) - black76.price(F - h, K, T, S, True)) / (2 * h)
    assert np.allclose(fd, dc, atol=1e-5)
