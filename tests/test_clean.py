"""Cleaning layer: tau_cal / clock regime around the switch, exact w, underlying gaps, clock weights."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from audit import io
from audit import options as aopt
from clean import clock, panel, underlying

ROOT = Path(__file__).resolve().parents[1]
CFG = io.load_config(ROOT / "config" / "clean.yaml")
YEAR_S = CFG["conventions"]["year_days"] * 86400
SWITCH = pd.Timestamp(CFG["clock"]["switch_time"])
EXPIRY = pd.Timestamp("2025-07-25 08:00", tz="UTC")


def synthetic_day(label_hours: list[str], tau_shift_s: float = 0.0) -> pd.DataFrame:
    """One expiry, 3 strikes, call+put, a fresh snapshot at hh:59:59 for each label, plus one atm pair."""
    rows = []
    for lab in label_hours:
        ts = pd.Timestamp(lab, tz="UTC")
        obs = ts + pd.Timedelta("59min59.5s")
        tau = ((EXPIRY - obs).total_seconds() + tau_shift_s) / YEAR_S
        F = 100_000.0
        for key, K in [("strike", 90_000.0), ("strike", 100_000.0), ("strike", 110_000.0), ("atm", F)]:
            iv = 0.5 + 0.1 * abs(np.log(K / F))
            for opt in ["call", "put"]:
                rows.append({"timestamp": ts, "expiry": EXPIRY, "expiry_label": "25JUL25", "strike": K, "option_type": opt,
                             "raw_timestamp": obs.value / 1e6, "surface_key_type": key, "implied_vol": iv, "tau_years": tau,
                             "forward_price": F, "delta": 0.5 if opt == "call" else -0.5, "is_stale": False,
                             "mid": 1.0, "r_squared": 0.99, "vol_curve_check": "GOOD"})
    return pd.DataFrame(rows)


def run(df):
    df = aopt.prepare(df)
    es, _ = aopt.expiry_snapshots(df, aopt.new_state(), panel.audit_cfg(CFG))
    return panel.clean_day(df, es, df.timestamp.min().strftime("%Y-%m-%d"), CFG), df


@pytest.mark.parametrize("labels,regime", [
    (["2025-06-27 21:00", "2025-06-27 22:00", "2025-06-27 23:00"], "calendar"),
    (["2025-06-28 00:00", "2025-06-28 01:00"], "vol_time"),
])
def test_tau_cal_both_sides_of_switch(labels, regime):
    shift = 0.0 if regime == "calendar" else 5 * 3600.0  # post-switch tau_years deliberately off calendar
    (curves, _, _, _), _ = run(synthetic_day(labels, tau_shift_s=shift))
    expect = (curves.expiry - curves.obs_time).dt.total_seconds() / YEAR_S
    assert np.allclose(curves.tau_cal, expect, rtol=0, atol=1e-15)
    assert (curves.clock_regime == regime).all()
    if regime == "calendar":
        assert curves.tau_vol.isna().all()
    else:
        assert np.allclose(curves.tau_vol * YEAR_S, curves.tau_cal * YEAR_S + shift)


def test_pre_switch_disagreement_is_flagged_anomaly():
    (curves, _, _, _), _ = run(synthetic_day(["2025-05-03 14:00"], tau_shift_s=120.0))
    assert (curves.clock_regime == "calendar_anomaly").all() and curves.tau_vol.notna().all()


def test_switch_boundary_label():
    ts = pd.Series(pd.to_datetime(["2025-06-27 23:00", "2025-06-28 00:00"], utc=True))
    tau = pd.Series([0.1, 0.1])
    assert panel.clock_regime(ts, tau, tau, CFG).tolist() == ["calendar", "vol_time"]


def test_w_exact_and_no_raw_iv_synthetic():
    (curves, points, _, _), df = run(synthetic_day(["2025-06-28 03:00"], tau_shift_s=3600.0))
    raw = df[(df.option_type == "call") & ~df.is_atm].set_index(["timestamp", "expiry", "strike"])
    p = points.set_index(["timestamp", "expiry", "strike"])
    assert (p.w == (raw.implied_vol ** 2 * raw.tau_years).reindex(p.index)).all()
    atm = df[df.is_atm & (df.option_type == "call")].iloc[0]
    assert curves.atm_w.iloc[0] == atm.implied_vol ** 2 * atm.tau_years
    assert "implied_vol" not in curves.columns and "implied_vol" not in points.columns
    assert len(points) == 3  # listed strikes only, call/put collapsed, atm excluded


@pytest.mark.parametrize("date", ["2025-06-27", "2025-06-28"])
def test_w_exact_on_raw_data(date):
    df, _, _ = io.read_day(ROOT / CFG["paths"]["raw_root"], "options", CFG["asset"], date)
    if df is None:
        pytest.skip("raw data not available")
    (curves, points, _, _), d = run(df)
    calls = d[(d.option_type == "call") & ~d.is_atm].set_index(["timestamp", "expiry", "strike"])
    p = points.set_index(["timestamp", "expiry", "strike"])
    assert (p.w.to_numpy() == (calls.implied_vol ** 2 * calls.tau_years).reindex(p.index).to_numpy()).all()


def test_underlying_zero_run_and_gap_span():
    ts = pd.date_range("2025-04-01", periods=12, freq="h", tz="UTC")
    close = [100, 101, 102, 102, 102, 102, 103, np.nan, np.nan, 104, 104, 105.0]
    raw = pd.DataFrame({"timestamp": ts, "close": close}).dropna()
    cfg = {"underlyings": {"zero_run_min_h": 3}}
    out, stats = underlying.build(raw, "2025-04-01", "2025-04-01", cfg)
    out = out.iloc[:12]
    assert out.in_zero_run.tolist()[3:6] == [True, True, True] and not out.in_zero_run.iloc[10]  # run of 1 kept
    assert out.missing_reason.tolist()[7:9] == ["absent", "absent"]
    r6 = out.iloc[6]
    assert r6.ret_span_h == 4 and r6.log_ret == pytest.approx(np.log(103 / 102))  # across the masked run
    r9 = out.iloc[9]
    assert r9.ret_span_h == 3 and r9.log_ret == pytest.approx(np.log(104 / 103))  # across the absent gap
    assert pd.isna(out.log_ret.iloc[7]) and stats["zero_run_hours_masked"] == 3


def test_clock_recovers_weights_per_dst_regime():
    """Synthetic clock: a block fixed in UTC plus a US-session block that moves one UTC hour with US DST."""
    def true_weights(regime):
        w = np.full(168, 0.7)
        hr = np.arange(168) % 24
        w[hr == 1] = 1.5                                         # fixed in UTC
        start = 13 if regime == "edt" else 14                    # US block moves with DST
        w[(hr >= start) & (hr < start + 7)] = 1.8
        return w
    ts = pd.date_range("2025-10-13", periods=168 * 6, freq="h", tz="UTC")  # spans the 2025-11-02 DST change
    reg = clock.dst_regime(pd.Series(ts), CFG["clock"]["local_tz"]).to_numpy()
    how = clock.hour_of_week(pd.Series(ts)).to_numpy()
    tw = {r: true_weights(r) for r in ["edt", "est"]}
    step = np.array([tw[r][h] for r, h in zip(reg, how)])
    vol_h = np.cumsum(step)
    rows = []
    for exp_days in [40, 80]:
        tau_h = exp_days * 24 + vol_h[-1] - vol_h + 1000
        rows.append(pd.DataFrame({"timestamp": ts, "obs_time": ts + pd.Timedelta("59min59s"),
                                  "expiry": pd.Timestamp("2026-01-30", tz="UTC") + pd.Timedelta(days=exp_days),
                                  "tau_vol": tau_h / (CFG["conventions"]["year_days"] * 24), "clock_regime": "vol_time",
                                  "tenor_days_cal": 30.0, "flag_late_snapshot": False, "flag_short_tenor": False,
                                  "flag_low_r2": False, "vendor_check": "GOOD"}))
    tab, dates, stats = clock.estimate(pd.concat(rows, ignore_index=True), CFG)
    assert np.allclose(tab.weight_edt, tw["edt"], atol=1e-9) and np.allclose(tab.weight_est, tw["est"], atol=1e-9)
    assert stats["frac_resid_gt_1e-3"] == 0 and stats["hours_differing_edt_est"] == 7 * 2  # each day: block start and end hour differ


def test_flagged_curves_are_kept_with_flags():
    """Former deletion rules 1b/3a/3b/3c now only flag: the curves stay in curves and points."""
    df = synthetic_day(["2025-07-10 03:00", "2025-07-10 04:00", "2025-07-10 05:00", "2025-07-10 06:00"])
    late = df.timestamp == pd.Timestamp("2025-07-10 04:00", tz="UTC")
    df.loc[late, "raw_timestamp"] -= 30 * 60 * 1000          # snapshot 30 min before bar end
    df.loc[df.timestamp == pd.Timestamp("2025-07-10 05:00", tz="UTC"), "r_squared"] = 0.5
    df.loc[df.timestamp == pd.Timestamp("2025-07-10 06:00", tz="UTC"), "vol_curve_check"] = "IV_RV_DIFF"
    (curves, points, pairs, ledger), _ = run(df)
    c = curves.set_index(curves.timestamp.dt.hour)
    assert len(curves) == 4 and points.groupby(["timestamp", "expiry"]).ngroups == 4
    assert c.flag_late_snapshot.tolist() == [False, True, False, False]
    assert c.flag_low_r2.tolist() == [False, False, True, False]
    assert c.vendor_check.tolist() == ["GOOD", "GOOD", "GOOD", "IV_RV_DIFF"]
    assert c.vendor_check_class.tolist() == ["GOOD", "GOOD", "GOOD", "value_screen"]
    assert not c.flag_short_tenor.any()
    assert "pass_research_default" not in curves.columns
    led = pd.DataFrame(ledger).set_index("rule")
    assert set(led.action) == {"delete", "flag"}
    assert led.loc["3c_vendor_check=IV_RV_DIFF", "curves_flagged"] == 1
    assert led.loc[f"1b_flag_late_snapshot_gt_{CFG['rules']['max_age_min']:g}min", "curves_flagged"] == 1


def test_short_tenor_flag_uses_calendar_tenor():
    near = synthetic_day(["2025-07-24 20:00"])                 # expiry 2025-07-25 08:00 -> ~11 h left
    (curves, _, _, _), _ = run(near)
    assert curves.flag_short_tenor.all() and (curves.tenor_days_cal < CFG["rules"]["min_tenor_days"]).all()


def test_calendar_pairs_cover_all_combinations():
    """Three expiries at one label -> 3 pairs; adjacent pairs carry the per-curve n_calendar_viol."""
    frames = []
    for exp, iv_scale in [("2025-07-25 08:00", 1.0), ("2025-08-29 08:00", 0.5), ("2025-09-26 08:00", 1.0)]:
        d = synthetic_day(["2025-07-10 03:00"])
        d["expiry"] = pd.Timestamp(exp, tz="UTC")
        d["tau_years"] = (d.expiry - pd.to_datetime(d.raw_timestamp, unit="ms", utc=True)).dt.total_seconds() / YEAR_S
        d["implied_vol"] *= iv_scale                            # middle expiry has much lower total variance
        frames.append(d)
    (curves, _, pairs, _), _ = run(pd.concat(frames, ignore_index=True))
    assert len(pairs) == 3 and pairs.adjacent_in_panel.sum() == 2
    assert (pairs.expiry_short < pairs.expiry_long).all()
    adj = pairs[pairs.adjacent_in_panel].set_index("expiry_long").n_calendar_viol
    c = curves.set_index("expiry")
    assert (c.n_calendar_viol.reindex(adj.index) == adj).all()
    assert adj.iloc[0] > 0                                      # Aug has less total variance than the shorter Jul
    assert c.n_calendar_viol.iloc[0] == 0                       # shortest curve has no neighbour
