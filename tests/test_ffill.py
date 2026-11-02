"""Forward-fill identification at the (timestamp, expiry) level on synthetic option frames."""
import numpy as np
import pandas as pd

from audit import options

CFG = {"snapshot": {"freshness_edges_min": [10, 60], "intra_label_spread_flag_s": 60}}
E_A = pd.Timestamp("2025-06-27 08:00", tz="UTC")
E_B = pd.Timestamp("2025-09-26 08:00", tz="UTC")
E_C = pd.Timestamp("2025-07-25 08:00", tz="UTC")


def ms(ts: str) -> float:
    return pd.Timestamp(ts, tz="UTC").value / 1e6


def rows(label: str, expiry: pd.Timestamp, snap: str, n: int = 4, stale: bool = False) -> list[dict]:
    """n strike rows (calls) for one expiry at one label, all sharing snapshot time `snap`."""
    ts = pd.Timestamp(label, tz="UTC")
    return [{"timestamp": ts, "expiry": expiry, "expiry_label": expiry.strftime("%d%b%y").upper(), "strike": 90_000.0 + 1000 * i,
             "raw_timestamp": ms(snap), "is_stale": stale, "tau_years": 0.1, "surface_key_type": "strike",
             "mid": np.nan if stale else 100.0, "implied_vol": 0.5} for i in range(n)]


def day1() -> pd.DataFrame:
    d = "2025-05-01"
    r = []
    r += rows(f"{d} 02:00", E_A, f"{d} 02:59:59.100") + rows(f"{d} 02:00", E_B, f"{d} 02:59:59.400")
    # 03: both copied from 02, not yet stale
    r += rows(f"{d} 03:00", E_A, f"{d} 02:59:59.100") + rows(f"{d} 03:00", E_B, f"{d} 02:59:59.400")
    # 04: A fresh, B copied and marked stale -> mixed
    r += rows(f"{d} 04:00", E_A, f"{d} 04:59:58.900") + rows(f"{d} 04:00", E_B, f"{d} 02:59:59.400", stale=True)
    # 05: A, B fresh; C appears for the first time while previous label had data
    r += (rows(f"{d} 05:00", E_A, f"{d} 05:59:59") + rows(f"{d} 05:00", E_B, f"{d} 05:59:59.2")
          + rows(f"{d} 05:00", E_C, f"{d} 05:59:59.3"))
    # 06: late snapshot (06:05) -> real but 55 minutes old at bar end
    r += (rows(f"{d} 06:00", E_A, f"{d} 06:05:00") + rows(f"{d} 06:00", E_B, f"{d} 06:05:00.2")
          + rows(f"{d} 06:00", E_C, f"{d} 06:05:00.3"))
    # 23: fresh A and B only
    r += rows(f"{d} 23:00", E_A, f"{d} 23:59:59") + rows(f"{d} 23:00", E_B, f"{d} 23:59:59.1")
    return options.prepare(pd.DataFrame(r))


def day2() -> pd.DataFrame:
    d0, d = "2025-05-01", "2025-05-02"
    r = []
    # 00: A copied across midnight from previous day's 23:00; B fresh; C reappears (new vs. 23:00 label)
    r += (rows(f"{d} 00:00", E_A, f"{d0} 23:59:59") + rows(f"{d} 00:00", E_B, f"{d} 00:59:59")
          + rows(f"{d} 00:00", E_C, f"{d} 00:59:59.2"))
    return options.prepare(pd.DataFrame(r))


def run(df, date, state):
    es, new_state = options.expiry_snapshots(df, state, CFG)
    return es, options.hour_grid(date, es, df, CFG, state), new_state


def flag(es, label, expiry, col):
    return es.loc[(es.timestamp == pd.Timestamp(label, tz="UTC")) & (es.expiry == expiry), col].item()


def test_expiry_level_ffill_and_age():
    es, hours, _ = run(day1(), "2025-05-01", options.new_state())
    assert not flag(es, "2025-05-01 02:00", E_A, "is_ffill")
    assert flag(es, "2025-05-01 03:00", E_A, "is_ffill") and flag(es, "2025-05-01 03:00", E_B, "is_ffill")
    assert not flag(es, "2025-05-01 04:00", E_A, "is_ffill") and flag(es, "2025-05-01 04:00", E_B, "is_ffill")
    # Age is per expiry: bar end minus that expiry's snapshot.
    assert flag(es, "2025-05-01 04:00", E_B, "age_s") == (pd.Timestamp("2025-05-01 05:00", tz="UTC")
                                                          - pd.Timestamp("2025-05-01 02:59:59.400", tz="UTC")).total_seconds()
    assert flag(es, "2025-05-01 06:00", E_A, "freshness_tier") == "10-60m"
    assert flag(es, "2025-05-01 02:00", E_A, "freshness_tier") == "<10m"
    assert flag(es, "2025-05-01 05:00", E_C, "is_new_in_label")
    # Rule agrees with "copied snapshot predates its bar start" on every row.
    assert (es.is_ffill == es.snap_before_label).all()


def test_hour_summary():
    _, h, _ = run(day1(), "2025-05-01", options.new_state())
    h = h.set_index(h.timestamp.dt.hour)
    assert len(h) == 24 and (~h.has_data).sum() == 24 - 6
    assert h.loc[2, "hour_type"] == "real" and not h.loc[2, "is_partial"]
    assert h.loc[3, "hour_type"] == "ffill" and h.loc[3, "ffill_unflagged_rows"] == 8
    assert h.loc[4, "hour_type"] == "mixed" and h.loc[4, "partial_mixed"] and h.loc[4, "frac_ffill_exp"] == 0.5
    assert h.loc[4, "ffill_unflagged_rows"] == 0 and h.loc[4, "n_stale"] == 4
    assert h.loc[5, "hour_type"] == "real" and h.loc[5, "partial_new_expiry"] and h.loc[5, "n_new_exp"] == 1
    assert h.loc[6, "freshness_tier"] == "10-60m" and h.loc[6, "has_real"]
    assert h.loc[23, "n_dropped_exp"] == 0  # 06 -> 23 is not adjacent, so no drop is counted
    assert h.loc[2, "n_raw_ts"] == 2 and not h.spread_flag.any()
    assert h.has_real.sum() == 5
    gaps = options.longest_gaps(h.reset_index(drop=True), "has_real", 3)
    assert gaps.iloc[0].length_h == 16 and gaps.iloc[0].start == pd.Timestamp("2025-05-01 07:00", tz="UTC")


def test_state_carries_across_midnight():
    _, _, st = run(day1(), "2025-05-01", options.new_state())
    es2, h2, _ = run(day2(), "2025-05-02", st)
    assert flag(es2, "2025-05-02 00:00", E_A, "is_ffill")
    assert not flag(es2, "2025-05-02 00:00", E_B, "is_ffill")
    assert flag(es2, "2025-05-02 00:00", E_C, "is_new_in_label")
    h0 = h2.iloc[0]
    assert h0.hour_type == "mixed" and h0.partial_new_expiry and h0.prev_label_has_data


def test_without_state_midnight_copy_is_not_detected():
    """Sanity: the cross-midnight copy is only identifiable with the carried state."""
    es2, _, _ = run(day2(), "2025-05-02", options.new_state())
    assert not flag(es2, "2025-05-02 00:00", E_A, "is_ffill")
    assert flag(es2, "2025-05-02 00:00", E_A, "snap_before_label")
