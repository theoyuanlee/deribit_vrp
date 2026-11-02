"""Option surface panel: apply cleaning rules per day and emit curve- and point-level tables.

Rules are applied in a fixed order; each step records the rows and curves it removes.
The raw implied_vol never leaves this module: only w = implied_vol**2 * tau_years is kept.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from audit import options as aopt

HOUR = pd.Timedelta("1h")
KEYS = ["timestamp", "expiry"]


def audit_cfg(cfg: dict) -> dict:
    """Minimal audit config needed to reuse the audit's forward-fill detection."""
    return {"snapshot": {"freshness_edges_min": [cfg["rules"]["max_age_min"], 60], "intra_label_spread_flag_s": 60}}


def total_variance(iv, tau_years):
    return iv * iv * tau_years


def tau_cal(expiry: pd.Series, obs_time: pd.Series, year_days: float) -> pd.Series:
    return (expiry - obs_time).dt.total_seconds() / (year_days * 86400)


def clock_regime(timestamp: pd.Series, tau_years: pd.Series, tau_c: pd.Series, cfg: dict) -> pd.Series:
    """'vol_time' from the switch label on; before it 'calendar', or 'calendar_anomaly' when
    tau_years disagrees with calendar time by more than calendar_tol_s."""
    switch = pd.Timestamp(cfg["clock"]["switch_time"])
    off_s = (tau_years - tau_c).abs() * cfg["conventions"]["year_days"] * 86400
    return pd.Series(np.select([timestamp >= switch, off_s > cfg["clock"]["calendar_tol_s"]],
                               ["vol_time", "calendar_anomaly"], "calendar"), index=timestamp.index)


class Ledger:
    """Sequential rule accounting: rows and curves removed at each step."""

    def __init__(self, date: str):
        self.date, self.rows = date, []

    def step(self, rule: str, before: pd.DataFrame, after: pd.DataFrame):
        cb, ca = before.groupby(KEYS).ngroups, after.groupby(KEYS).ngroups if len(after) else 0
        self.rows.append({"date": self.date, "rule": rule, "rows_in": len(before), "rows_removed": len(before) - len(after),
                          "curves_in": cb, "curves_removed": cb - ca})
        return after


def clean_day(df: pd.DataFrame, es: pd.DataFrame, date: str, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """df: audit-prepared raw rows for one day; es: audit expiry snapshots (with is_ffill)."""
    rc = cfg["rules"]
    led = Ledger(date)
    df = df.merge(es[KEYS + ["is_ffill", "is_new_in_label"]], on=KEYS, how="left")
    mixed = es.groupby("timestamp").is_ffill.agg(lambda s: s.any() and not s.all())
    df["in_mixed_hour"] = df.timestamp.map(mixed).fillna(False).astype(bool)
    df["curve_obs_ms"] = df.groupby(KEYS).raw_timestamp.transform("max")
    df["obs_time"] = pd.to_datetime(df.curve_obs_ms, unit="ms", utc=True)
    df["age_s"] = (df.timestamp + HOUR - df.obs_time).dt.total_seconds()

    # 1. Snapshots: real (non-forward-filled) and fresh.
    x = led.step("1a_drop_forward_filled_curves", df, df[~df.is_ffill])
    x = led.step(f"1b_drop_age_gt_{rc['max_age_min']:g}min", x, x[x.age_s <= rc["max_age_min"] * 60])

    # 2. Lingering rows inside a curve; atm: only the latest C/P pair.
    lag_s = (x.curve_obs_ms - x.raw_timestamp) / 1000
    x = led.step(f"2a_drop_rows_older_than_curve_by_{rc['lingering_tol_s']:g}s", x, x[lag_s <= rc["lingering_tol_s"]])
    atm_latest = x.is_atm & (x.raw_timestamp == x.curve_obs_ms)
    keep = ~x.is_atm | atm_latest
    # If several atm pairs share the latest timestamp, keep the last C and last P only.
    dup = x[keep & x.is_atm].duplicated(KEYS + ["option_type"], keep="last")
    keep &= ~dup.reindex(x.index, fill_value=False)
    x = led.step("2b_atm_keep_latest_pair", x, x[keep])

    # 3. Expiries: calendar tenor, r^2, vol_curve_check (curve-level, on the retained rows).
    yd = cfg["conventions"]["year_days"]
    x = x.assign(tau_cal=tau_cal(x.expiry, x.obs_time, yd))
    x = led.step(f"3a_drop_tenor_lt_{rc['min_tenor_days']:g}d", x, x[x.tau_cal * yd >= rc["min_tenor_days"]])
    r2 = x.groupby(KEYS).r_squared.transform("min")
    x = led.step(f"3b_drop_r2_lt_{rc['min_r_squared']:g}", x, x[r2 >= rc["min_r_squared"]])
    good = x.vol_curve_check.eq(rc["require_vol_curve_check"]).groupby([x.timestamp, x.expiry]).transform("all")
    x = led.step("3c_drop_vol_curve_check_not_good", x, x[good])

    curves, points = build_tables(x, cfg)
    return curves, points, led.rows


def build_tables(x: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rc = cfg["rules"]
    if x.empty:
        return pd.DataFrame(), pd.DataFrame()
    x = x.assign(w=total_variance(x.implied_vol, x.tau_years),
                 iv_floor=(x.implied_vol - rc["iv_floor"]).abs() <= rc["iv_floor_tol"])

    # Points: listed strikes only, one row per (obs_time, expiry, strike). Call and put share the IV.
    lst = x[~x.is_atm]
    c = lst[lst.option_type == "call"].set_index(KEYS + ["strike"])
    p = lst[lst.option_type == "put"].set_index(KEYS + ["strike"])
    idx = c.index.union(p.index)
    first = c.reindex(idx).combine_first(p.reindex(idx))
    pts = pd.DataFrame(index=idx)
    pts["obs_time"] = first.obs_time
    pts["forward"] = first.forward_price
    pts["w"] = first.w
    pts["w_put_minus_call"] = (p.w.reindex(idx) - c.w.reindex(idx))  # expected 0 (same IV); kept as a check
    pts["call_delta"] = c.delta.reindex(idx).fillna(p.delta.reindex(idx) + 1.0)
    pts["iv_floor"] = first.iv_floor.astype(bool)
    pts["is_stale"] = (c.is_stale.reindex(idx).fillna(False).astype(bool) | p.is_stale.reindex(idx).fillna(False).astype(bool))
    pts["has_call"], pts["has_put"] = idx.isin(c.index), idx.isin(p.index)
    pts = pts.reset_index()
    pts["logm"] = np.log(pts.strike / pts.forward)

    # Curves.
    g = x.groupby(KEYS)
    cur = g.agg(obs_time=("obs_time", "first"), expiry_label=("expiry_label", "first"), forward=("forward_price", "first"),
                tau_years=("tau_years", "first"), n_tau=("tau_years", "nunique"), tau_cal=("tau_cal", "first"),
                age_s=("age_s", "first"), r_squared=("r_squared", "first"), n_rows=("strike", "size"),
                n_stale_rows=("is_stale", "sum"), in_mixed_hour=("in_mixed_hour", "first"),
                is_new_in_label=("is_new_in_label", "first"))
    atm = x[x.is_atm & (x.option_type == "call")].groupby(KEYS).agg(atm_w=("w", "first"), atm_strike=("strike", "first"))
    cur = cur.join(atm, how="left")
    pg = pts.groupby(KEYS)
    cur = cur.join(pg.agg(n_strikes=("strike", "size"), logm_min=("logm", "min"), logm_max=("logm", "max"),
                          n_iv_floor=("iv_floor", "sum"), n_unpaired=("has_put", lambda s: int((~s).sum()))), how="left")
    cur = cur.reset_index()
    cur["has_atm"] = cur.atm_w.notna()
    cur["tenor_days_cal"] = cur.tau_cal * cfg["conventions"]["year_days"]
    cur["clock_regime"] = clock_regime(cur.timestamp, cur.tau_years, cur.tau_cal, cfg)
    cur["tau_vol"] = cur.tau_years.where(cur.clock_regime != "calendar")
    cur = cur.merge(arbitrage_flags(pts, cur, cfg), on=KEYS, how="left")
    cols = ["obs_time", "timestamp", "expiry", "expiry_label", "forward", "tau_cal", "tau_vol", "clock_regime",
            "tenor_days_cal", "atm_w", "atm_strike", "has_atm", "r_squared", "n_strikes", "logm_min", "logm_max",
            "age_s", "in_mixed_hour", "is_new_in_label", "n_iv_floor", "n_unpaired", "n_stale_rows", "n_tau",
            "n_butterfly_viol", "n_calendar_viol"]
    pts_cols = ["obs_time", "timestamp", "expiry", "strike", "logm", "w", "call_delta", "iv_floor", "is_stale",
                "has_call", "has_put", "w_put_minus_call"]
    return cur[cols], pts[pts_cols]


def call_from_w(F, K, w):
    sd = np.sqrt(np.maximum(w, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * w) / sd
    return np.where(sd > 0, F * norm.cdf(d1) - K * norm.cdf(d1 - sd), np.maximum(F - K, 0.0))


def arbitrage_flags(pts: pd.DataFrame, cur: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per curve: butterfly violations (adjacent call slopes non-decreasing in K, prices from w) and
    calendar violations versus the next-shorter retained expiry at the same label."""
    fc = cfg["flags"]
    g = fc["calendar_k_grid"]
    kgrid = np.arange(g["min"], g["max"] + 1e-12, g["step"])
    rows = []
    for ts, snap in pts.groupby("timestamp"):
        prev = None
        for exp, c in snap.sort_values(["expiry", "strike"]).groupby("expiry", sort=True):
            K, k, w = c.strike.to_numpy(), c.logm.to_numpy(), c.w.to_numpy()
            nb = 0
            if len(K) >= 3:
                C = call_from_w(c.forward.iloc[0], K, w)
                s = np.diff(C) / np.diff(K)
                nb = int((np.diff(s) < -fc["butterfly_slope_tol"]).sum())
            nc = 0
            if prev is not None:
                k0, w0 = prev
                m = (kgrid >= max(k0.min(), k.min())) & (kgrid <= min(k0.max(), k.max()))
                if m.any():
                    nc = int((np.interp(kgrid[m], k, w) < np.interp(kgrid[m], k0, w0) - fc["calendar_w_tol"]).sum())
            rows.append({"timestamp": ts, "expiry": exp, "n_butterfly_viol": nb, "n_calendar_viol": nc})
            prev = (k, w)
    return pd.DataFrame(rows, columns=KEYS + ["n_butterfly_viol", "n_calendar_viol"])
