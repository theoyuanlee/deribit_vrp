"""Option surface panel: delete non-observations, flag everything else, emit curve/point/pair tables.

Deleted (not real observations): 1a forward-filled curves, 2a lingering rows, 2b stale atm pairs.
Flagged (real observations, kept): 1b late snapshot, 3a short tenor, 3b low r^2, 3c vendor check.
Which flags exclude a curve is decided at use time by clean.filters.apply_research_filter().
The raw implied_vol never leaves this module: only w = implied_vol**2 * tau_years is kept.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from audit import options as aopt
from clean import filters

HOUR = pd.Timedelta("1h")
KEYS = ["timestamp", "expiry"]
LEDGER_COLS = ["date", "action", "rule", "rows_in", "rows_removed", "curves_in", "curves_removed",
               "rows_flagged", "curves_flagged"]


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
    """Rule accounting. 'delete' rows run in sequence (rows/curves removed); 'flag' rows count curves
    (and their rows) carrying each flag in the final panel. Flags may overlap."""

    def __init__(self, date: str):
        self.date, self.rows = date, []

    def delete(self, rule: str, before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
        cb, ca = before.groupby(KEYS).ngroups, after.groupby(KEYS).ngroups if len(after) else 0
        self.rows.append({"date": self.date, "action": "delete", "rule": rule, "rows_in": len(before),
                          "rows_removed": len(before) - len(after), "curves_in": cb, "curves_removed": cb - ca})
        return after

    def flag(self, rule: str, curves: pd.DataFrame, flagged: pd.Series):
        self.rows.append({"date": self.date, "action": "flag", "rule": rule, "rows_in": int(curves.n_rows.sum()),
                          "curves_in": len(curves), "rows_flagged": int(curves.n_rows[flagged].sum()),
                          "curves_flagged": int(flagged.sum())})


def clean_day(df: pd.DataFrame, es: pd.DataFrame, date: str, cfg: dict):
    """df: audit-prepared raw rows for one day; es: audit expiry snapshots (with is_ffill).
    Returns (curves, points, calendar_pairs, ledger_rows)."""
    rc = cfg["rules"]
    led = Ledger(date)
    df = df.merge(es[KEYS + ["is_ffill", "is_new_in_label"]], on=KEYS, how="left")
    mixed = es.groupby("timestamp").is_ffill.agg(lambda s: s.any() and not s.all())
    df["in_mixed_hour"] = df.timestamp.map(mixed).fillna(False).astype(bool)
    df["curve_obs_ms"] = df.groupby(KEYS).raw_timestamp.transform("max")
    df["obs_time"] = pd.to_datetime(df.curve_obs_ms, unit="ms", utc=True)
    df["age_s"] = (df.timestamp + HOUR - df.obs_time).dt.total_seconds()

    # Deletions: only data that are not real observations.
    x = led.delete("1a_drop_forward_filled_curves", df, df[~df.is_ffill])
    lag_s = (x.curve_obs_ms - x.raw_timestamp) / 1000
    x = led.delete(f"2a_drop_rows_older_than_curve_by_{rc['lingering_tol_s']:g}s", x, x[lag_s <= rc["lingering_tol_s"]])
    atm_latest = x.is_atm & (x.raw_timestamp == x.curve_obs_ms)
    keep = ~x.is_atm | atm_latest
    # If several atm pairs share the latest timestamp, keep the last C and last P only.
    dup = x[keep & x.is_atm].duplicated(KEYS + ["option_type"], keep="last")
    keep &= ~dup.reindex(x.index, fill_value=False)
    x = led.delete("2b_atm_keep_latest_pair", x, x[keep])

    x = x.assign(tau_cal=tau_cal(x.expiry, x.obs_time, cfg["conventions"]["year_days"]))
    curves, points, pairs = build_tables(x, cfg)

    # Flags (curve level). Same thresholds as the former deletion rules, now only recorded.
    if len(curves):
        led.flag(f"1b_flag_late_snapshot_gt_{rc['max_age_min']:g}min", curves, curves.flag_late_snapshot)
        led.flag(f"3a_flag_short_tenor_lt_{rc['min_tenor_days']:g}d", curves, curves.flag_short_tenor)
        led.flag(f"3b_flag_low_r2_lt_{rc['min_r_squared']:g}", curves, curves.flag_low_r2)
        for cls, values in cfg["vendor_check_classes"].items():
            if cls == "GOOD":
                continue
            for v in values:
                led.flag(f"3c_vendor_check={v}", curves, curves.vendor_check == v)
    return curves, points, pairs, led.rows


def curve_flags(x: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-curve flag columns from the retained rows. Thresholds and comparisons mirror the former
    deletion rules exactly (a curve that used to pass a rule has the flag False)."""
    rc = cfg["rules"]
    yd = cfg["conventions"]["year_days"]
    g = x.groupby(KEYS)
    f = g.agg(age_s=("age_s", "first"), tau_cal=("tau_cal", "first"), r2_min=("r_squared", "min"),
              vendor_check=("vol_curve_check", "first"), n_vendor_check=("vol_curve_check", "nunique"))
    if (f.n_vendor_check > 1).any():
        bad = f[f.n_vendor_check > 1].index[:5].tolist()
        raise ValueError(f"vol_curve_check is not constant within curve(s), e.g. {bad}")
    out = pd.DataFrame(index=f.index)
    out["flag_late_snapshot"] = ~(f.age_s <= rc["max_age_min"] * 60)
    out["flag_short_tenor"] = ~(f.tau_cal * yd >= rc["min_tenor_days"])
    out["flag_low_r2"] = ~(f.r2_min >= rc["min_r_squared"])          # missing r^2 is flagged
    out["vendor_check"] = f.vendor_check.astype(str)
    out["vendor_check_class"] = filters.classify_vendor_check(f.vendor_check.astype(object), cfg)
    return out.reset_index()


def build_tables(x: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rc = cfg["rules"]
    if x.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
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
    cur["n_strikes"] = cur.n_strikes.fillna(0).astype(int)
    cur["has_atm"] = cur.atm_w.notna()
    cur["tenor_days_cal"] = cur.tau_cal * cfg["conventions"]["year_days"]
    cur["clock_regime"] = clock_regime(cur.timestamp, cur.tau_years, cur.tau_cal, cfg)
    cur["tau_vol"] = cur.tau_years.where(cur.clock_regime != "calendar")
    cur = cur.merge(curve_flags(x, cfg), on=KEYS, how="left")
    arb, pairs = arbitrage_flags(pts, cur, cfg)
    cur = cur.merge(arb, on=KEYS, how="left")
    for col in ["n_butterfly_viol", "n_calendar_viol"]:
        cur[col] = cur[col].fillna(0).astype(int)
    cols = ["obs_time", "timestamp", "expiry", "expiry_label", "forward", "tau_cal", "tau_vol", "clock_regime",
            "tenor_days_cal", "atm_w", "atm_strike", "has_atm", "r_squared", "n_strikes", "logm_min", "logm_max",
            "age_s", "in_mixed_hour", "is_new_in_label", "n_iv_floor", "n_unpaired", "n_stale_rows", "n_tau",
            "flag_late_snapshot", "flag_short_tenor", "flag_low_r2", "vendor_check", "vendor_check_class",
            "n_butterfly_viol", "n_calendar_viol", "n_rows"]
    pts_cols = ["obs_time", "timestamp", "expiry", "strike", "logm", "w", "call_delta", "iv_floor", "is_stale",
                "has_call", "has_put", "w_put_minus_call"]
    return cur[cols], pts[pts_cols], pairs


def call_from_w(F, K, w):
    sd = np.sqrt(np.maximum(w, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * w) / sd
    return np.where(sd > 0, F * norm.cdf(d1) - K * norm.cdf(d1 - sd), np.maximum(F - K, 0.0))


PAIR_COLS = ["timestamp", "expiry_short", "expiry_long", "tenor_days_short", "tenor_days_long",
             "n_grid_checked", "n_calendar_viol", "adjacent_in_panel"]


def arbitrage_flags(pts: pd.DataFrame, cur: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per curve: butterfly violations (adjacent call slopes non-decreasing in K, prices from w) and
    calendar violations versus the next-shorter panel curve at the same label.
    Also every (shorter, longer) expiry pair at each label, so any research filter can re-derive adjacency.
    Calendar check: total variance on the log-moneyness grid, inside both curves' strike ranges."""
    fc = cfg["flags"]
    g = fc["calendar_k_grid"]
    kgrid = np.arange(g["min"], g["max"] + 1e-12, g["step"])
    tenor = cur.set_index(KEYS).tenor_days_cal
    rows, pair_frames = [], []
    for ts, snap in pts.groupby("timestamp"):
        expiries, W, V = [], [], []
        for exp, c in snap.sort_values(["expiry", "strike"]).groupby("expiry", sort=True):
            K, k, w = c.strike.to_numpy(), c.logm.to_numpy(), c.w.to_numpy()
            nb = 0
            if len(K) >= 3:
                C = call_from_w(c.forward.iloc[0], K, w)
                s = np.diff(C) / np.diff(K)
                nb = int((np.diff(s) < -fc["butterfly_slope_tol"]).sum())
            rows.append({"timestamp": ts, "expiry": exp, "n_butterfly_viol": nb})
            expiries.append(exp)
            W.append(np.interp(kgrid, k, w))                       # pointwise, so identical to interpolating a subset
            V.append((kgrid >= k.min()) & (kgrid <= k.max()))
        W, V = np.array(W), np.array(V)
        i, j = np.triu_indices(len(expiries), k=1)                  # i shorter, j longer (sorted by expiry)
        both = V[i] & V[j]
        viol = both & (W[j] < W[i] - fc["calendar_w_tol"])
        e = np.array(expiries, dtype=object)
        pair_frames.append(pd.DataFrame({
            "timestamp": ts, "expiry_short": e[i], "expiry_long": e[j],
            "tenor_days_short": tenor.reindex(list(zip([ts] * len(i), e[i]))).to_numpy(),
            "tenor_days_long": tenor.reindex(list(zip([ts] * len(j), e[j]))).to_numpy(),
            "n_grid_checked": both.sum(axis=1), "n_calendar_viol": viol.sum(axis=1), "adjacent_in_panel": j == i + 1}))
    arb = pd.DataFrame(rows, columns=KEYS + ["n_butterfly_viol"])
    pairs = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame(columns=PAIR_COLS)
    pairs["expiry_short"] = pd.to_datetime(pairs.expiry_short, utc=True)
    pairs["expiry_long"] = pd.to_datetime(pairs.expiry_long, utc=True)
    adj = pairs[pairs.adjacent_in_panel][["timestamp", "expiry_long", "n_calendar_viol"]]
    arb = arb.merge(adj.rename(columns={"expiry_long": "expiry"}), on=KEYS, how="left")
    arb["n_calendar_viol"] = arb.n_calendar_viol.fillna(0).astype(int)
    return arb, pairs[PAIR_COLS]
