"""E. Underlying (perp theo_mid candles) layer. The series is small, so it is loaded whole and every
return / OHLC check runs on the full continuous series (23:00 -> 00:00 included)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import io

HOUR = pd.Timedelta("1h")


def load(root, asset: str, dates: list[str]) -> tuple[pd.DataFrame, list[dict], list[dict], list[dict]]:
    frames, reports, mans, xcheck = [], [], [], []
    for d in dates:
        df, rep, man = io.read_day(root, "underlyings", asset, d)
        if rep is not None:
            reports.append(io.flatten_report(rep, "underlyings", d))
        if man is not None:
            mans.append(io.check_manifest(man, "underlyings", d))
        if df is None:
            continue
        df = df.assign(file_date=d)
        frames.append(df)
        if rep is not None:
            xcheck += io.crosscheck(rep, report_actuals(df), "underlyings", d)
    u = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return u, reports, mans, xcheck


def report_actuals(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df), "processed_rows": len(df), "unique_timestamps": int(df.timestamp.nunique()),
        "is_stale_rows": int(df.is_stale.sum()), "duplicate_key_rows": int(df.timestamp.duplicated().sum()),
        "missing_price_rows": int(df.close.isna().sum()), "expected_grid_points": 24,
        "missing_grid_points": 24 - int(df.timestamp.nunique()),
        "timestamp_min": df.timestamp.min(), "timestamp_max": df.timestamp.max(),
    }


def audit(u: pd.DataFrame, start: str, end: str, cfg: dict) -> dict:
    uc = cfg["underlyings"]
    grid = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta("23h"), freq="h")
    counts = u.groupby("timestamp").size()
    out_of_grid = u[~u.timestamp.isin(grid)]
    label_vs_file = u[u.timestamp.dt.strftime("%Y-%m-%d") != u.file_date]

    s = u.sort_values(["timestamp"]).drop_duplicates("timestamp", keep="first").set_index("timestamp").reindex(grid)
    s.index.name = "timestamp"
    s["present"] = s.close.notna() | s.open.notna()
    s["n_dup"] = counts.reindex(grid).fillna(0).astype(int)
    prev_close = s.close.shift(1)            # previous grid hour; NaN if missing
    s["prev_close"] = prev_close
    s["log_ret"] = np.log(s.close / prev_close)
    s["cross_day"] = s.index.hour == 0
    tol = uc["open_eq_rel_tol"]
    s["open_eq_prev_close"] = (s.open - prev_close).abs() <= tol * prev_close.abs()
    s["open_eq_own_close"] = (s.open - s.close).abs() <= tol * s.close.abs()
    s["close_in_range"] = (s.low <= s.close) & (s.close <= s.high)
    s["low_le_high"] = s.low <= s.high
    s["high_lt_open"] = s.high < s.open
    s["low_gt_open"] = s.low > s.open

    sd = s.log_ret.shift(1).rolling(uc["roll_window_h"], min_periods=uc["roll_min_periods"]).std()
    s["trailing_std"] = sd
    s["z"] = s.log_ret / sd
    s["is_jump"] = s.z.abs() > uc["jump_k"]

    p = s[s.present]
    have_prev = p.prev_close.notna()
    checks = []

    def add(name, checked, viol):
        checked = checked.astype(bool)
        checks.append({"check": name, "n_checked": int(checked.sum()), "n_viol": int((viol & checked).sum())})

    add("grid_missing_hours", pd.Series(True, index=s.index), ~s.present)
    add("grid_duplicate_hours", pd.Series(True, index=s.index), s.n_dup > 1)
    add("rows_outside_grid", pd.Series(True, index=out_of_grid.index), pd.Series(True, index=out_of_grid.index))
    add("label_date_ne_file_date", pd.Series(True, index=u.index), u.index.isin(label_vs_file.index))
    add("open_eq_prev_close_intraday", have_prev & ~p.cross_day, ~p.open_eq_prev_close)
    add("open_eq_prev_close_cross_day", have_prev & p.cross_day, ~p.open_eq_prev_close)
    add("open_eq_own_close_cross_day", p.cross_day, p.open_eq_own_close)
    add("open_eq_own_close_intraday", ~p.cross_day, p.open_eq_own_close)
    add("close_outside_low_high", pd.Series(True, index=p.index), ~p.close_in_range)
    add("low_gt_high", pd.Series(True, index=p.index), ~p.low_le_high)
    add("high_lt_open", pd.Series(True, index=p.index), p.high_lt_open)
    add("low_gt_open", pd.Series(True, index=p.index), p.low_gt_open)
    add("high_lt_open_intraday", ~p.cross_day, p.high_lt_open)
    add("low_gt_open_intraday", ~p.cross_day, p.low_gt_open)
    add("mid_ne_close", pd.Series(True, index=p.index), p.mid != p.close)
    add("mark_ne_close", pd.Series(True, index=p.index), p.mark_price != p.close)
    add("non_positive_close", pd.Series(True, index=p.index), p.close <= 0)
    add("is_stale", pd.Series(True, index=p.index), p.is_stale.astype(bool))
    add("quote_ok_false", pd.Series(True, index=p.index), ~p.quote_ok.astype(bool))
    add("log_ret_available", pd.Series(True, index=p.index), p.log_ret.isna())
    add(f"jump_gt_{uc['jump_k']:g}sd", p.z.notna(), p.is_jump)
    add("zero_return", p.log_ret.notna(), p.log_ret == 0)

    jumps = s.loc[s.is_jump, ["prev_close", "close", "log_ret", "trailing_std", "z", "high", "low"]].reset_index()

    zero = (s.log_ret == 0).to_numpy()
    run = np.cumsum(np.r_[True, zero[1:] != zero[:-1]])
    zr = []
    for r in np.unique(run[zero]):
        idx = np.flatnonzero(run == r)
        if len(idx) >= uc["zero_run_min_h"]:
            zr.append({"start": s.index[idx[0]], "end": s.index[idx[-1]], "length_h": len(idx), "close": s.close.iloc[idx[0]]})
    zero_runs = pd.DataFrame(zr, columns=["start", "end", "length_h", "close"])

    miss = ~s.present.to_numpy()
    run = np.cumsum(np.r_[True, miss[1:] != miss[:-1]])
    gaps = [{"start": s.index[np.flatnonzero(run == r)[0]], "length_h": int((run == r).sum())} for r in np.unique(run[miss])]

    series_cols = ["present", "n_dup", "open", "high", "low", "close", "prev_close", "log_ret", "trailing_std", "z",
                   "is_jump", "cross_day", "open_eq_prev_close", "open_eq_own_close", "close_in_range", "high_lt_open", "low_gt_open"]
    return {"series": s[series_cols].reset_index(), "checks": pd.DataFrame(checks), "jumps": jumps,
            "zero_runs": zero_runs, "missing_runs": pd.DataFrame(gaps, columns=["start", "length_h"])}
