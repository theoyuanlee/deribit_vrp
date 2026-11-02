"""Underlying hourly panel on the full grid. Zero-return runs are treated as stale prices (close -> missing);
gaps are never interpolated: the next valid close carries one multi-hour return with its span."""
from __future__ import annotations

import numpy as np
import pandas as pd

from audit import io


def load_raw(root, asset: str, dates: list[str]) -> pd.DataFrame:
    frames = []
    for d in dates:
        df, _, _ = io.read_day(root, "underlyings", asset, d)
        if df is not None and len(df):
            frames.append(df[["timestamp", "close"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["timestamp", "close"])


def zero_run_mask(close: pd.Series, min_h: int) -> pd.Series:
    """Bars whose log return vs. the previous grid hour is exactly zero, in runs of >= min_h bars."""
    zero = (np.log(close / close.shift(1)) == 0).to_numpy()
    run = np.cumsum(np.r_[True, zero[1:] != zero[:-1]])
    lengths = pd.Series(run).map(pd.Series(run[zero]).value_counts()).fillna(0).to_numpy()
    return pd.Series(zero & (lengths >= min_h), index=close.index)


def build(raw: pd.DataFrame, start: str, end: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    grid = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta("23h"), freq="h")
    raw = raw.sort_values("timestamp")
    n_dup = int(raw.timestamp.duplicated().sum())
    s = raw.drop_duplicates("timestamp", keep="first").set_index("timestamp").close.reindex(grid)
    absent = s.isna()
    zr = zero_run_mask(s, cfg["underlyings"]["zero_run_min_h"])
    close = s.mask(zr)
    out = pd.DataFrame({"timestamp": grid, "close": close.to_numpy()})
    out["is_missing"] = out.close.isna()
    out["missing_reason"] = np.select([absent.to_numpy(), zr.to_numpy()], ["absent", "zero_run"], "")
    out["in_zero_run"] = zr.to_numpy()
    valid = out[~out.is_missing]
    prev_close = valid.close.shift(1)
    span = (valid.timestamp - valid.timestamp.shift(1)).dt.total_seconds() / 3600
    out.loc[valid.index, "log_ret"] = np.log(valid.close / prev_close)
    out.loc[valid.index, "ret_span_h"] = span
    stats = {"grid_hours": len(out), "duplicates_dropped": n_dup, "absent_hours": int(absent.sum()),
             "zero_run_hours_masked": int(zr.sum()), "missing_hours": int(out.is_missing.sum()),
             "returns": int(out.log_ret.notna().sum()), "multi_hour_returns": int((out.ret_span_h > 1).sum()),
             "max_span_h": float(out.ret_span_h.max())}
    return out, stats
