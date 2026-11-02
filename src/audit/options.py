"""B. Option snapshot layer on a full 24h UTC grid.

Snapshot unit is (timestamp label, expiry): each expiry curve carries its own raw_timestamp.
An expiry is forward-filled at a label when its raw_timestamp equals the last one seen for that
expiry at any earlier label (state is carried across days).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOUR = pd.Timedelta("1h")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["snap_time"] = pd.to_datetime(df["raw_timestamp"], unit="ms", utc=True)
    df["is_atm"] = df["surface_key_type"].eq("atm")
    df["price_null"] = df["mid"].isna()
    return df


def new_state() -> dict:
    return {"last_raw_ts": {}, "prev_ts": None, "prev_expiries": set()}


def freshness_tier(age_s: pd.Series, edges_min: list[float]) -> pd.Series:
    bins = [-np.inf] + [e * 60 for e in edges_min] + [np.inf]
    labels = [f"<{edges_min[0]:g}m"] + [f"{a:g}-{b:g}m" for a, b in zip(edges_min[:-1], edges_min[1:])] + [f">{edges_min[-1]:g}m"]
    return pd.cut(age_s, bins=bins, labels=labels, right=False).astype(str)


def tier_labels(edges_min: list[float]) -> list[str]:
    return freshness_tier(pd.Series([-1.0] + [e * 60 for e in edges_min]), edges_min).tolist()


def expiry_snapshots(df: pd.DataFrame, state: dict, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """One row per (timestamp, expiry) with snapshot time, age, ffill and new-listing flags."""
    es = (
        df.groupby(["timestamp", "expiry"], sort=True)
        .agg(
            expiry_label=("expiry_label", "first"),
            n_rows=("strike", "size"),
            n_atm_rows=("is_atm", "sum"),
            n_raw_ts=("raw_timestamp", "nunique"),
            raw_ts=("raw_timestamp", "max"),  # curve snapshot = latest row snapshot
            raw_ts_min=("raw_timestamp", "min"),
            n_stale=("is_stale", "sum"),
            n_price_null=("price_null", "sum"),
            tau_years=("tau_years", "mean"),
        )
        .reset_index()
    )
    es["snap_time"] = pd.to_datetime(es["raw_ts"], unit="ms", utc=True)
    es["age_s"] = (es["timestamp"] + HOUR - es["snap_time"]).dt.total_seconds()
    es["freshness_tier"] = freshness_tier(es["age_s"], cfg["snapshot"]["freshness_edges_min"])

    state = {"last_raw_ts": dict(state["last_raw_ts"]), "prev_ts": state["prev_ts"], "prev_expiries": set(state["prev_expiries"])}
    is_ffill, is_new, prev_has = [], [], []
    for ts, grp in es.groupby("timestamp", sort=True):
        prev_ok = state["prev_ts"] is not None and state["prev_ts"] == ts - HOUR
        for e, r in zip(grp["expiry"], grp["raw_ts"]):
            is_ffill.append(state["last_raw_ts"].get(e) == r)
            is_new.append(prev_ok and e not in state["prev_expiries"])
            prev_has.append(prev_ok)
        state["last_raw_ts"].update(zip(grp["expiry"], grp["raw_ts"]))
        state["prev_ts"], state["prev_expiries"] = ts, set(grp["expiry"])
    es["is_ffill"] = is_ffill
    es["is_new_in_label"] = is_new
    es["prev_label_has_data"] = prev_has
    # Independent cross-check: a copied snapshot must predate its own bar start.
    es["snap_before_label"] = es["snap_time"] < es["timestamp"]
    return es, state


def hour_grid(date: str, es: pd.DataFrame, df: pd.DataFrame, cfg: dict, state_before: dict) -> pd.DataFrame:
    grid = pd.DataFrame({"timestamp": pd.date_range(pd.Timestamp(date, tz="UTC"), periods=24, freq="h")})
    edges = cfg["snapshot"]["freshness_edges_min"]
    if len(es):
        es = es.assign(
            ffill_rows=np.where(es.is_ffill, es.n_rows, 0),
            ffill_unflagged_rows=np.where(es.is_ffill, es.n_rows - es.n_stale, 0),
            real_snap=es.snap_time.where(~es.is_ffill),
            ffill_disagree=es.is_ffill != es.snap_before_label,
            real_raw_ts_max=es.raw_ts.where(~es.is_ffill),
            real_raw_ts_min=es.raw_ts.where(~es.is_ffill),
        )
        h = es.groupby("timestamp").agg(
            n_rows=("n_rows", "sum"),
            n_expiries=("expiry", "size"),
            n_ffill_exp=("is_ffill", "sum"),
            n_new_exp=("is_new_in_label", "sum"),
            prev_label_has_data=("prev_label_has_data", "first"),
            max_age_s=("age_s", "max"),
            min_age_s=("age_s", "min"),
            n_stale=("n_stale", "sum"),
            n_price_null=("n_price_null", "sum"),
            ffill_rows=("ffill_rows", "sum"),
            ffill_unflagged_rows=("ffill_unflagged_rows", "sum"),
            snap_time=("snap_time", "max"),
            real_snap_time=("real_snap", "max"),
            raw_ts_max=("real_raw_ts_max", "max"),
            raw_ts_min=("real_raw_ts_min", "min"),
            n_ffill_rule_disagree=("ffill_disagree", "sum"),
        )
        h["n_raw_ts"] = df.groupby("timestamp")["raw_timestamp"].nunique()
        h = h.reset_index()
    else:
        num = ["n_rows", "n_expiries", "n_ffill_exp", "n_new_exp", "max_age_s", "min_age_s", "n_stale", "n_price_null",
               "ffill_rows", "ffill_unflagged_rows", "raw_ts_max", "raw_ts_min", "n_ffill_rule_disagree", "n_raw_ts"]
        h = pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
                          **{c: pd.Series(dtype=float) for c in num},
                          "prev_label_has_data": pd.Series(dtype=object),
                          "snap_time": pd.Series(dtype="datetime64[ns, UTC]"),
                          "real_snap_time": pd.Series(dtype="datetime64[ns, UTC]")})
    out = grid.merge(h, on="timestamp", how="left")
    out.insert(0, "date", date)
    out["has_data"] = out["n_expiries"].fillna(0) > 0
    for c in ["n_rows", "n_expiries", "n_ffill_exp", "n_new_exp", "n_stale", "n_price_null", "ffill_rows",
              "ffill_unflagged_rows", "n_raw_ts", "n_ffill_rule_disagree"]:
        out[c] = out[c].fillna(0).astype(int)
    out["n_real_exp"] = out["n_expiries"] - out["n_ffill_exp"]
    out["frac_ffill_exp"] = np.where(out.has_data, out.n_ffill_exp / out.n_expiries.where(out.n_expiries > 0), np.nan)
    out["stale_frac"] = np.where(out.has_data, out.n_stale / out.n_rows.where(out.n_rows > 0), np.nan)
    # Spread of curve snapshot times among non-copied expiries only; copied curves and lingering
    # rows inside a curve (n_raw_ts > 1, reported separately) legitimately carry older timestamps.
    out["raw_ts_spread_s"] = (out["raw_ts_max"] - out["raw_ts_min"]) / 1000.0
    out["spread_flag"] = out["raw_ts_spread_s"] > cfg["snapshot"]["intra_label_spread_flag_s"]
    out["has_real"] = out["n_real_exp"] > 0
    out["hour_type"] = np.select(
        [~out.has_data, out.n_ffill_exp == 0, out.n_real_exp == 0], ["missing", "real", "ffill"], "mixed")
    out["partial_mixed"] = out.hour_type.eq("mixed")
    out["partial_new_expiry"] = out.has_data & (out.n_new_exp > 0) & out.prev_label_has_data.astype("boolean").fillna(False).astype(bool)
    out["is_partial"] = out.partial_mixed | out.partial_new_expiry
    out["real_snap_offset_min"] = (out["real_snap_time"] - out["timestamp"]).dt.total_seconds() / 60.0

    # Expiries dropped vs. previous label (needs previous-day state for 00:00).
    prev = {state_before["prev_ts"]: state_before["prev_expiries"]} if state_before["prev_ts"] is not None else {}
    prev.update({ts: set(g.expiry) for ts, g in es.groupby("timestamp")} if len(es) else {})
    out["n_dropped_exp"] = [
        len(prev[ts - HOUR] - prev.get(ts, set())) if (ts - HOUR) in prev and ts in prev else 0 for ts in out.timestamp
    ]
    out["freshness_tier"] = np.where(out.has_data, freshness_tier(out["max_age_s"], edges), "missing")
    return out.drop(columns=["raw_ts_max", "raw_ts_min"])


def report_actuals(df: pd.DataFrame, es: pd.DataFrame, hours: pd.DataFrame) -> dict:
    """Values to compare with the options report.json."""
    real = es[~es.is_ffill]
    flag = lambda c: int(df[c].fillna(0).astype(float).sum())
    return {
        "rows": len(df),
        "processed_rows": len(df),
        "raw_rows": int(real.n_rows.sum()),
        "is_stale_rows": int(df.is_stale.sum()),
        "stale_rows": int(df.is_stale.sum()),
        "unique_timestamps": int(df.timestamp.nunique()),
        "unique_expiry": int(df.expiry.nunique()),
        "unique_instrument_id": int(df.instrument_id.nunique()),
        "instruments": int(df.instrument_id.nunique()),
        "missing_price_rows": int(df.price_null.sum()),
        "bad_bid_ask_rows": flag("bad_bid_ask"),
        "crossed_market_rows": flag("crossed_market"),
        "non_positive_mid_rows": flag("non_positive_mid"),
        "quote_ok_rows": flag("quote_ok"),
        "duplicate_key_rows": int(df.duplicated(["timestamp", "instrument_name"]).sum()),
        "expected_grid_points": 24,
        "missing_grid_points": int((~hours.has_data).sum()),
        "timestamp_min": df.timestamp.min(),
        "timestamp_max": df.timestamp.max(),
        "raw_timestamp_min": df.snap_time.min(),
        "raw_timestamp_max": df.snap_time.max(),
    }


def longest_gaps(hours: pd.DataFrame, mask_col: str, top: int) -> pd.DataFrame:
    """Longest runs of consecutive grid hours where `mask_col` is False."""
    h = hours.sort_values("timestamp").reset_index(drop=True)
    bad = ~h[mask_col].astype(bool)
    run_id = (bad != bad.shift()).cumsum()
    runs = (
        h[bad].groupby(run_id[bad])
        .agg(start=("timestamp", "min"), end=("timestamp", "max"), length_h=("timestamp", "size"),
             n_missing=("has_data", lambda s: int((~s).sum())))
        .reset_index(drop=True)
    )
    runs["n_ffill_only"] = runs["length_h"] - runs["n_missing"]
    return runs.sort_values(["length_h", "start"], ascending=[False, True]).head(top).reset_index(drop=True)
