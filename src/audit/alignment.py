"""F. Time-alignment check: short-dated option forward vs. underlying close at lags -1/0/+1 hours.

Option label h carries a snapshot taken near h:59:59 (bar end); underlying label h has close at bar end.
If conventions agree, lag 0 should give the tightest and smoothest log(F/close).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOUR = pd.Timedelta("1h")


def candidates(es: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per label with real snapshots: the shortest non-forward-filled expiry with tenor > min_tenor_h."""
    e = es[~es.is_ffill].assign(tenor_h=lambda x: (x.expiry - x.snap_time).dt.total_seconds() / 3600)  # calendar hours
    e = e[e.tenor_h > cfg["alignment"]["min_tenor_h"]]
    pick = e.sort_values("tenor_h").groupby("timestamp").head(1)
    return pick[["timestamp", "expiry", "expiry_label", "tenor_h", "forward", "snap_time"]].sort_values("timestamp").reset_index(drop=True)


def run(cand: pd.DataFrame, series: pd.DataFrame, cfg: dict) -> dict:
    close = series.set_index("timestamp").close
    a = cand.copy()
    lags = cfg["alignment"]["lags_h"]
    for L in lags:
        a[f"close_lag{L:+d}"] = close.reindex(a.timestamp + L * HOUR).to_numpy()
        a[f"d_lag{L:+d}"] = np.log(a.forward / a[f"close_lag{L:+d}"])
    a["age_min"] = (a.timestamp + HOUR - a.snap_time).dt.total_seconds() / 60
    fresh_max = cfg["snapshot"]["freshness_edges_min"][0]
    stats = []
    for subset, sel in [("all_real", a.index == a.index), (f"age<{fresh_max:g}m", a.age_min < fresh_max)]:
        b = a[sel]
        consec = (b.timestamp.diff() == HOUR) & (b.expiry == b.expiry.shift())
        for L in lags:
            d = b[f"d_lag{L:+d}"]
            dd = d.diff()[consec]
            q = d.abs().quantile([0.5, 0.9, 0.99])
            stats.append({"subset": subset, "lag_h": L, "n": int(d.notna().sum()), "median_abs_d": q[0.5],
                          "p90_abs_d": q[0.9], "p99_abs_d": q[0.99], "median_d": d.median(),
                          "n_consecutive": int(dd.notna().sum()), "std_hourly_change_d": dd.std(),
                          "median_abs_hourly_change_d": dd.abs().median()})
    a["month"] = a.timestamp.dt.strftime("%Y-%m")
    monthly = a.groupby("month").agg(n=("d_lag+0", "size"), median_d_lag0=("d_lag+0", "median"),
                                     median_abs_d_lag0=("d_lag+0", lambda s: s.abs().median()),
                                     median_tenor_h=("tenor_h", "median")).reset_index()
    return {"detail": a, "stats": pd.DataFrame(stats), "monthly": monthly}
