"""Hour-of-week vol-time weights estimated from post-switch tau_years decrements.

For consecutive obs of the same long-dated curve one calendar hour apart, the weight of the calendar
hour ending at the later obs is -d(tau_years) * year_days * 24 / d(obs hours). Snapshots sit at ~hh:59:59,
so the interval is essentially the label's own calendar hour: hour_of_week = Mon 00:00 UTC -> 0 ... Sun 23:00 -> 167.

The clock mixes time zones: the US-session block follows New York time (it moves one UTC hour at each US DST
change) while other blocks stay fixed in UTC. No single time zone makes it periodic, so weights are estimated
per UTC hour-of-week separately for each US DST regime (EDT / EST, from `clock.local_tz`).
Units: vol-time hours per calendar hour (a calendar clock would give 1.0 everywhere; a week sums to 168).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOUR = pd.Timedelta("1h")
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def hour_of_week(ts: pd.Series) -> pd.Series:
    return ts.dt.dayofweek * 24 + ts.dt.hour


def dst_regime(ts: pd.Series, tz: str) -> pd.Series:
    """'edt' / 'est' (generally: 'dst' / 'std') for the local zone at each UTC timestamp."""
    loc = ts.dt.tz_convert(tz)
    return pd.Series(np.where(loc.map(lambda t: bool(t.dst())), "edt", "est"), index=ts.index)


def observations(curves: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cc = cfg["clock"]
    yd = cfg["conventions"]["year_days"]
    c = curves[(curves.clock_regime == "vol_time") & (curves.tenor_days_cal > cc["min_tenor_days"])]
    c = c.sort_values(["expiry", "obs_time"])
    g = c.groupby("expiry")
    d_obs_h = g.obs_time.diff().dt.total_seconds() / 3600
    ok = (g.timestamp.diff() == HOUR) & ((d_obs_h * 3600 - 3600).abs() <= cc["pair_gap_tol_s"])
    wt = -g.tau_vol.diff() * yd * 24 / d_obs_h
    obs = pd.DataFrame({"timestamp": c.timestamp, "expiry": c.expiry, "weight": wt})[ok]
    # One value per label: median across long-dated curves (they share the clock).
    per = obs.groupby("timestamp").agg(weight=("weight", "median"), n_curves=("weight", "size"),
                                       spread=("weight", lambda s: s.max() - s.min())).reset_index()
    per["how"] = hour_of_week(per.timestamp)
    per["dst"] = dst_regime(per.timestamp, cc["local_tz"])
    return per


def estimate(curves: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    per = observations(curves, cfg)
    tab = pd.DataFrame({"how": range(168)})
    tab["dow_utc"] = [DOW[h // 24] for h in tab.how]
    tab["hour_utc"] = tab.how % 24
    if per.empty:
        return tab, per, {"n_obs": 0}
    hat = per.groupby(["dst", "how"]).weight.median().rename("weight_hat")
    per = per.join(hat, on=["dst", "how"])
    per["resid"] = per.weight - per.weight_hat
    for reg in ["edt", "est"]:
        p = per[per.dst == reg]
        g = p.groupby("how")
        tab[f"weight_{reg}"] = g.weight.median().reindex(tab.how).to_numpy()
        tab[f"n_obs_{reg}"] = g.size().reindex(tab.how).fillna(0).astype(int).to_numpy()
        tab[f"mad_{reg}"] = g.resid.apply(lambda s: s.abs().median()).reindex(tab.how).to_numpy()
        tab[f"max_abs_resid_{reg}"] = g.resid.apply(lambda s: s.abs().max()).reindex(tab.how).to_numpy()
    per["date"] = per.timestamp.dt.strftime("%Y-%m-%d")
    by_date = per.groupby("date").agg(
        n=("resid", "size"), dst=("dst", lambda s: "/".join(sorted(set(s)))),
        max_abs_resid=("resid", lambda s: s.abs().max()), sum_resid=("resid", "sum"),
        dow=("timestamp", lambda s: s.iloc[0].day_name()[:3]),
        worst_utc_hour=("resid", lambda s: int(per.loc[s.abs().idxmax(), "timestamp"].hour)))
    by_date = by_date.sort_values("max_abs_resid", ascending=False).reset_index()
    stats = {"n_obs": len(per), "first": per.timestamp.min(), "last": per.timestamp.max(),
             "n_obs_edt": int((per.dst == "edt").sum()), "n_obs_est": int((per.dst == "est").sum()),
             "week_sum_edt": float(tab.weight_edt.sum()), "week_sum_est": float(tab.weight_est.sum()),
             "hours_missing_edt": int(tab.weight_edt.isna().sum()), "hours_missing_est": int(tab.weight_est.isna().sum()),
             "frac_resid_gt_1e-3": float((per.resid.abs() > 1e-3).mean()),
             "frac_resid_gt_1e-2": float((per.resid.abs() > 1e-2).mean()),
             "n_dates_resid_gt_1e-2": int((by_date.max_abs_resid > 1e-2).sum()),
             "median_cross_curve_spread": float(per.spread.median()),
             "hours_differing_edt_est": int(((tab.weight_edt - tab.weight_est).abs() > 1e-3).sum())}
    return tab, by_date, stats
