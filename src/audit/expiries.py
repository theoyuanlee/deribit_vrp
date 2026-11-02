"""C. Expiry layer: per-(timestamp, expiry) curve metrics and lifecycle / listing summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd

HOUR = pd.Timedelta("1h")


def curve_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per (timestamp, expiry): forward, r^2, ATM IV (from the synthetic atm row), listed strike coverage."""
    keys = ["timestamp", "expiry"]
    base = df.groupby(keys).agg(
        tenor_days=("tenor_days", "first"),
        forward=("forward_price", "first"),
        n_forward=("forward_price", "nunique"),
        r_squared=("r_squared", "first"),
        n_r_squared=("r_squared", "nunique"),
    )
    # Old atm rows (named after an earlier forward) can linger inside a fresh curve; take the atm row
    # from the curve's own (latest) snapshot.
    curve_ts = df.groupby(keys)["raw_timestamp"].transform("max")
    old = df["raw_timestamp"] < curve_ts
    atm = df[df.is_atm & ~old].groupby(keys).agg(atm_iv=("implied_vol", "first"), atm_strike=("strike", "first"))
    lingering = df.assign(old=old, old_atm=old & df.is_atm, old_atm_unflagged=old & df.is_atm & ~df.is_stale).groupby(keys).agg(
        n_rows_older_snapshot=("old", "sum"), n_atm_older=("old_atm", "sum"), n_atm_older_unflagged=("old_atm_unflagged", "sum"))
    listed = df[~df.is_atm].groupby(keys).agg(
        n_strikes=("strike", "nunique"), strike_min=("strike", "min"), strike_max=("strike", "max"))
    out = base.join(atm, how="left").join(lingering, how="left").join(listed, how="left").reset_index()
    out["n_strikes"] = out["n_strikes"].fillna(0).astype(int)
    out["logm_min"] = np.log(out.strike_min / out.forward)
    out["logm_max"] = np.log(out.strike_max / out.forward)
    return out


def tenor_bucket(days: pd.Series, edges: list[float]) -> pd.Series:
    labels = [f"{a:g}-{b:g}d" for a, b in zip(edges[:-1], edges[1:])]
    return pd.cut(days, bins=edges, labels=labels, right=False).astype(str)


def _presence(es: pd.DataFrame, hours: pd.DataFrame) -> pd.DataFrame:
    """Boolean matrix: grid hour x expiry, True where the expiry has rows at that label."""
    m = es.assign(v=True).pivot_table(index="timestamp", columns="expiry", values="v", aggfunc="any", fill_value=False)
    return m.reindex(hours.sort_values("timestamp").timestamp, fill_value=False).astype(bool)


def lifecycle(es: pd.DataFrame, hours: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-expiry first/last appearance and presence ratio; plus runs of intermittent absence."""
    pres = _presence(es, hours)
    has_data = hours.set_index("timestamp").has_data.reindex(pres.index).to_numpy()
    real = es[~es.is_ffill].groupby("expiry").size()
    labels = es.groupby("expiry").expiry_label.first()
    life_rows, gap_rows = [], []
    ts = pres.index
    for e in pres.columns:
        col = pres[e].to_numpy()
        idx = np.flatnonzero(col)
        i0, i1 = idx[0], idx[-1]
        life, life_data = col[i0:i1 + 1], has_data[i0:i1 + 1]
        absent_data = (~life) & life_data
        life_rows.append({
            "expiry": e, "expiry_label": labels[e], "first_seen": ts[i0], "last_seen": ts[i1],
            "hours_before_expiry_last_seen": (e - (ts[i1] + HOUR)).total_seconds() / 3600,
            "n_life_hours": len(life), "n_present": int(life.sum()), "n_present_real": int(real.get(e, 0)),
            "n_absent_while_others_present": int(absent_data.sum()),
            "n_absent_hour_missing": int(((~life) & (~life_data)).sum()),
            "frac_present": life.sum() / len(life),
            "frac_present_when_data": life.sum() / max(life_data.sum(), 1),
        })
        # Runs of absence inside the life window that contain at least one hour with other data.
        absent = ~life
        run = np.cumsum(np.r_[True, absent[1:] != absent[:-1]])
        for r in np.unique(run[absent]):
            sel = (run == r) & absent
            if absent_data[sel].any():
                j = np.flatnonzero(sel)
                gap_rows.append({"expiry": e, "expiry_label": labels[e], "gap_start": ts[i0 + j[0]],
                                 "gap_end": ts[i0 + j[-1]], "length_h": len(j),
                                 "n_hours_others_present": int(absent_data[sel].sum())})
    life = pd.DataFrame(life_rows).sort_values("expiry").reset_index(drop=True)
    gaps = pd.DataFrame(gap_rows, columns=["expiry", "expiry_label", "gap_start", "gap_end", "length_h", "n_hours_others_present"])
    return life, gaps


def deribit_candidates(start: pd.Timestamp, end: pd.Timestamp, hour_utc: int) -> pd.DataFrame:
    days = pd.date_range(start.normalize(), end.normalize(), freq="D", tz="UTC") + pd.Timedelta(hours=hour_utc)
    df = pd.DataFrame({"expiry": days})
    df["daily"] = True
    df["weekly"] = df.expiry.dt.dayofweek == 4
    nxt = df.expiry + pd.Timedelta("7D")
    df["monthly"] = df.weekly & (nxt.dt.month != df.expiry.dt.month)
    df["quarterly"] = df.monthly & df.expiry.dt.month.isin([3, 6, 9, 12])
    return df


def deribit_expected(es: pd.DataFrame, hours: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Heuristic: which expiries Deribit's listing rules suggest should be present at each real-snapshot hour."""
    ec = cfg["expiries"]
    lst = ec["deribit_listing"]
    t_hours = hours.loc[hours.has_real, "timestamp"].sort_values()
    if t_hours.empty:
        return pd.DataFrame()
    cand = deribit_candidates(t_hours.min(), t_hours.max() + pd.Timedelta("800D"), ec["expiry_hour_utc"])
    min_tte = pd.Timedelta(hours=ec["min_time_to_expiry_h"])
    arrays = {k: cand.loc[cand[k], "expiry"].dt.tz_convert(None).to_numpy() for k in lst}
    pairs = []
    for t in t_hours:
        ref = np.datetime64((t + HOUR + min_tte).tz_convert(None))
        for kind, n in lst.items():
            arr = arrays[kind]
            i = np.searchsorted(arr, ref, side="right")
            for e in arr[i:i + n]:
                pairs.append((t, pd.Timestamp(e, tz="UTC"), kind))
    exp = pd.DataFrame(pairs, columns=["timestamp", "expiry", "kind"])
    kinds = exp.groupby("expiry").kind.agg(lambda s: "+".join(sorted(set(s))))
    exp = exp.drop_duplicates(["timestamp", "expiry"])
    present = set(zip(es.timestamp, es.expiry))
    exp["present"] = [(t, e) in present for t, e in zip(exp.timestamp, exp.expiry)]
    out = exp.groupby("expiry").agg(first_expected=("timestamp", "min"), last_expected=("timestamp", "max"),
                                    n_expected_hours=("timestamp", "size"), n_present=("present", "sum")).reset_index()
    out["kinds"] = out.expiry.map(kinds)
    out["frac_present"] = out.n_present / out.n_expected_hours
    out["status"] = np.select(
        [out.n_present == 0, out.frac_present < ec["frequent_missing_frac"]],
        ["possibly_missing_never_seen", "possibly_missing_frequent"], "ok")
    observed = es.groupby("expiry").expiry_label.first()
    off = observed[~observed.index.isin(cand.expiry)]
    off_rule = pd.DataFrame({"expiry": off.index, "kinds": "observed_off_rule", "status": "off_rule",
                             "n_present": es[es.expiry.isin(off.index)].groupby("expiry").size().reindex(off.index).to_numpy()})
    out = pd.concat([out, off_rule], ignore_index=True)
    out["expiry_label"] = out.expiry.dt.strftime("%-d%b%y").str.upper()
    return out.sort_values("expiry").reset_index(drop=True)


def r2_by_tenor(es_real: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    thr = cfg["expiries"]["r2_threshold"]
    g = es_real.groupby("tenor_bucket", observed=True).r_squared
    out = g.describe(percentiles=[0.01, 0.05, 0.5]).rename(columns={"count": "n"})
    out[f"frac_below_{thr:g}"] = g.apply(lambda s: (s < thr).mean())
    order = {b: i for i, b in enumerate(tenor_bucket(pd.Series(cfg["expiries"]["tenor_buckets_days"][:-1]),
                                                     cfg["expiries"]["tenor_buckets_days"]))}
    return out.reset_index().sort_values("tenor_bucket", key=lambda s: s.map(order)).reset_index(drop=True)


def cm_bracket(es: pd.DataFrame, hours: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair of expiries bracketing the constant-maturity target at each label with data, and its switches.

    Tenor is measured from the bar end (label + 1h), so switches reflect listing availability,
    not the frozen tau of copied snapshots.
    """
    target = cfg["expiries"]["cm_target_days"]
    e = es[["timestamp", "expiry", "expiry_label", "is_ffill"]].copy()
    e["days_label"] = (e.expiry - (e.timestamp + HOUR)).dt.total_seconds() / 86400
    lo = e[e.days_label <= target].sort_values("days_label").groupby("timestamp").last()
    hi = e[e.days_label > target].sort_values("days_label").groupby("timestamp").first()
    br = pd.DataFrame(index=hours.loc[hours.has_data, "timestamp"].sort_values())
    br["lower"] = lo.expiry_label.reindex(br.index)
    br["upper"] = hi.expiry_label.reindex(br.index)
    br["lower_days"] = lo.days_label.reindex(br.index)
    br["upper_days"] = hi.days_label.reindex(br.index)
    br["bracketed"] = br.lower.notna() & br.upper.notna()
    br["pair"] = br.lower.fillna("-") + "/" + br.upper.fillna("-")
    br = br.reset_index()
    prev = br.shift()
    sw = br[(br.pair != prev.pair) & prev.pair.notna()].copy()
    sw["prev_pair"] = prev.pair[sw.index]
    sw["prev_timestamp"] = prev.timestamp[sw.index]
    present = es.groupby("timestamp").expiry_label.agg(set)
    kinds = []
    for _, r in sw.iterrows():
        p_lo, p_hi = br.loc[_ - 1, "lower"], br.loc[_ - 1, "upper"]
        here = present.get(r.timestamp, set())
        if r.lower == p_hi:
            kinds.append("roll")
        elif (isinstance(p_lo, str) and p_lo not in here) or (isinstance(p_hi, str) and p_hi not in here):
            kinds.append("member_missing")
        elif r.lower not in (p_lo, p_hi) or r.upper not in (p_lo, p_hi):
            kinds.append("new_listing_or_return")
        else:
            kinds.append("other")
    sw["kind"] = kinds
    return br, sw[["prev_timestamp", "timestamp", "prev_pair", "pair", "kind", "lower_days", "upper_days"]].reset_index(drop=True)


def tau_clock(es: pd.DataFrame, cfg: dict) -> dict:
    """Which clock does tau_years run on? Per real label: offset of tau vs. calendar time-to-expiry for
    long-dated curves, and tau decrement per calendar hour (1.0 on a calendar clock)."""
    year_h = cfg["conventions"]["year_days"] * 24
    min_days = cfg["expiries"]["tau_clock_min_days"]
    e = es[~es.is_ffill].copy()
    e["cal_h"] = (e.expiry - e.snap_time).dt.total_seconds() / 3600
    e["tau_h"] = e.tau_years * year_h
    e = e[e.cal_h > min_days * 24].sort_values(["expiry", "timestamp"])
    e["offset_h"] = e.tau_h - e.cal_h
    step = e.groupby("expiry").timestamp.diff() == HOUR
    rate = (-e.groupby("expiry").tau_h.diff() / (e.groupby("expiry").snap_time.diff().dt.total_seconds() / 3600))
    e["rate"] = rate.where(step)
    h = e.groupby("timestamp").agg(n_long=("expiry", "size"), offset_h=("offset_h", "median"),
                                   offset_spread_h=("offset_h", lambda s: s.max() - s.min()), rate=("rate", "median"))
    tol_h = cfg["invariants"]["tau_tol_s"] / 3600
    h["calendar_like"] = (h.offset_h.abs() <= tol_h) & ((h.rate - 1).abs() < 0.01).fillna(True)
    # Switch = start of the first sustained (>= 95% of a 48h window) non-calendar stretch.
    nc = (~h.calendar_like).astype(float)
    roll = nc.rolling(48).mean()
    switch = None
    if (roll > 0.95).any():
        before = h.index[h.index <= roll[roll > 0.95].index[0]]
        cal_before = before[h.calendar_like.reindex(before).to_numpy()]
        switch = h.index[h.index > cal_before[-1]][0] if len(cal_before) else h.index[0]
    h = h.reset_index()
    profile = None
    if switch is not None:
        after = h[(h.timestamp >= switch) & h.rate.notna()]
        profile = after.assign(dow=after.timestamp.dt.dayofweek, hour=after.timestamp.dt.hour) \
            .pivot_table(index="hour", columns="dow", values="rate", aggfunc="median")
        profile.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(profile.columns)]
        profile = profile.reset_index()
    return {"hourly": h, "switch": switch, "profile": profile}
