"""D. Contract-level invariants, flag accounting and static no-arbitrage checks (per day)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import black76

FLAG_COLS = ["bad_bid_ask", "crossed_market", "non_positive_mid", "missing_price", "quote_ok", "is_stale", "price_null"]
EXAMPLE_COLS = ["timestamp", "instrument_name", "expiry_label", "strike", "option_type", "implied_vol",
                "forward_price", "tau_years", "mid", "delta", "is_stale"]


class Checker:
    """Accumulates (check, n_checked, n_viol) for all rows and for real (non-forward-filled) rows."""

    def __init__(self, df: pd.DataFrame, date: str, max_examples: int):
        self.df, self.date, self.max_examples = df, date, max_examples
        self.counts, self.examples = [], []

    def add(self, check: str, checked: pd.Series, viol: pd.Series, extra: dict | None = None):
        checked, viol = checked.astype(bool), (viol & checked).astype(bool)
        real = ~self.df["is_ffill"]
        self.counts.append({"date": self.date, "check": check, "n_checked": int(checked.sum()), "n_viol": int(viol.sum()),
                            "n_checked_real": int((checked & real).sum()), "n_viol_real": int((viol & real).sum())})
        if viol.any():
            ex = self.df.loc[viol, EXAMPLE_COLS].head(self.max_examples).copy()
            for k, v in (extra or {}).items():
                ex[k] = v[viol].head(self.max_examples).to_numpy()
            ex.insert(0, "check", check)
            ex.insert(0, "date", self.date)
            self.examples.append(ex)


def check_day(df: pd.DataFrame, es: pd.DataFrame, date: str, cfg: dict) -> dict:
    ic, year_s = cfg["invariants"], cfg["conventions"]["year_days"] * 86400
    df = df.merge(es[["timestamp", "expiry", "is_ffill"]], on=["timestamp", "expiry"], how="left")
    ck = Checker(df, date, ic["max_examples_per_check_per_day"])
    has_px = ~df.price_null
    is_call = df.option_type.eq("call")
    everything = pd.Series(True, index=df.index)

    # Quote equality.
    eq = (df.bid == df.ask) & (df.ask == df.mid) & (df.mid == df.mark_price) & (df.mark_price == df.theoretical_mid)
    ck.add("bid_ask_mid_mark_equal", has_px, ~eq)
    nulls = df[["bid", "ask", "mid", "mark_price", "theoretical_mid"]].isna()
    ck.add("price_cols_null_together", everything, nulls.any(axis=1) & ~nulls.all(axis=1))

    # Black-76 (undiscounted). Negative / zero prices are counted separately, never as pricing errors.
    model = black76.price(df.forward_price, df.strike, df.tau_years, df.implied_vol, is_call)
    abs_err = (df.mid - model).abs()
    rel_err = abs_err / np.abs(model)
    b76_viol = (abs_err > ic["b76_abs_tol"]) & ~(rel_err <= ic["b76_rel_tol"])
    ck.add("b76_price", has_px, b76_viol, {"model_price": pd.Series(model, index=df.index), "abs_err": abs_err, "rel_err": rel_err})
    ck.add("negative_price", has_px, df.mid < 0)
    ck.add("zero_price", has_px, df.mid == 0)
    ck.add("price_below_intrinsic", has_px,
           df.mid < np.where(is_call, np.maximum(df.forward_price - df.strike, 0), np.maximum(df.strike - df.forward_price, 0)) - ic["b76_abs_tol"])

    # Delta.
    mdelta = black76.delta(df.forward_price, df.strike, df.tau_years, df.implied_vol, is_call)
    ck.add("delta_b76", df.delta.notna(), (df.delta - mdelta).abs() > ic["delta_abs_tol"],
           {"model_delta": pd.Series(mdelta, index=df.index)})
    ck.add("delta_null", everything, df.delta.isna())

    # Call IV == put IV at the same (timestamp, expiry, strike).
    pv = df.pivot_table(index=["timestamp", "expiry", "strike"], columns="option_type", values="implied_vol", aggfunc="first")
    pv = pv.reindex(columns=["call", "put"])
    paired = pv.notna().all(axis=1)
    cp = df.set_index(["timestamp", "expiry", "strike"]).index
    diff = (pv.call - pv.put).abs().reindex(cp).to_numpy()
    ck.add("call_put_iv_equal", pd.Series(paired.reindex(cp).to_numpy(), index=df.index),
           pd.Series(diff > 0, index=df.index))
    ck.add("call_put_unpaired", everything, pd.Series(~paired.reindex(cp).to_numpy(), index=df.index))

    # tau vs (expiry - raw_timestamp) on a 365.25-day year; also the error a 365-day assumption would carry.
    dt_s = (df.expiry - df.snap_time).dt.total_seconds()
    tau_err = df.tau_years * year_s - dt_s
    ck.add("tau_vs_expiry_minus_raw_ts", everything, tau_err.abs() > ic["tau_tol_s"], {"tau_err_s": tau_err})
    tenor_err = df.tenor_days - df.tau_years * 365.0
    ck.add("tenor_days_eq_tau_x_365", everything, tenor_err.abs() > 1e-9)
    tau365_err_s = (df.tau_years * 365 * 86400 - dt_s).abs()

    # Stale <=> price null; IV kept on stale rows.
    ck.add("price_null_iff_stale", everything, df.price_null != df.is_stale)
    ck.add("stale_iv_null", df.is_stale, df.implied_vol.isna())
    ck.add("iv_null", everything, df.implied_vol.isna())

    # IV range; floor reported separately.
    at_floor = (df.implied_vol - ic["iv_floor"]).abs() <= ic["iv_floor_tol"]
    ck.add("iv_out_of_range", df.implied_vol.notna(), (df.implied_vol < ic["iv_min"]) | (df.implied_vol > ic["iv_max"]))
    ck.add("iv_at_floor", df.implied_vol.notna(), at_floor)
    ck.add("iv_below_floor", df.implied_vol.notna(), df.implied_vol < ic["iv_floor"] - ic["iv_floor_tol"])

    # Rows whose snapshot is older than their curve's snapshot (lingering copies, e.g. old atm rows).
    older = df.raw_timestamp < df.groupby(["timestamp", "expiry"]).raw_timestamp.transform("max")
    ck.add("row_older_than_curve_snapshot", everything, older)
    ck.add("row_older_than_curve_unflagged", everything, older & ~df.is_stale)
    ck.add("atm_rows_per_curve_gt_2", df.is_atm, df.groupby(["timestamp", "expiry"]).is_atm.transform("sum") > 2)

    # Misc structural.
    ck.add("surface_iv_eq_implied_vol", everything, ~((df.surface_iv == df.implied_vol) | (df.surface_iv.isna() & df.implied_vol.isna())))
    ck.add("forward_eq_underlying_price", everything, df.forward_price != df.underlying_price)
    ck.add("duplicate_key", everything, df.duplicated(["timestamp", "instrument_name"], keep=False))
    ck.add("atm_strike_eq_forward", df.is_atm, (df.strike - df.forward_price).abs() > 1e-9 * df.forward_price)
    ck.add("vol_curve_check_not_good", everything, df.vol_curve_check.ne("GOOD"))
    ck.add("r_squared_null", everything, df.r_squared.isna())

    # Flag combinations (overlap accounting).
    flags = pd.DataFrame({c: df[c].fillna(0).astype(float).astype(bool) for c in FLAG_COLS})
    flags["quote_quality_flag"] = df.quote_quality_flag.astype(str)
    flags["is_ffill"] = df.is_ffill
    combos = flags.value_counts().rename("n").reset_index()
    combos.insert(0, "date", date)

    # IV floor points.
    fl = df.loc[at_floor & ~df.is_atm, ["timestamp", "expiry", "expiry_label", "strike", "option_type", "tau_years",
                                        "forward_price", "snap_time", "is_ffill", "is_stale"]].copy()
    fl["days"] = (fl.expiry - fl.snap_time).dt.total_seconds() / 86400  # calendar days
    fl["logm"] = np.log(fl.strike / fl.forward_price)
    fl.insert(0, "date", date)

    arb = no_arbitrage(df, cfg)
    arb.insert(0, "date", date)
    misc = {"date": date, "tau365_max_err_h": float(tau365_err_s.max() / 3600) if len(df) else np.nan,
            "b76_max_abs_err": float(abs_err[has_px].max()) if has_px.any() else np.nan,
            "b76_max_rel_err_px_gt_abs_tol": float(rel_err[has_px & (np.abs(model) > ic["b76_abs_tol"])].max())
            if (has_px & (np.abs(model) > ic["b76_abs_tol"])).any() else np.nan}
    return {
        "counts": pd.DataFrame(ck.counts),
        "examples": pd.concat(ck.examples, ignore_index=True) if ck.examples else pd.DataFrame(),
        "flag_combos": combos,
        "iv_floor": fl,
        "arb": arb,
        "misc": misc,
    }


def no_arbitrage(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per real snapshot label: butterfly (adjacent call slopes non-decreasing in K) and calendar
    (total variance non-decreasing in tenor at fixed log-moneyness) violation counts.
    Uses listed, non-stale, priced call rows of non-forward-filled expiries only."""
    ic = cfg["invariants"]
    g = ic["calendar_k_grid"]
    kgrid = np.arange(g["min"], g["max"] + 1e-12, g["step"])
    calls = df[(df.option_type == "call") & ~df.is_atm & ~df.price_null & ~df.is_ffill]
    rows = []
    for ts, snap in calls.groupby("timestamp"):
        bf_checked = bf_viol = mono_viol = 0
        worst_bf = 0.0
        curves = []
        for exp, c in snap.groupby("expiry"):
            c = c.sort_values("strike")
            K, C = c.strike.to_numpy(), c.mid.to_numpy()
            if len(K) >= 2:
                slope = np.diff(C) / np.diff(K)
                mono_viol += int(((slope > ic["butterfly_slope_tol"]) | (slope < -1 - ic["butterfly_slope_tol"])).sum())
                if len(K) >= 3:
                    ds = np.diff(slope)
                    bf_checked += len(ds)
                    bf_viol += int((ds < -ic["butterfly_slope_tol"]).sum())
                    worst_bf = min(worst_bf, float(ds.min()))
            F, T = c.forward_price.iloc[0], c.tau_years.iloc[0]
            k = np.log(K / F)
            w = c.implied_vol.to_numpy() ** 2 * T
            curves.append((T, k, w, c.expiry_label.iloc[0]))
        curves.sort(key=lambda x: x[0])
        cal_checked = cal_viol = 0
        cal_pairs = []
        for (T0, k0, w0, l0), (T1, k1, w1, l1) in zip(curves[:-1], curves[1:]):
            m = (kgrid >= max(k0.min(), k1.min())) & (kgrid <= min(k0.max(), k1.max()))
            if not m.any():
                continue
            wa, wb = np.interp(kgrid[m], k0, w0), np.interp(kgrid[m], k1, w1)
            v = wb < wa - ic["calendar_w_tol"]
            cal_checked += int(m.sum())
            cal_viol += int(v.sum())
            if v.any():
                cal_pairs.append(f"{l0}>{l1}")
        rows.append({"timestamp": ts, "n_expiries": len(curves), "butterfly_checked": bf_checked,
                     "butterfly_viol": bf_viol, "butterfly_worst_dslope": worst_bf, "call_slope_bound_viol": mono_viol,
                     "calendar_checked": cal_checked, "calendar_viol": cal_viol, "calendar_viol_pairs": ",".join(cal_pairs)})
    return pd.DataFrame(rows, columns=["timestamp", "n_expiries", "butterfly_checked", "butterfly_viol",
                                       "butterfly_worst_dslope", "call_slope_bound_viol", "calendar_checked",
                                       "calendar_viol", "calendar_viol_pairs"])
