#!/usr/bin/env python
"""Run the read-only data-quality audit.

Examples:
    python scripts/run_audit.py                                   # full range from config
    python scripts/run_audit.py --dates 2025-04-24 --out data/audit/sample_2025-04-24
    python scripts/run_audit.py --start 2025-05-01 --end 2025-05-07
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit import alignment, expiries, invariants, io, options, report, underlyings  # noqa: E402
import build_info  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "config" / "audit.yaml"))
    p.add_argument("--start", help="first date (inclusive), applied to both datasets")
    p.add_argument("--end", help="last date (inclusive), applied to both datasets")
    p.add_argument("--dates", help="comma-separated explicit dates (overrides --start/--end)")
    p.add_argument("--out", help="output directory (default: config paths.out_dir)")
    return p.parse_args()


def select_dates(rng: dict, args) -> list[str]:
    full = io.date_range(rng["start"], rng["end"])
    if args.dates:
        want = [d.strip() for d in args.dates.split(",") if d.strip()]
        return [d for d in want if d in full]
    lo, hi = args.start or rng["start"], args.end or rng["end"]
    return [d for d in full if lo <= d <= hi]


def run_options(dates: list[str], root: Path, cfg: dict) -> dict:
    asset = cfg["asset"]
    acc = defaultdict(list)
    state = options.new_state()
    t0 = time.time()
    last = None
    for i, d in enumerate(dates):
        prev_day = (pd.Timestamp(d) - pd.Timedelta("1D")).strftime("%Y-%m-%d")
        if last != prev_day:
            # Seed ffill state from the previous day's file so subsets match a full run at midnight.
            pdf, _, _ = io.read_day(root, "options", asset, prev_day)
            if pdf is not None and len(pdf):
                _, state = options.expiry_snapshots(options.prepare(pdf), state, cfg)
        last = d
        df, rep, man = io.read_day(root, "options", asset, d)
        if rep is not None:
            acc["reports"].append(io.flatten_report(rep, "options", d))
        if man is not None:
            acc["manifest"].append(io.check_manifest(man, "options", d))
        if df is None or df.empty:
            acc["hours"].append(options.hour_grid(d, pd.DataFrame(), pd.DataFrame(), cfg, state))
            continue
        df = options.prepare(df)
        state_before = state
        es, state = options.expiry_snapshots(df, state, cfg)
        es = es.merge(expiries.curve_metrics(df), on=["timestamp", "expiry"], how="left")
        es.insert(0, "date", d)
        hours = options.hour_grid(d, es, df, cfg, state_before)
        if rep is not None:
            acc["xcheck"] += io.crosscheck(rep, options.report_actuals(df, es, hours), "options", d)
        inv = invariants.check_day(df, es, d, cfg)
        acc["es"].append(es)
        acc["hours"].append(hours)
        for k in ["counts", "examples", "flag_combos", "iv_floor", "arb"]:
            acc[f"inv_{k}"].append(inv[k])
        acc["inv_misc"].append(inv["misc"])
        if (i + 1) % 20 == 0 or i + 1 == len(dates):
            print(f"  options {i + 1}/{len(dates)} {d}  ({time.time() - t0:.0f}s)", flush=True)
    cat = lambda k: pd.concat([x for x in acc[k] if len(x)], ignore_index=True) if any(len(x) for x in acc[k]) else pd.DataFrame()
    es = cat("es")
    if len(es):
        es["days"] = (es.expiry - es.snap_time).dt.total_seconds() / 86400  # calendar days
        es["tau_days"] = es.tau_years * cfg["conventions"]["year_days"]
        es["tenor_bucket"] = expiries.tenor_bucket(es.days, cfg["expiries"]["tenor_buckets_days"])
    max_total = cfg["invariants"]["max_examples_per_check_total"]
    ex = cat("inv_examples")
    if len(ex):
        ex = ex.groupby("check").head(max_total).reset_index(drop=True)
    return {
        "es": es, "hours": cat("hours"), "reports": pd.DataFrame(acc["reports"]), "manifest": pd.DataFrame(acc["manifest"]),
        "xcheck": pd.DataFrame(acc["xcheck"]), "inv_counts": cat("inv_counts"), "inv_examples": ex,
        "flag_combos": cat("inv_flag_combos"), "iv_floor": cat("inv_iv_floor"), "arb": cat("inv_arb"),
        "inv_misc": acc["inv_misc"],
    }


def main():
    started = build_info.now_utc()
    args = parse_args()
    cfg = io.load_config(args.config)
    root = ROOT / cfg["paths"]["raw_root"]
    out = Path(args.out) if args.out else ROOT / cfg["paths"]["out_dir"]
    if not out.is_absolute():
        out = ROOT / out
    if root.resolve() in [out.resolve(), *out.resolve().parents]:
        sys.exit(f"refusing to write inside raw data tree: {out}")
    asset = cfg["asset"]
    opt_dates = select_dates(cfg["ranges"]["options"], args)
    und_dates = select_dates(cfg["ranges"]["underlyings"], args)
    print(f"options: {len(opt_dates)} days, underlyings: {len(und_dates)} days -> {out}")

    r = run_options(opt_dates, root, cfg)
    r["inventory"] = pd.concat([io.inventory(root, "options", asset, opt_dates),
                                io.inventory(root, "underlyings", asset, und_dates)], ignore_index=True)

    # Underlyings: read whole range at once; pad +/-1 day for alignment lags without extending the audit grid.
    pad = io.date_range((pd.Timestamp(und_dates[0]) - pd.Timedelta("1D")).strftime("%Y-%m-%d"),
                        (pd.Timestamp(und_dates[-1]) + pd.Timedelta("1D")).strftime("%Y-%m-%d")) if und_dates else []
    full = io.date_range(cfg["ranges"]["underlyings"]["start"], cfg["ranges"]["underlyings"]["end"])
    pad_only = [d for d in pad if d in full and d not in und_dates]
    u_all, u_reps, u_mans, u_x = underlyings.load(root, asset, sorted(set(und_dates) | set(pad_only)))
    keep = lambda rows: [x for x in rows if x["date"] in und_dates]
    r["reports"] = pd.concat([r["reports"], pd.DataFrame(keep(u_reps))], ignore_index=True)
    r["manifest"] = pd.concat([r["manifest"], pd.DataFrame(keep(u_mans))], ignore_index=True)
    r["xcheck"] = pd.concat([r["xcheck"], pd.DataFrame(keep(u_x))], ignore_index=True)
    if und_dates and len(u_all):
        r["und"] = underlyings.audit(u_all[u_all.file_date.isin(und_dates)], und_dates[0], und_dates[-1], cfg)
        padded = underlyings.audit(u_all, min(u_all.file_date), max(u_all.file_date), cfg)["series"]
        if len(r["es"]):
            r["align"] = alignment.run(alignment.candidates(r["es"], cfg), padded, cfg)

    s = report.summarise(r, cfg)
    tables = {
        "file_inventory": r["inventory"], "file_reports": r["reports"], "file_manifest_checks": r["manifest"],
        "file_report_crosscheck": r["xcheck"],
        "options_hours": r["hours"], "options_expiry_snapshots": r["es"],
        "options_daily": s["daily"], "options_monthly": s["monthly"], "options_gaps_no_real": s["gaps_real"],
        "options_gaps_no_data": s["gaps_data"], "options_snapshot_minute_dist": s["minute_dist"],
        "invariant_counts_daily": r["inv_counts"], "invariant_counts": s.get("inv"), "invariant_examples": r["inv_examples"],
        "invariant_flag_combos": s.get("flag_combos"), "invariant_iv_floor": r["iv_floor"], "invariant_arbitrage": r["arb"],
        "invariant_misc_daily": pd.DataFrame(r["inv_misc"]),
    }
    for k in ["life", "intermittent", "deribit", "r2_tenor", "low_r2", "bracket", "switches"]:
        tables[f"expiry_{k}"] = s.get(k)
    if "tau_clock" in s:
        tables["expiry_tau_clock_hourly"] = s["tau_clock"]["hourly"]
        tables["expiry_tau_clock_profile"] = s["tau_clock"]["profile"]
        tables["expiry_tau_clock_monthly"] = s.get("tau_monthly")
    if "und" in r:
        for k, v in r["und"].items():
            tables[f"underlying_{k}"] = v
    if "align" in r:
        for k, v in r["align"].items():
            tables[f"alignment_{k}"] = v
    report.write_parquet(tables, out)
    report.figures(r, s, cfg, out)
    meta = {
        "opt_start": opt_dates[0] if opt_dates else "-", "opt_end": opt_dates[-1] if opt_dates else "-", "n_opt_days": len(opt_dates),
        "und_start": und_dates[0] if und_dates else "-", "und_end": und_dates[-1] if und_dates else "-", "n_und_days": len(und_dates),
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "config": Path(args.config).name,
        "files": [p.name for p in out.glob("*.parquet")] + [p.name for p in out.glob("*.png")],
    }
    (out / "audit_summary.md").write_text(report.markdown(r, s, cfg, meta))
    build_info.write(out, ROOT, "scripts/run_audit.py", Path(args.config), vars(args), started)
    print(f"done -> {out / 'audit_summary.md'}")


if __name__ == "__main__":
    main()
