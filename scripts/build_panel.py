#!/usr/bin/env python
"""Build the definition-agnostic research panel (cleaning layer) into data/processed/.

Examples:
    python scripts/build_panel.py
    python scripts/build_panel.py --dates 2025-04-24,2025-06-27,2025-06-28 --out data/processed/sample
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit import io  # noqa: E402
from audit import options as aopt  # noqa: E402
from clean import clock, panel, report, underlying  # noqa: E402
import build_info  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "config" / "clean.yaml"))
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--dates", help="comma-separated option dates (overrides --start/--end)")
    p.add_argument("--out")
    return p.parse_args()


def select_dates(rng, args):
    full = io.date_range(rng["start"], rng["end"])
    if args.dates:
        return [d for d in (x.strip() for x in args.dates.split(",")) if d in full]
    lo, hi = args.start or rng["start"], args.end or rng["end"]
    return [d for d in full if lo <= d <= hi]


def main():
    started = build_info.now_utc()
    args = parse_args()
    cfg = io.load_config(args.config)
    acfg = panel.audit_cfg(cfg)
    root = ROOT / cfg["paths"]["raw_root"]
    out = Path(args.out) if args.out else ROOT / cfg["paths"]["out_dir"]
    out = out if out.is_absolute() else ROOT / out
    if root.resolve() in [out.resolve(), *out.resolve().parents]:
        sys.exit(f"refusing to write inside raw data tree: {out}")
    out.mkdir(parents=True, exist_ok=True)
    asset = cfg["asset"]
    dates = select_dates(cfg["ranges"]["options"], args)
    print(f"options: {len(dates)} days -> {out}")

    state, last = aopt.new_state(), None
    curves, points, pairs, ledger = [], [], [], []
    t0 = time.time()
    for i, d in enumerate(dates):
        prev_day = (pd.Timestamp(d) - pd.Timedelta("1D")).strftime("%Y-%m-%d")
        if last != prev_day:  # seed forward-fill state from the previous day's file
            pdf, _, _ = io.read_day(root, "options", asset, prev_day)
            if pdf is not None and len(pdf):
                _, state = aopt.expiry_snapshots(aopt.prepare(pdf), state, acfg)
        last = d
        df, _, _ = io.read_day(root, "options", asset, d)
        if df is None or df.empty:
            continue
        df = aopt.prepare(df)
        es, state = aopt.expiry_snapshots(df, state, acfg)
        c, p, pr, led = panel.clean_day(df, es, d, cfg)
        curves.append(c), points.append(p), pairs.append(pr), ledger.extend(led)
        if (i + 1) % 25 == 0 or i + 1 == len(dates):
            print(f"  {i + 1}/{len(dates)} {d} ({time.time() - t0:.0f}s)", flush=True)
    curves = pd.concat([c for c in curves if len(c)], ignore_index=True)
    points = pd.concat([p for p in points if len(p)], ignore_index=True)
    pairs = pd.concat([p for p in pairs if len(p)], ignore_index=True)
    ledger = pd.DataFrame(ledger, columns=panel.LEDGER_COLS)

    # Underlying: always the full year (zero-run detection needs the continuous series).
    ur = cfg["ranges"]["underlyings"]
    und, und_stats = underlying.build(underlying.load_raw(root, asset, io.date_range(ur["start"], ur["end"])),
                                      ur["start"], ur["end"], cfg)
    clock_tab, clock_dates, clock_stats = clock.estimate(curves, cfg)

    curves.to_parquet(out / "surface_curves.parquet", index=False)
    points.to_parquet(out / "surface_points.parquet", index=False)
    pairs.to_parquet(out / "surface_calendar_pairs.parquet", index=False)
    und.to_parquet(out / "underlying_hourly.parquet", index=False)
    clock_tab.to_parquet(out / "hour_of_week_clock.parquet", index=False)
    clock_dates.to_parquet(out / "hour_of_week_clock_residual_dates.parquet", index=False)
    ledger.to_parquet(out / "clean_rule_ledger_daily.parquet", index=False)
    meta = {"opt_dates": f"{dates[0]} 至 {dates[-1]}（{len(dates)} 天）" if len(dates) > 3 else ", ".join(dates),
            "und_range": f"{ur['start']} 至 {ur['end']}", "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "files": sorted(p.name for p in out.glob("*.parquet"))}
    (out / "clean_report.md").write_text(report.markdown(ledger, curves, points, pairs, und_stats, und, clock_tab, clock_dates,
                                                         clock_stats, cfg, meta))
    build_info.write(out, ROOT, "scripts/build_panel.py", Path(args.config), vars(args), started)
    print(f"done -> {out / 'clean_report.md'}  curves={len(curves)} points={len(points)} pairs={len(pairs)}")


if __name__ == "__main__":
    main()
