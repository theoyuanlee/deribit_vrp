"""clean_report.md."""
from __future__ import annotations

import pandas as pd

from audit.report import md_table, pct


def rule_table(ledger: pd.DataFrame) -> pd.DataFrame:
    order = list(dict.fromkeys(ledger.rule))
    t = ledger.groupby("rule", sort=False)[["rows_in", "rows_removed", "curves_in", "curves_removed"]].sum().reindex(order)
    n0r, n0c = t.rows_in.iloc[0], t.curves_in.iloc[0]
    t["rows_removed_pct_of_step"] = 100 * t.rows_removed / t.rows_in
    t["rows_removed_pct_of_raw"] = 100 * t.rows_removed / n0r
    t["curves_removed_pct_of_step"] = 100 * t.curves_removed / t.curves_in.where(t.curves_in > 0)
    t["curves_removed_pct_of_raw"] = 100 * t.curves_removed / n0c
    return t.reset_index()


def monthly(curves: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    c = curves.assign(month=curves.timestamp.dt.strftime("%Y-%m"))
    raw = ledger[ledger.rule == ledger.rule.iloc[0]].assign(month=lambda x: x.date.str[:7]).groupby("month").curves_in.sum()
    g = c.groupby("month")
    t = pd.DataFrame({
        "raw_curves": raw,
        "usable_curves": g.size(),
        "labels_with_curves": g.timestamp.nunique(),
        "median_curves_per_label": c.groupby(["month", "timestamp"]).size().groupby("month").median(),
        "curves_ge_7d": g.tenor_days_cal.apply(lambda s: (s >= 7).sum()),
        "curves_ge_30d": g.tenor_days_cal.apply(lambda s: (s >= 30).sum()),
        "labels_bracketing_30d": c.groupby(["month", "timestamp"]).tenor_days_cal
            .apply(lambda s: (s <= 30).any() and (s > 30).any()).groupby("month").sum(),
        "no_atm": g.has_atm.apply(lambda s: (~s).sum()),
        "with_butterfly_viol": g.n_butterfly_viol.apply(lambda s: (s > 0).sum()),
        "with_calendar_viol": g.n_calendar_viol.apply(lambda s: (s > 0).sum()),
    })
    t["usable_pct"] = 100 * t.usable_curves / t.raw_curves
    return t.reset_index().rename(columns={"index": "month"})


def markdown(ledger, curves, points, und_stats, und, clock_tab, clock_dates, clock_stats, cfg, meta) -> str:
    L = []
    w = L.append
    rc = cfg["rules"]
    w("# 清洗层报告（研究面板）\n")
    w(f"- 期权日期：{meta['opt_dates']}；标的：{meta['und_range']}；生成：{meta['generated']}；配置 `config/clean.yaml`。")
    w("- `data/raw/` 只读；面板中不含原始 `implied_vol`，只保存 w = implied_vol² × tau_years 以及 tau_cal / tau_vol。\n")

    w("## 约定\n")
    w(f"- `obs_time` = 该曲线（保留行中）最新的 raw_timestamp；年龄 = timestamp + 1h − obs_time。")
    w(f"- `tau_cal` = (expiry − obs_time) / {cfg['conventions']['year_days']:g} 天。")
    w(f"- `clock_regime`：`calendar`（{cfg['clock']['switch_time']} 之前，tau_years 与 tau_cal 相差 ≤ {cfg['clock']['calendar_tol_s']:g}s）；"
      "`calendar_anomaly`（切换前但 tau_years 与日历时间不一致）；`vol_time`（切换之后）。")
    w("- `tau_vol` = tau_years（`vol_time` 与 `calendar_anomaly`），`calendar` 时为 NaN。"
      "因此任何曲线都可以用 IV² = w / (tau_vol if notna else tau_cal) 还原原始 IV。")
    w("- `atm_w`、点的 `w` 都按原始 tau_years 计算，价格可由 w 和 forward 用无折现 Black-76 重算（σ√T = √w）。\n")

    w("## 规则剔除（按顺序执行）\n")
    rt = rule_table(ledger)
    w(md_table(rt, floatfmt="{:.2f}"))
    n0r, n0c = rt.rows_in.iloc[0], rt.curves_in.iloc[0]
    w(f"原始 {n0r} 行 / {n0c} 条曲线 → 保留 {n0r - rt.rows_removed.sum()} 行（{pct(n0r - rt.rows_removed.sum(), n0r)}）/ "
      f"{len(curves)} 条曲线（{pct(len(curves), n0c)}）。每步的百分比相对于进入该步的数量（`_of_step`）和原始数量（`_of_raw`）。\n")
    w(f"规则 4（IV = {rc['iv_floor']} 下限）不删除：保留的点中 {int(points.iv_floor.sum())} 个带 `iv_floor` 标记，"
      f"涉及 {int((curves.n_iv_floor > 0).sum())} 条曲线。\n")

    w("## 保留曲线的质量标记\n")
    flags = pd.DataFrame([
        ("clock_regime = calendar", int((curves.clock_regime == "calendar").sum())),
        ("clock_regime = calendar_anomaly", int((curves.clock_regime == "calendar_anomaly").sum())),
        ("clock_regime = vol_time", int((curves.clock_regime == "vol_time").sum())),
        ("无 atm 行（atm_w 为空）", int((~curves.has_atm).sum())),
        ("在部分更新（mixed）小时中", int(curves.in_mixed_hour.sum())),
        ("该标签新出现的到期日", int(curves.is_new_in_label.sum())),
        ("含 IV 下限点", int((curves.n_iv_floor > 0).sum())),
        ("有 call/put 不成对的行权价", int((curves.n_unpaired > 0).sum())),
        ("保留行中仍有 is_stale", int((curves.n_stale_rows > 0).sum())),
        ("曲线内 tau_years 不唯一", int((curves.n_tau > 1).sum())),
        ("蝶式违反 > 0（由 w 重算 call 价）", int((curves.n_butterfly_viol > 0).sum())),
        ("日历违反 > 0（对比同标签下更短的保留到期日）", int((curves.n_calendar_viol > 0).sum())),
    ], columns=["标记", "曲线数"])
    flags["占比"] = [pct(v, len(curves)) for v in flags["曲线数"]]
    w(md_table(flags))
    w(f"点（行权价）：{len(points)}；call/put w 不一致的点 {int((points.w_put_minus_call.fillna(0) != 0).sum())}；"
      f"is_stale 点 {int(points.is_stale.sum())}。\n")

    w("## 每月可用曲线数\n")
    w(md_table(monthly(curves, ledger), floatfmt="{:.1f}"))

    w("## 标的\n")
    w(md_table(pd.DataFrame(list(und_stats.items()), columns=["项目", "值"])))
    zr = und[und.in_zero_run]
    if len(zr):
        runs = zr.assign(g=(zr.timestamp.diff() != pd.Timedelta("1h")).cumsum()).groupby("g").agg(
            start=("timestamp", "min"), end=("timestamp", "max"), hours=("timestamp", "size"))
        w("被置为缺失的零收益片段：\n"); w(md_table(runs.reset_index(drop=True)))
    mh = und[und.ret_span_h > 1][["timestamp", "ret_span_h", "log_ret"]]
    w("跨缺口的多小时收益（`ret_span_h` > 1）：\n"); w(md_table(mh, max_rows=40))

    w("## Hour-of-week vol-time 权重\n")
    s = clock_stats
    if s.get("n_obs"):
        tz = cfg["clock"]["local_tz"]
        w(f"- 观测：{s['n_obs']} 个标签（{s['first']:%Y-%m-%d} 至 {s['last']:%Y-%m-%d}；EDT {s['n_obs_edt']}，EST {s['n_obs_est']}），"
          f"每个标签取日历期限 > {cfg['clock']['min_tenor_days']} 天曲线的中位数；同一标签不同曲线之间的跨度中位数 {s['median_cross_curve_spread']:.2e}。")
        w(f"- **时钟混合了两种时区**：美盘那一段跟随 {tz}（美国夏令时切换时在 UTC 下平移 1 小时），其余时段固定在 UTC。"
          "任何单一时区下都不是周期的，因此按 UTC hour-of-week、分 EDT / EST 两个时段分别估计。"
          f"两套权重有 {s['hours_differing_edt_est']} 个小时不同。使用时按该小时 {tz} 是否处于夏令时选列。")
        w(f"- 权重之和：EDT {s['week_sum_edt']:.3f}，EST {s['week_sum_est']:.3f}（日历时钟为 168）；缺估计的小时：EDT {s['hours_missing_edt']}，EST {s['hours_missing_est']}。")
        w(f"- 残差（观测 − 同时段同小时的中位数）：|resid| > 1e-3 的比例 {100 * s['frac_resid_gt_1e-3']:.2f}%，> 1e-2 的比例 {100 * s['frac_resid_gt_1e-2']:.2f}%，"
          f"涉及 {s['n_dates_resid_gt_1e-2']} 个日期。"
          + ("在各自时段内**不是**严格周期性的，偏离最大的日期见下表。" if s["frac_resid_gt_1e-2"] > 0 else "在各自时段内严格周期。") + "\n")
        for reg in ["edt", "est"]:
            piv = clock_tab.pivot(index="hour_utc", columns="dow_utc", values=f"weight_{reg}")[["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]]
            w(f"{reg.upper()} 时段权重（行 = UTC 小时，列 = UTC 星期）：\n"); w(md_table(piv.reset_index(), floatfmt="{:.4f}"))
            w("每天合计：" + "，".join(f"{k} {v:.3f}" for k, v in piv.sum().items()) + "\n")
        w(f"残差最大的 {cfg['clock']['top_residual_dates']} 个日期：\n")
        w(md_table(clock_dates.head(cfg["clock"]["top_residual_dates"]), floatfmt="{:.4f}"))
    else:
        w("_所选日期中没有切换后的连续观测。_\n")

    w("## 输出\n")
    for f in meta["files"]:
        w(f"- `{f}`")
    return "\n".join(L) + "\n"
