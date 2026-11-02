"""clean_report.md."""
from __future__ import annotations

import numpy as np
import pandas as pd

from audit.report import md_table, pct
from clean import filters

KEYS = ["timestamp", "expiry"]

# Why each vol_curve_check value is in its class (shown in the report; see also config/clean.yaml).
VENDOR_REASONS = {
    "EARLY_EXIT_INSUFFICIENT_STRIKES": "按输入的行权价覆盖判断，不是按 IV 数值；smile 形状与相邻 GOOD 曲线几乎重合，但供应商自己没有完成拟合。",
    "FAILED_PIPELINE": "拟合流程失败；smile 是平的或折线，全部在 1 天以内。",
    "MONEYNESS_DISCREPANCY": "r² = 0、smile 锯齿状，数据本身不一致；全部在 1 天以内。",
    "IV_RV_DIFF": "按 IV 与 RV 的差距筛选——正是 VRP 本身，删掉会截掉 VRP 的尾部；smile 与前后 38–50 小时的 GOOD 曲线重合。"
                  "集中在 2025-06-30 至 07-09 的远期限曲线（27MAR26、26DEC25），紧跟 06-28 的 tau 时钟切换，可能是切换本身造成的。",
    "BAD_BASE_VAL": "按名字是对水平值（base value）的检查；smile 与相邻 GOOD 曲线重合；在 2025-10-10、10-11 暴跌时出现，删掉会截掉极端行情。",
    "VOL_SPIKE": "按名字是对波动率跳升的筛选；长期限例子翼部偏低（这部分已由 flag_low_r2 标出），短期限例子正常。",
    "BAD_SHAPE": "形状检查，两种情况都有：有的翼部被截平（拟合问题），也有 2026-02-05 那种整体比相邻曲线高 20–30 个点、可能是真实跳升的情况；"
                 "默认保留，研究时把剔除它作为敏感性检验。",
}


def delete_table(ledger: pd.DataFrame) -> pd.DataFrame:
    d = ledger[ledger.action == "delete"]
    order = list(dict.fromkeys(d.rule))
    t = d.groupby("rule", sort=False)[["rows_in", "rows_removed", "curves_in", "curves_removed"]].sum().reindex(order).astype(int)
    n0r, n0c = t.rows_in.iloc[0], t.curves_in.iloc[0]
    t["rows_removed_pct_of_raw"] = 100 * t.rows_removed / n0r
    t["curves_removed_pct_of_raw"] = 100 * t.curves_removed / n0c
    return t.reset_index()


def flag_table(ledger: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = ledger[ledger.action == "flag"]
    order = list(dict.fromkeys(f.rule))
    t = f.groupby("rule", sort=False)[["rows_in", "rows_flagged", "curves_in", "curves_flagged"]].sum().reindex(order).astype(int)
    t["curves_flagged_pct"] = 100 * t.curves_flagged / t.curves_in
    cls = filters.vendor_class_map(cfg)
    t["class"] = [cls.get(r.split("=", 1)[1], "") if r.startswith("3c_vendor_check=") else "" for r in t.index]
    return t.reset_index()[["rule", "class", "curves_flagged", "curves_flagged_pct", "rows_flagged", "curves_in", "rows_in"]]


def flag_matrix(curves: pd.DataFrame) -> pd.DataFrame:
    """Curve-level indicator columns for each flag/class, used for overlap tables."""
    return pd.DataFrame({
        "late_snapshot": curves.flag_late_snapshot, "short_tenor": curves.flag_short_tenor,
        "low_r2": curves.flag_low_r2, "fit_quality": curves.vendor_check_class == "fit_quality",
        "value_screen": curves.vendor_check_class == "value_screen", "ambiguous": curves.vendor_check_class == "ambiguous"})


def overlap_table(curves: pd.DataFrame) -> pd.DataFrame:
    m = flag_matrix(curves).astype(int)
    return (m.T @ m).rename_axis("标记（行 ∩ 列）").reset_index()


def combo_table(curves: pd.DataFrame) -> pd.DataFrame:
    m = flag_matrix(curves)
    combo = m.apply(lambda r: " + ".join(c for c in m.columns if r[c]) or "（无标记）", axis=1)
    return combo.value_counts().rename_axis("标记组合").rename("曲线数").reset_index()


def bracket_labels(curves: pd.DataFrame, target_days: float) -> int:
    """Labels at which some curve has tenor <= target and some curve has tenor > target (bar-end tenor)."""
    d = (curves.expiry - (curves.timestamp + pd.Timedelta("1h"))).dt.total_seconds() / 86400
    g = pd.DataFrame({"timestamp": curves.timestamp, "lo": d <= target_days, "hi": d > target_days}).groupby("timestamp")
    return int((g.lo.any() & g.hi.any()).sum())


def former_rules_mask(curves: pd.DataFrame) -> pd.Series:
    """The curve set the former deletion rules kept (fixed; used only for comparisons with the old panel)."""
    return ~curves.flag_late_snapshot & ~curves.flag_short_tenor & ~curves.flag_low_r2 & (curves.vendor_check == "GOOD")


def calendar_change(curves: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    """For curves the former rules kept: calendar violations vs the next-shorter curve among the former set
    (what the old panel reported) against the next-shorter curve in the full panel (what n_calendar_viol is now)."""
    old = curves[former_rules_mask(curves)].sort_values(KEYS)
    prev_exp = old.groupby("timestamp").expiry.shift(1)
    key = pd.DataFrame({"timestamp": old.timestamp, "expiry_short": prev_exp, "expiry_long": old.expiry}).dropna()
    old_viol = key.merge(pairs[["timestamp", "expiry_short", "expiry_long", "n_calendar_viol"]],
                         on=["timestamp", "expiry_short", "expiry_long"], how="left")
    old_viol = old_viol.set_index(["timestamp", "expiry_long"]).n_calendar_viol
    o = old.set_index(KEYS)
    before = old_viol.reindex(o.index).fillna(0).astype(int)
    full = curves.sort_values(KEYS)
    full_prev = full.assign(prev=full.groupby("timestamp").expiry.shift(1)).set_index(KEYS).prev.reindex(o.index)
    old_prev = pd.Series(prev_exp.to_numpy(), index=o.index)
    neighbour_changed = ~((full_prev == old_prev) | (full_prev.isna() & old_prev.isna()))
    changed = before != o.n_calendar_viol
    return {"n_former": len(o), "n_neighbour_changed": int(neighbour_changed.sum()), "n_changed": int(changed.sum()),
            "n_changed_without_neighbour_change": int((changed & ~neighbour_changed).sum()),
            "n_up": int((o.n_calendar_viol > before).sum()), "n_down": int((o.n_calendar_viol < before).sum()),
            "curves_with_viol_before": int((before > 0).sum()), "curves_with_viol_now": int((o.n_calendar_viol > 0).sum()),
            "old_n_calendar_viol": before}


def monthly(curves: pd.DataFrame, ledger: pd.DataFrame, keep_default: pd.Series) -> pd.DataFrame:
    c = curves.assign(month=curves.timestamp.dt.strftime("%Y-%m"), default=keep_default.to_numpy())
    first_rule = ledger[ledger.action == "delete"].rule.iloc[0]
    raw = ledger[ledger.rule == first_rule].assign(month=lambda x: x.date.str[:7]).groupby("month").curves_in.sum()
    g = c.groupby("month")
    d = c[c.default]
    t = pd.DataFrame({
        "raw_curves": raw,
        "panel_curves": g.size(),
        "research_default_curves": d.groupby("month").size(),
        "default_labels": d.groupby("month").timestamp.nunique(),
        "default_median_curves_per_label": d.groupby(["month", "timestamp"]).size().groupby("month").median(),
        "default_labels_bracketing_7d": d.groupby("month").apply(lambda x: bracket_labels(x, 7)),
        "default_labels_bracketing_30d": d.groupby("month").apply(lambda x: bracket_labels(x, 30)),
        "panel_with_butterfly_viol": g.n_butterfly_viol.apply(lambda s: (s > 0).sum()),
        "panel_with_calendar_viol": g.n_calendar_viol.apply(lambda s: (s > 0).sum()),
    })
    t["default_pct_of_raw"] = 100 * t.research_default_curves / t.raw_curves
    return t.reset_index().rename(columns={"index": "month"})


def markdown(ledger, curves, points, pairs, und_stats, und, clock_tab, clock_dates, clock_stats, cfg, meta) -> str:
    L = []
    w = L.append
    rc = cfg["rules"]
    keep_default, filter_items = filters.research_mask(curves, cfg=cfg)
    w("# 清洗层报告（研究面板）\n")
    w(f"- 期权日期：{meta['opt_dates']}；标的：{meta['und_range']}；生成信息（提交、配置、时间）见同目录下的 `_build_info.json`。")
    w("- `data/raw/` 只读；面板中不含原始 `implied_vol`，只保存 w = implied_vol² × tau_years 以及 tau_cal / tau_vol。")
    w("- **原则：面板只删除不是真实观测的数据（复制的曲线、曲线内残留的旧行），其余真实观测即使质量可疑也保留，只加标记。**"
      "研究代码统一通过 `clean.filters.apply_research_filter()` 现场决定用哪些曲线；面板里不保存“是否通过”的列。\n")

    w("## 约定\n")
    w(f"- `obs_time` = 该曲线（保留行中）最新的 raw_timestamp；年龄 = timestamp + 1h − obs_time。")
    w(f"- `tau_cal` = (expiry − obs_time) / {cfg['conventions']['year_days']:g} 天。")
    w(f"- `clock_regime`：`calendar`（{cfg['clock']['switch_time']} 之前，tau_years 与 tau_cal 相差 ≤ {cfg['clock']['calendar_tol_s']:g}s）；"
      "`calendar_anomaly`（切换前但 tau_years 与日历时间不一致）；`vol_time`（切换之后）。")
    w("- `tau_vol` = tau_years（`vol_time` 与 `calendar_anomaly`），`calendar` 时为 NaN。"
      "因此任何曲线都可以用 IV² = w / (tau_vol if notna else tau_cal) 还原原始 IV。")
    w("- `atm_w`、点的 `w` 都按原始 tau_years 计算，价格可由 w 和 forward 用无折现 Black-76 重算（σ√T = √w）。\n")

    w("## 规则：删除（按顺序执行）\n")
    dt = delete_table(ledger)
    w(md_table(dt, floatfmt="{:.2f}"))
    n0r, n0c = int(dt.rows_in.iloc[0]), int(dt.curves_in.iloc[0])
    w(f"原始 {n0r} 行 / {n0c} 条曲线 → 面板 {n0r - dt.rows_removed.sum()} 行（{pct(n0r - dt.rows_removed.sum(), n0r)}）/ "
      f"{len(curves)} 条曲线（{pct(len(curves), n0c)}）。\n")

    w("## 规则：标记（不删除，标记之间可能重叠）\n")
    w(md_table(flag_table(ledger, cfg), floatfmt="{:.2f}"))
    w(f"阈值：`flag_late_snapshot` 年龄 > {rc['max_age_min']:g} 分钟；`flag_short_tenor` 日历 tenor < {rc['min_tenor_days']:g} 天；"
      f"`flag_low_r2` r² < {rc['min_r_squared']:g}（r² 缺失也标记）；`vendor_check` 是 vol_curve_check 的原始值，`vendor_check_class` 是它的分类。\n")
    w(f"规则 4（IV = {rc['iv_floor']} 下限）不删除：面板的点中 {int(points.iv_floor.sum())} 个带 `iv_floor` 标记，"
      f"涉及 {int((curves.n_iv_floor > 0).sum())} 条曲线。\n")
    w("标记重叠（曲线数，对角线为各标记自身）：\n"); w(md_table(overlap_table(curves)))
    w("标记组合：\n"); w(md_table(combo_table(curves)))

    w("### vendor_check 的分类与理由\n")
    cls_map = filters.vendor_class_map(cfg)
    vc = curves.vendor_check.value_counts()
    rows = [{"vol_curve_check": v, "分类": cls_map[v], "面板曲线数": int(vc.get(v, 0)),
             "其中默认过滤后保留": int((curves.vendor_check[keep_default] == v).sum()),
             "理由": VENDOR_REASONS.get(v, "—" if v == "GOOD" else "（未记录理由）")} for v in cls_map]
    w(md_table(pd.DataFrame(rows)))

    w("## 默认研究过滤（`research_default_filter`）\n")
    opts = filters.resolve_options(cfg)
    w("配置：" + "，".join(f"`{k}: {str(v).lower()}`" for k, v in opts.items()) + "。")
    w("每一项单独标记的曲线数，以及按顺序执行时剔除的曲线数：\n"); w(md_table(filter_items))
    w(f"默认过滤后：**{int(keep_default.sum())}** 条曲线（面板 {len(curves)} 条的 {pct(keep_default.sum(), len(curves))}）。\n")
    former = former_rules_mask(curves)
    w(f"与旧版（删除规则）比较：旧规则保留的曲线 {int(former.sum())} 条；默认过滤多保留 "
      f"{int((keep_default & ~former).sum())} 条，其中 value_screen {int((keep_default & (curves.vendor_check_class == 'value_screen')).sum())} 条、"
      f"ambiguous {int((keep_default & (curves.vendor_check_class == 'ambiguous')).sum())} 条。")
    w(f"括住 7 天的标签：旧规则 {bracket_labels(curves[former], 7)}，默认过滤 {bracket_labels(curves[keep_default], 7)}；"
      f"括住 30 天的标签：旧规则 {bracket_labels(curves[former], 30)}，默认过滤 {bracket_labels(curves[keep_default], 30)}。\n")

    w("## 质量标记（面板全部曲线）\n")
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
        ("日历违反 > 0（对比同标签下更短的面板到期日）", int((curves.n_calendar_viol > 0).sum())),
    ], columns=["标记", "曲线数"])
    flags["占比"] = [pct(v, len(curves)) for v in flags["曲线数"]]
    w(md_table(flags))
    w(f"点（行权价）：{len(points)}；call/put w 不一致的点 {int((points.w_put_minus_call.fillna(0) != 0).sum())}；"
      f"is_stale 点 {int(points.is_stale.sum())}。\n")

    w("### 日历违反：期限对明细与旧版的差别\n")
    w(f"`surface_calendar_pairs.parquet` 记录同一快照内**所有**（较短, 较长）到期日组合，共 {len(pairs)} 对，"
      f"其中在面板中相邻的 {int(pairs.adjacent_in_panel.sum())} 对；研究阶段按任意过滤条件重新确定相邻关系后，直接查这张表。"
      "曲线表的 `n_calendar_viol` = 该曲线与面板中紧邻的更短到期日之间的违反数。\n")
    cc = calendar_change(curves, pairs)
    w(f"对旧规则也保留的 {cc['n_former']} 条曲线：面板中紧邻的更短到期日发生变化的有 {cc['n_neighbour_changed']} 条"
      f"（中间插入了原先被删除、现在只加标记的曲线）；`n_calendar_viol` 因此改变的有 {cc['n_changed']} 条"
      f"（变大 {cc['n_up']}，变小 {cc['n_down']}），相邻关系不变而违反数改变的 {cc['n_changed_without_neighbour_change']} 条。"
      f"有日历违反的曲线：旧口径 {cc['curves_with_viol_before']} 条，现口径 {cc['curves_with_viol_now']} 条。\n")

    w("## 每月曲线数\n")
    w(md_table(monthly(curves, ledger, keep_default), floatfmt="{:.1f}"))

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
