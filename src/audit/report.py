"""Aggregation, parquet output, PNG figures and audit_summary.md."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# CJK-capable fallbacks so Chinese titles render; the first available one is used.
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Hiragino Sans GB", "PingFang SC", "Heiti TC",
                                   "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

from . import expiries, options

# Reference palette (dataviz skill): sequential blues for ordinal tiers, categorical slots 1-3, muted gray.
BLUES = ["#1f5fae", "#5b9be6", "#b5d3f5"]
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
GRAY, INK, MUTED, GRID = "#c9c8c2", "#0b0b0b", "#52514e", "#e6e5e0"


# ---------------------------------------------------------------- helpers
def md_table(df: pd.DataFrame, floatfmt: str = "{:.4g}", max_rows: int | None = None) -> str:
    if df is None or len(df) == 0:
        return "_（无）_\n"
    d = df.head(max_rows) if max_rows else df

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        if isinstance(v, pd.Timestamp):
            return v.strftime("%Y-%m-%d %H:%M") if v.second == 0 else v.strftime("%Y-%m-%d %H:%M:%S")
        return str(v)

    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    if max_rows and len(df) > max_rows:
        lines.append(f"\n_（共 {len(df)} 行，仅显示前 {max_rows} 行；完整明细见 parquet）_")
    return "\n".join(lines) + "\n"


def pct(a, b) -> str:
    return f"{100 * a / b:.1f}%" if b else "n/a"


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def write_parquet(tables: dict[str, pd.DataFrame], out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if df is None:
            continue
        df = df.copy()
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
        df.to_parquet(out / f"{name}.parquet", index=False)


# ---------------------------------------------------------------- summaries
def period_table(hours: pd.DataFrame, es: pd.DataFrame, key: str, tiers: list[str]) -> pd.DataFrame:
    h = hours.assign(month=hours.timestamp.dt.strftime("%Y-%m"))
    g = h.groupby(key)
    t = pd.DataFrame({
        "grid_hours": g.size(),
        "data_hours": g.has_data.sum(),
        "real_snapshot_hours": g.has_real.sum(),
        "all_real_hours": g.hour_type.apply(lambda s: (s == "real").sum()),
        "mixed_hours": g.partial_mixed.sum(),
        "new_expiry_hours": g.partial_new_expiry.sum(),
        "ffill_only_hours": g.hour_type.apply(lambda s: (s == "ffill").sum()),
        "missing_hours": g.hour_type.apply(lambda s: (s == "missing").sum()),
    })
    for tier in tiers:
        t[f"tier_{tier}"] = g.freshness_tier.apply(lambda s, tier=tier: (s == tier).sum())
        t[f"real_tier_{tier}"] = g.apply(lambda x, tier=tier: ((x.freshness_tier == tier) & x.has_real).sum())
    t["stale_rows"] = g.n_stale.sum()
    t["ffill_unflagged_rows"] = g.ffill_unflagged_rows.sum()
    t["real_pct"] = 100 * t.real_snapshot_hours / t.grid_hours
    t[f"fresh_{tiers[0]}_pct"] = 100 * t[f"tier_{tiers[0]}"] / t.grid_hours
    if key == "date":
        t["n_expiries_max"] = g.n_expiries.max()
    return t.reset_index()


def summarise(r: dict, cfg: dict) -> dict:
    """Compute every derived summary table from the raw per-layer results."""
    hours, es = r["hours"], r["es"]
    tiers = options.tier_labels(cfg["snapshot"]["freshness_edges_min"])
    s = {"tiers": tiers}
    s["daily"] = period_table(hours, es, "date", tiers)
    s["monthly"] = period_table(hours, es, "month", tiers)
    top = cfg["snapshot"]["top_gaps"]
    s["gaps_real"] = options.longest_gaps(hours, "has_real", top)
    s["gaps_data"] = options.longest_gaps(hours, "has_data", top)
    rh = hours[hours.has_real]
    s["minute_dist"] = (np.floor(rh.real_snap_offset_min).astype(int).value_counts().sort_index()
                        .rename_axis("minute_after_label").rename("n_hours").reset_index())
    s["es_tier"] = es.freshness_tier.value_counts().rename_axis("tier").rename("n").reset_index()
    s["es_tier_real"] = es[~es.is_ffill].freshness_tier.value_counts().rename_axis("tier").rename("n").reset_index()

    # Expiry layer.
    if len(es):
        s["life"], s["intermittent"] = expiries.lifecycle(es, hours)
        s["deribit"] = expiries.deribit_expected(es, hours, cfg)
        er = es[~es.is_ffill].copy()
        s["r2_tenor"] = expiries.r2_by_tenor(er, cfg)
        thr = cfg["expiries"]["r2_threshold"]
        s["low_r2"] = er.loc[er.r_squared < thr, ["timestamp", "expiry_label", "days", "r_squared", "n_strikes", "atm_iv"]] \
            .sort_values("r_squared").reset_index(drop=True)
        s["bracket"], s["switches"] = expiries.cm_bracket(es, hours, cfg)
        s["tau_clock"] = expiries.tau_clock(es, cfg)
        sw_ts = s["tau_clock"]["switch"]
        c = r["inv_counts"]
        if sw_ts is not None and len(c):
            t = c[c.check == "tau_vs_expiry_minus_raw_ts"].copy()
            t["period"] = np.where(pd.to_datetime(t.date).dt.tz_localize("UTC") < sw_ts.normalize(), "before_switch", "from_switch_day")
            s["tau_split"] = t.groupby("period")[["n_checked", "n_viol", "n_checked_real", "n_viol_real"]].sum().reset_index()
            h = s["tau_clock"]["hourly"]
            h = h.assign(month=h.timestamp.dt.strftime("%Y-%m"))
            s["tau_monthly"] = h.groupby("month").agg(n_labels=("timestamp", "size"), calendar_like_frac=("calendar_like", "mean"),
                                                      offset_h_median=("offset_h", "median"), offset_h_min=("offset_h", "min"),
                                                      offset_h_max=("offset_h", "max"), rate_median=("rate", "median")).reset_index()
    # Invariants.
    c = r["inv_counts"]
    if len(c):
        agg = c.groupby("check", sort=False)[["n_checked", "n_viol", "n_checked_real", "n_viol_real"]].sum().reset_index()
        agg["viol_rate"] = agg.n_viol / agg.n_checked.where(agg.n_checked > 0)
        agg["viol_rate_real"] = agg.n_viol_real / agg.n_checked_real.where(agg.n_checked_real > 0)
        s["inv"] = agg
    fc = r["flag_combos"]
    if len(fc):
        keys = [k for k in fc.columns if k not in ("date", "n")]
        s["flag_combos"] = fc.groupby(keys, dropna=False).n.sum().sort_values(ascending=False).reset_index()
        s["flag_totals"] = pd.DataFrame(
            {"flag": k, "n_rows": int(fc.loc[fc[k].astype(bool), "n"].sum()) if fc[k].dtype == bool else None}
            for k in keys if fc[k].dtype == bool)
    fl = r["iv_floor"]
    if len(fl):
        flr = fl[~fl.is_ffill]
        edges = cfg["expiries"]["tenor_buckets_days"]
        flr = flr.assign(tenor_bucket=expiries.tenor_bucket(flr.days, edges),
                         logm_bucket=pd.cut(flr.logm, [-np.inf, -1, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1, np.inf]).astype(str))
        s["floor_by_expiry"] = flr.groupby("expiry_label").agg(n=("strike", "size"), first=("timestamp", "min"),
                                                               last=("timestamp", "max"), days_min=("days", "min"),
                                                               days_max=("days", "max")).sort_values("n", ascending=False).reset_index()
        s["floor_by_tenor"] = flr.groupby("tenor_bucket").size().rename("n").reset_index()
        s["floor_by_logm"] = flr.groupby(["logm_bucket", "option_type"]).size().unstack(fill_value=0).reset_index()
        s["floor_n_real"], s["floor_n_all"] = len(flr), len(fl)
    arb = r["arb"]
    if len(arb):
        s["arb_totals"] = {k: int(arb[k].sum()) for k in ["butterfly_checked", "butterfly_viol", "call_slope_bound_viol",
                                                          "calendar_checked", "calendar_viol"]}
        s["arb_totals"]["n_snapshots"] = len(arb)
        s["arb_totals"]["snap_with_bfly"] = int((arb.butterfly_viol > 0).sum())
        s["arb_totals"]["snap_with_cal"] = int((arb.calendar_viol > 0).sum())
        pairs = arb.calendar_viol_pairs.str.split(",").explode()
        s["cal_pairs"] = pairs[pairs.astype(bool)].value_counts().rename_axis("pair").rename("n_snapshots").reset_index()
    xc = r["xcheck"]
    if len(xc):
        s["xcheck_summary"] = xc.groupby(["dataset", "field"]).match.agg(n_days="size", n_mismatch=lambda m: int((~m).sum())).reset_index()
    return s


# ---------------------------------------------------------------- figures
def figures(r: dict, s: dict, cfg: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    tiers = s["tiers"]
    hours, es = r["hours"], r["es"]

    # 1. Daily hours by freshness tier.
    d = s["daily"]
    x = pd.to_datetime(d.date)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    bottom = np.zeros(len(d))
    for tier, col in zip(tiers, BLUES):
        v = d[f"tier_{tier}"].to_numpy()
        ax.bar(x, v, bottom=bottom, width=0.9, color=col, label=f"有数据，最大年龄 {tier}", edgecolor="white", linewidth=0.3)
        bottom += v
    ax.bar(x, d.missing_hours, bottom=bottom, width=0.9, color=GRAY, label="无数据", edgecolor="white", linewidth=0.3)
    ax.plot(x, d.real_snapshot_hours, color=INK, lw=1, label="真实快照小时数")
    ax.set_ylim(0, 24.5); ax.set_ylabel("小时 / 天", color=MUTED)
    ax.set_title("每日小时数：按快照新鲜度分档", loc="left", fontsize=11, color=INK)
    ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12))
    _style(ax); fig.tight_layout(); fig.savefig(out / "daily_hours.png", dpi=140); plt.close(fig)

    # 2. Snapshot age distribution per (timestamp, expiry).
    fig, ax = plt.subplots(figsize=(8, 3.6))
    age_min = es.age_s / 60
    bins = np.logspace(-2, np.log10(max(age_min.max(), 100) * 1.1), 70)
    for flag, lab, col in [(False, "真实快照", CAT[0]), (True, "前向填充", CAT[1])]:
        v = age_min[(es.is_ffill == flag) & (age_min > 0)]
        if len(v):
            ax.hist(v, bins=bins, color=col, alpha=0.85, label=f"{lab} (n={len(v)})")
    for e in cfg["snapshot"]["freshness_edges_min"]:
        ax.axvline(e, color=MUTED, lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("快照年龄（分钟，bar 结束时刻 − 快照时间）", color=MUTED); ax.set_ylabel("(timestamp, expiry) 数", color=MUTED)
    ax.set_title("快照年龄分布", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8); _style(ax); fig.tight_layout(); fig.savefig(out / "snapshot_age.png", dpi=140); plt.close(fig)

    # 3. Expiry presence heatmap: share of each day's 24 labels where the expiry is present.
    if len(es):
        p = es.groupby(["expiry_label", "expiry", "date"]).size().rename("n").reset_index()
        p["share"] = p.n / 24
        m = p.pivot_table(index=["expiry", "expiry_label"], columns="date", values="share", fill_value=0)
        m = m.reindex(columns=sorted(hours.date.unique()), fill_value=0).sort_index()
        fig, ax = plt.subplots(figsize=(12, min(10, max(3, 0.12 * len(m) + 1))))
        im = ax.imshow(m.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        step = max(1, len(m) // 40)
        ax.set_yticks(range(0, len(m), step)); ax.set_yticklabels([l for _, l in m.index][::step], fontsize=6)
        xt = list(range(0, m.shape[1], max(1, m.shape[1] // 12)))
        ax.set_xticks(xt); ax.set_xticklabels([m.columns[i] for i in xt], fontsize=7, rotation=30)
        cb = fig.colorbar(im, ax=ax, fraction=0.02); cb.set_label("当日出现的小时比例", fontsize=8)
        ax.set_title("到期日出现情况（行=到期日，列=日期）", loc="left", fontsize=11, color=INK)
        fig.tight_layout(); fig.savefig(out / "expiry_heatmap.png", dpi=140); plt.close(fig)

        # 4. r^2 over time.
        er = es[~es.is_ffill]
        fig, ax = plt.subplots(figsize=(12, 3.6))
        groups = [(er.days < 7, "tenor < 7d"), ((er.days >= 7) & (er.days < 30), "7–30d"), (er.days >= 30, "≥ 30d")]
        for (msk, lab), col in zip(groups, CAT):
            ax.scatter(er.timestamp[msk], er.r_squared[msk], s=3, color=col, alpha=0.5, label=lab, linewidths=0)
        dmin = er.groupby("date").r_squared.min()
        ax.plot(pd.to_datetime(dmin.index).tz_localize("UTC") + pd.Timedelta("12h"), dmin.values, color=INK, lw=0.8, label="当日最低")
        ax.axhline(cfg["expiries"]["r2_threshold"], color=MUTED, ls="--", lw=0.8)
        ax.set_ylim(max(-0.05, er.r_squared.min() - 0.05), 1.01); ax.set_ylabel("r²", color=MUTED)
        ax.set_title("每个（快照, 到期日）曲线拟合 r²", loc="left", fontsize=11, color=INK)
        ax.legend(frameon=False, fontsize=8, ncol=4, markerscale=3); _style(ax)
        fig.tight_layout(); fig.savefig(out / "r2_over_time.png", dpi=140); plt.close(fig)

    # 4b. tau clock offset.
    tc = s.get("tau_clock")
    if tc is not None and len(tc["hourly"]):
        h = tc["hourly"]
        fig, ax = plt.subplots(figsize=(12, 3.4))
        ax.plot(h.timestamp, h.offset_h, ".", ms=1.5, color=CAT[0])
        ax.axhline(0, color=MUTED, lw=0.8)
        if tc["switch"] is not None:
            ax.axvline(tc["switch"], color=CAT[1], lw=1, ls="--")
            ax.text(tc["switch"], ax.get_ylim()[1], f" 切换 {tc['switch']:%Y-%m-%d}", color=INK, fontsize=8, va="top")
        ax.set_ylabel("tau − 日历期限（小时）", color=MUTED)
        ax.set_title(f"tau 时钟：长期限（>{cfg['expiries']['tau_clock_min_days']}d）曲线 tau 相对日历时间的偏差", loc="left", fontsize=11, color=INK)
        _style(ax); fig.tight_layout(); fig.savefig(out / "tau_clock.png", dpi=140); plt.close(fig)

    # 5. Alignment.
    al = r.get("align")
    if al is not None and len(al["detail"]):
        st, det = al["stats"], al["detail"]
        st = st[st.subset == st.subset.iloc[-1]].reset_index(drop=True)  # fresh-only subset
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 3]})
        ax = axes[0]
        ax.bar([f"{L:+d}h" for L in st.lag_h], st.median_abs_d * 1e4, color=CAT[0], width=0.6)
        for i, v in enumerate(st.median_abs_d * 1e4):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_ylabel("median |log(F/close)| (bp)", color=MUTED); ax.set_title(f"按 lag（{st.subset.iloc[0]}）", loc="left", fontsize=10, color=INK); _style(ax)
        ax = axes[1]
        ax.plot(det.timestamp, det["d_lag+0"] * 1e4, ".", ms=1.5, color=CAT[0])
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylabel("log(F/close) lag 0 (bp)", color=MUTED)
        ax.set_title("最短 tenor(>%gh) 期权 forward 相对 perp close：基差随时间" % cfg["alignment"]["min_tenor_h"], loc="left", fontsize=10, color=INK)
        _style(ax); fig.tight_layout(); fig.savefig(out / "alignment.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------- markdown
def markdown(r: dict, s: dict, cfg: dict, meta: dict) -> str:
    hours, es = r["hours"], r["es"]
    tiers = s["tiers"]
    yd = cfg["conventions"]["year_days"]
    G = len(hours)
    L = []
    w = L.append
    w(f"# BTC 期权与标的小时数据审计\n")
    w(f"- 期权日期：{meta['opt_start']} 至 {meta['opt_end']}（{meta['n_opt_days']} 天）；标的：{meta['und_start']} 至 {meta['und_end']}（{meta['n_und_days']} 天）")
    w("- 生成信息（提交、配置、时间）见同目录下的 `_build_info.json`。")
    w("- 本审计只检测、只报告，未修改 `data/raw/` 中任何数据。\n")

    w("## 约定\n")
    tc = s.get("tau_clock", {})
    sw_ts = tc.get("switch")
    w(f"- `tau_years` 按 **{yd:g} 天/年** 计算：tau = (expiry − raw_timestamp)/{yd:g}d（核对容差 {cfg['invariants']['tau_tol_s']} 秒）。"
      f"后续 RV 年化统一使用 **{yd:g}×24 = {yd * 24:g}** 小时/年。")
    if sw_ts is not None:
        w(f"  - **注意：上述日历时钟只在 {sw_ts:%Y-%m-%d %H:%M} UTC 之前成立。** 从该时刻起 `tau_years` 是加权的 vol-time 时钟"
          "（每周总和仍为 7 天，但工作日、周末、日内各时段权重不同），与日历时间相差可达一天以上。详见“C2. tau 时钟”。")
    w("- `tenor_days` 列 = tau_years × 365，不是日历天数。本报告中的“天数/tenor”一律用日历时间 (expiry − 快照时间) 计算；"
      "只有 B76 定价、delta 和日历套利的 total variance 使用原始 `tau_years`（与数据生成方式一致）。")
    w("- 快照单位是（timestamp, expiry）：每个到期日的曲线有自己的 `raw_timestamp`。"
      "某到期日在某标签上的 `raw_timestamp` 与该到期日上一次出现时相同 ⇒ 前向填充（状态跨日传递）。")
    w("- 年龄 = bar 结束时刻（timestamp + 1h）− 快照时间，按到期日计算；小时层取最大年龄。"
      f"新鲜度分档：{', '.join(tiers)}（边界见配置 `snapshot.freshness_edges_min`）。")
    w("- 小时类型：`real`=所有到期日都是新快照；`ffill`=全部被复制；`mixed`=部分新部分复制；"
      "“部分更新”= mixed 或在上一小时有数据的情况下有新到期日出现。")
    w("- `quote_age_ms` 原始列 = timestamp − raw_timestamp（真实快照为负），本审计不使用它。\n")

    # Key numbers.
    w("## 关键数字\n")
    nreal, ndata = int(hours.has_real.sum()), int(hours.has_data.sum())
    tier_counts = hours.freshness_tier.value_counts()
    rows = [
        ("期权 UTC 网格小时数", G, ""),
        ("有数据的小时", ndata, pct(ndata, G)),
        ("真实快照小时（≥1 个到期日为新快照）", nreal, pct(nreal, G)),
        ("全部到期日都新鲜的小时（real）", int((hours.hour_type == "real").sum()), pct((hours.hour_type == "real").sum(), G)),
        ("部分更新小时：mixed", int(hours.partial_mixed.sum()), pct(hours.partial_mixed.sum(), G)),
        ("部分更新小时：新到期日出现", int(hours.partial_new_expiry.sum()), pct(hours.partial_new_expiry.sum(), G)),
        ("全部为复制的小时（ffill）", int((hours.hour_type == "ffill").sum()), pct((hours.hour_type == "ffill").sum(), G)),
        ("无数据的小时", int((~hours.has_data).sum()), pct((~hours.has_data).sum(), G)),
    ]
    for t in tiers:
        rows.append((f"有数据且最大年龄 {t} 的小时", int(tier_counts.get(t, 0)), pct(tier_counts.get(t, 0), G)))
    rows += [
        ("期权总行数", int(hours.n_rows.sum()), ""),
        ("其中前向填充行", int(hours.ffill_rows.sum()), pct(hours.ffill_rows.sum(), hours.n_rows.sum())),
        ("**未被标为 stale 的前向填充行**", int(hours.ffill_unflagged_rows.sum()), pct(hours.ffill_unflagged_rows.sum(), hours.n_rows.sum())),
        ("stale 行", int(hours.n_stale.sum()), pct(hours.n_stale.sum(), hours.n_rows.sum())),
        ("ffill 判定与“快照早于 bar 开始”不一致的（ts, expiry）", int(hours.n_ffill_rule_disagree.sum()), ""),
        (f"标签内 raw_timestamp 跨度 > {cfg['snapshot']['intra_label_spread_flag_s']}s 的小时", int(hours.spread_flag.sum()), ""),
    ]
    if "r_squared" in es:
        er = es[~es.is_ffill]
        rows += [("出现过的到期日数", int(es.expiry.nunique()), ""),
                 (f"r² < {cfg['expiries']['r2_threshold']} 的真实（ts, expiry）", int((er.r_squared < cfg['expiries']['r2_threshold']).sum()), pct((er.r_squared < cfg['expiries']['r2_threshold']).sum(), len(er))),
                 ("r² 最低值", f"{er.r_squared.min():.4f}（{er.loc[er.r_squared.idxmin(), 'timestamp']:%Y-%m-%d %H:%M} {er.loc[er.r_squared.idxmin(), 'expiry_label']}）" if len(er) else "", "")]
    w(md_table(pd.DataFrame(rows, columns=["指标", "值", "占比"])))

    w("## 最严重的问题\n")
    for b in severe(r, s, cfg):
        w(f"- {b}")
    w("")

    w("## 按月可用率\n")
    m = s["monthly"]
    cols = ["month", "grid_hours", "data_hours", "real_snapshot_hours", "all_real_hours", "mixed_hours", "new_expiry_hours",
            "ffill_only_hours", "missing_hours"] + [f"real_tier_{t}" for t in tiers] + ["real_pct", f"fresh_{tiers[0]}_pct",
            "ffill_unflagged_rows", "stale_rows"]
    w(md_table(m[cols], floatfmt="{:.1f}"))
    w(f"`real_tier_*` = 真实快照小时按最大年龄分档；`fresh_{tiers[0]}_pct` = 最大年龄 {tiers[0]} 的小时占网格比例。\n")

    # A
    w("## A. 文件层\n")
    inv = r["inventory"]
    w(f"- 期望日期目录：期权 {int((inv.dataset == 'options').sum())}，标的 {int((inv.dataset == 'underlyings').sum())}；"
      f"三件套不齐的日期：{int((~inv.complete).sum())}；额外文件：{int(inv.n_other_files.sum())}；不在期望范围内的目录：{int((~inv.expected).sum())}。")
    mf = r["manifest"]
    if len(mf):
        bad = mf[~(mf.manifest_date_ok & mf.manifest_start_ok & mf.manifest_end_ok)]
        w(f"- manifest 日期/起止时间不一致的文件：{len(bad)}。")
    if "xcheck_summary" in s:
        w("- report.json 与 parquet 重新计算值的核对（`n_mismatch` = 不一致的天数）：\n")
        x = s["xcheck_summary"]
        w(md_table(x.sort_values(["n_mismatch", "dataset"], ascending=[False, True])))
        w("说明：期权的 `raw_rows` 按“非前向填充的（ts, expiry）行数”重算；`missing_grid_points` 按完整 24 小时网格重算；"
          "`missing_price_rows` 按价格为空的行数重算。这三个字段若不一致，说明 report 的定义与真实情况不同。\n")

    # B
    w("## B. 期权快照层\n")
    w("真实快照（按小时）在 bar 内的分钟分布（快照时间 − 标签时刻，取整分钟）：\n")
    w(md_table(s["minute_dist"], max_rows=30))
    w("每个（ts, expiry）的新鲜度分档（全部 / 仅真实快照）：\n")
    w(md_table(s["es_tier"].merge(s["es_tier_real"], on="tier", how="left", suffixes=("_all", "_real"))))
    w(f"连续无真实快照的最长 {cfg['snapshot']['top_gaps']} 段（含全部为复制的小时）：\n")
    w(md_table(s["gaps_real"]))
    w(f"连续无任何数据的最长 {cfg['snapshot']['top_gaps']} 段：\n")
    w(md_table(s["gaps_data"]))

    # C
    if "life" in s:
        w("## C. 到期日层\n")
        life, gaps = s["life"], s["intermittent"]
        w(f"- 出现过的到期日：{len(life)}；存续期内出现比例 < 100% 的：{int((life.frac_present < 1).sum())}；"
          f"在其他到期日有数据时缺席过的：{int((life.n_absent_while_others_present > 0).sum())}；间歇缺失段数：{len(gaps)}。")
        w("- 最后一次出现距到期时间（小时）中位数：%.1f。\n" % life.hours_before_expiry_last_seen.median())
        w("存续期内出现比例最低的到期日：\n")
        w(md_table(life.sort_values("frac_present_when_data")[["expiry_label", "first_seen", "last_seen", "n_life_hours", "n_present",
                   "n_present_real", "n_absent_while_others_present", "n_absent_hour_missing", "frac_present", "frac_present_when_data"]], max_rows=20))
        w("间歇性缺失（到期日缺席、但同一小时其他到期日有数据），按长度排序：\n")
        w(md_table(gaps.sort_values("length_h", ascending=False), max_rows=30))
        dr = s["deribit"]
        if len(dr):
            lst = cfg["expiries"]["deribit_listing"]
            w("### 按 Deribit 规则推断的到期日 —— **可能缺失（启发式）**\n")
            w(f"规则：08:00 UTC 到期；任一真实快照小时，应存在最近 {lst['daily']} 个每日、{lst['weekly']} 个周五、"
              f"{lst['monthly']} 个月末周五、{lst['quarterly']} 个季度（3/6/9/12 月末周五）合约（距到期 > {cfg['expiries']['min_time_to_expiry_h']}h）。"
              "这只是启发式推断，Deribit 实际挂牌数量会变化，以下条目**不视为错误**，仅供排查。\n")
            w(md_table(dr.status.value_counts().rename_axis("status").rename("n").reset_index()))
            w(md_table(dr[dr.status != "ok"][["expiry_label", "kinds", "status", "first_expected", "last_expected",
                                              "n_expected_hours", "n_present", "frac_present"]], max_rows=60))
        w("### r² 按 tenor 分桶\n")
        w(md_table(s["r2_tenor"]))
        w(f"r² < {cfg['expiries']['r2_threshold']} 的明细（最差 25 条，全量见 `expiry_low_r2.parquet`）：\n")
        w(md_table(s["low_r2"], max_rows=25))
        sw = s["switches"]
        br = s["bracket"]
        w(f"### {cfg['expiries']['cm_target_days']} 天恒定期限的括号到期日\n")
        w(f"- 有数据的小时 {len(br)}，其中未被括住（无法插值、需外推）的 {int((~br.bracketed).sum())}。")
        w(f"- 括号组合切换 {len(sw)} 次：" + "，".join(f"{k} {v}" for k, v in sw.kind.value_counts().items()) + "。"
          "`roll` = 原上端到期日跌破目标期限后成为下端（正常滚动）；其余类型来自到期日缺失或新挂牌。\n")
        w(md_table(sw[sw.kind != "roll"], max_rows=40))

    # C2
    if sw_ts is not None:
        w("## C2. tau 时钟\n")
        w(f"用日历期限 > {cfg['expiries']['tau_clock_min_days']} 天的真实曲线诊断。offset = tau×{yd:g}×24 − (expiry − 快照时间)（小时）；"
          "rate = 相邻两小时间 tau 的减少量 / 快照时间的间隔（日历时钟下应为 1.0）。\n")
        w(f"- 持续偏离日历时钟的起点：**{sw_ts:%Y-%m-%d %H:%M} UTC**。此前偶发偏离 "
          f"{int((~tc['hourly'].calendar_like & (tc['hourly'].timestamp < sw_ts)).sum())} 个标签；此后偶然与日历时钟重合 "
          f"{int((tc['hourly'].calendar_like & (tc['hourly'].timestamp >= sw_ts)).sum())} 个标签（vol-time 与日历时间交叉，不是真的切回）。")
        w("- 同一标签上，不同长期限到期日的 offset 几乎相同（中位跨度 %.4f 小时），说明是参考时钟的问题，而非单个到期日的问题。"
          % tc["hourly"].loc[tc["hourly"].timestamp >= sw_ts, "offset_spread_h"].median())
        w("- 影响：从切换时刻起，`implied_vol` 是相对 vol-time 的波动率（σ²·τ_vol 才是 total variance）。"
          "与按日历时间年化的 RV 比较时，需要换算 σ_cal² = σ²·τ_vol/τ_cal，否则 IV 的水平会随星期几和时段系统性摆动。"
          "本审计不做换算，只报告。\n")
        if "tau_split" in s:
            w("tau 一致性检查按切换前后拆分：\n"); w(md_table(s["tau_split"]))
        w("按月：\n"); w(md_table(s["tau_monthly"], floatfmt="{:.3f}"))
        if tc.get("profile") is not None:
            p = tc["profile"]
            w("切换后 rate 的中位数（行 = UTC 小时，列 = 星期；单位：vol-time 小时 / 日历小时）：\n")
            w(md_table(p, floatfmt="{:.3f}"))
            daily = p.drop(columns="hour").sum()
            w("每天合计（vol-time 小时）：" + "，".join(f"{k} {v:.2f}" for k, v in daily.items()) + f"；一周合计 {daily.sum():.1f}（日历为 168）。\n")

    # D
    if "inv" in s:
        w("## D. 合约层不变量\n")
        w(f"B76 判定：|误差| > {cfg['invariants']['b76_abs_tol']:g} 美元 **且** 相对误差 > {cfg['invariants']['b76_rel_tol']:g} 才算违反；"
          "负价格、零价格单独计数，不计入定价误差。`*_real` 只统计非前向填充的行。\n")
        w(md_table(s["inv"]))
        misc = pd.DataFrame(r["inv_misc"])
        if len(misc):
            w(f"- 若按 365 天/年解读 tau，最大误差 {misc.tau365_max_err_h.max():.2f} 小时；B76 最大绝对误差 {misc.b76_max_abs_err.max():.3g} 美元；"
              f"价格 > abs_tol 时最大相对误差 {misc.b76_max_rel_err_px_gt_abs_tol.max():.3g}。\n")
        if "flag_combos" in s:
            w("质量 flag 组合（行数）：\n")
            w(md_table(s["flag_combos"], max_rows=30))
        if "floor_n_real" in s:
            w(f"### IV = {cfg['invariants']['iv_floor']} 下限点\n")
            w(f"- 下限行（非 atm）：真实 {s['floor_n_real']}，含前向填充 {s['floor_n_all']}。\n")
            w("按到期日：\n"); w(md_table(s["floor_by_expiry"], max_rows=30))
            w("按 tenor 分桶：\n"); w(md_table(s["floor_by_tenor"]))
            w("按 log-moneyness 分桶（log(K/F)）：\n"); w(md_table(s["floor_by_logm"]))
        if "arb_totals" in s:
            a = s["arb_totals"]
            w("### 静态无套利\n")
            w(f"- 检查的真实快照（标签）：{a['n_snapshots']}。")
            w(f"- 蝶式（相邻斜率 (C[i+1]−C[i])/(K[i+1]−K[i]) 应单调递增）：检查 {a['butterfly_checked']} 处，违反 {a['butterfly_viol']}"
              f"（{pct(a['butterfly_viol'], a['butterfly_checked'])}），有违反的快照 {a['snap_with_bfly']}（{pct(a['snap_with_bfly'], a['n_snapshots'])}）。")
            w(f"- call 斜率越界（>0 或 <−1）：{a['call_slope_bound_viol']}。")
            w(f"- 日历（同一 log-moneyness 下 total variance 随 tenor 递增，相邻到期日、网格内插）：检查 {a['calendar_checked']} 点，违反 {a['calendar_viol']}"
              f"（{pct(a['calendar_viol'], a['calendar_checked'])}），有违反的快照 {a['snap_with_cal']}（{pct(a['snap_with_cal'], a['n_snapshots'])}）。\n")
            w("日历套利最常见的相邻到期日对：\n"); w(md_table(s["cal_pairs"], max_rows=15))
        ex = r["inv_examples"]
        if len(ex):
            w("违反示例（每项最多 3 条，全量见 `invariant_examples.parquet`）：\n")
            w(md_table(ex.groupby("check").head(3)[["check", "timestamp", "instrument_name", "implied_vol", "tau_years", "mid", "is_stale"]], max_rows=60))

    # E
    u = r.get("und")
    if u is not None:
        w("## E. 标的层\n")
        w(f"网格 {meta['und_start']} 00:00 至 {meta['und_end']} 23:00，共 {len(u['series'])} 小时。收益率在完整序列上计算（含 23:00→00:00）。\n")
        c = u["checks"].assign(viol_rate=lambda x: x.n_viol / x.n_checked.where(x.n_checked > 0))
        w(md_table(c))
        w("说明：open 的回退不跨文件，所以每天 00:00 的 open 等于自己的 close，而不是前一天 23:00 的 close。"
          "这会造成跨日的 open ≠ 上一根 close，以及 high < open / low > open。用收益率时应基于 close→close。\n")
        w(f"|log return| > {cfg['underlyings']['jump_k']:g}× 滚动标准差（过去 {cfg['underlyings']['roll_window_h']}h，不含当根）：\n")
        w(md_table(u["jumps"].sort_values("z", key=abs, ascending=False), max_rows=30))
        w(f"连续零收益片段（≥ {cfg['underlyings']['zero_run_min_h']}h）：\n"); w(md_table(u["zero_runs"], max_rows=30))
        w("缺失小时片段：\n"); w(md_table(u["missing_runs"], max_rows=30))

    # F
    al = r.get("align")
    if al is not None and len(al["detail"]):
        w("## F. 时间对齐检验\n")
        w(f"每个真实快照标签上，取 tenor 最短且 > {cfg['alignment']['min_tenor_h']}h 的（非复制）到期日 forward F，"
          "与标的 close（标签 + lag）比较。d = log(F/close)。d 的水平主要反映期货基差；"
          "对齐是否正确更应看 **d 的逐小时变化的标准差**，正确的 lag 应最小。\n")
        w(md_table(al["stats"], floatfmt="{:.3g}"))
        w("按月（lag 0）：\n"); w(md_table(al["monthly"], floatfmt="{:.3g}"))

    w("## 输出文件\n")
    for f in sorted(meta["files"]):
        w(f"- `{f}`")
    return "\n".join(L) + "\n"


def severe(r: dict, s: dict, cfg: dict) -> list[str]:
    """Auto-generated list of the most important problems, in a fixed priority order."""
    hours = r["hours"]
    G = len(hours)
    out = []
    tc = s.get("tau_clock", {})
    if tc.get("switch") is not None:
        out.append(f"**tau 时钟在 {tc['switch']:%Y-%m-%d %H:%M} UTC 由日历时间切换为加权 vol-time**："
                   "此后 tau 与 (expiry − 快照时间) 的差在 ±1 天量级，IV 是相对 vol-time 的波动率，做 VRP 前必须先决定如何换算。")
    if "inv" in s:
        inv0 = s["inv"].set_index("check")
        if "row_older_than_curve_unflagged" in inv0.index and inv0.loc["row_older_than_curve_unflagged", "n_viol"] > 0:
            out.append(f"**曲线内残留的旧快照行**：{int(inv0.loc['row_older_than_curve_snapshot', 'n_viol'])} 行的 raw_timestamp 早于所在曲线的快照"
                       f"（其中未标 stale {int(inv0.loc['row_older_than_curve_unflagged', 'n_viol'])} 行），主要是以旧 forward 命名的 atm 行；"
                       "按 `surface_key_type=='atm'` 取 ATM IV 会拿到多行，必须再按 raw_timestamp 取最新。")
    t0 = s["tiers"][0]
    nfresh = int(((hours.freshness_tier == t0) & hours.has_real).sum())
    out.append(f"**可用性**：{G} 个网格小时中，真实快照 {int(hours.has_real.sum())}（{pct(hours.has_real.sum(), G)}），"
               f"最大年龄 {t0} 的仅 {nfresh}（{pct(nfresh, G)}）；无数据 {int((~hours.has_data).sum())}，全部为复制 {int((hours.hour_type == 'ffill').sum())}。")
    uf = int(hours.ffill_unflagged_rows.sum())
    if uf:
        out.append(f"**未标 stale 的前向填充行 {uf}**（{pct(uf, hours.n_rows.sum())} 的行）：价格非空但实为旧快照，若直接使用会把旧价格当成新价格。")
    if hours.partial_mixed.any():
        out.append(f"**部分更新小时 {int(hours.partial_mixed.sum())}**：同一标签内有的到期日是新快照、有的是复制，期限结构横截面不同步。")
    if "gaps_real" in s and len(s["gaps_real"]):
        g = s["gaps_real"].iloc[0]
        out.append(f"最长的无真实快照段 {int(g.length_h)} 小时（{g.start:%Y-%m-%d %H:%M} 至 {g.end:%Y-%m-%d %H:%M}）。")
    if "xcheck_summary" in s:
        x = s["xcheck_summary"]
        bad = x[x.n_mismatch > 0]
        if len(bad):
            out.append("report.json 与实际不符的字段：" + "，".join(f"`{r.dataset}.{r.field}`（{r.n_mismatch} 天）" for r in bad.itertuples()) + "。")
    if "inv" in s:
        inv = s["inv"].set_index("check")
        for chk, label in [("b76_price", "B76 定价误差"), ("negative_price", "负价格"), ("call_put_iv_equal", "call/put IV 不等"),
                           ("delta_b76", "delta 与 B76 不符"), ("tau_vs_expiry_minus_raw_ts", "tau 不一致"),
                           ("price_null_iff_stale", "价格为空 ⇎ stale"), ("iv_out_of_range", "IV 超出范围"),
                           ("bid_ask_mid_mark_equal", "bid/ask/mid/mark 不等")]:
            if chk in inv.index and inv.loc[chk, "n_viol"] > 0:
                out.append(f"{label}：{int(inv.loc[chk, 'n_viol'])} 行（{pct(inv.loc[chk, 'n_viol'], inv.loc[chk, 'n_checked'])}）。")
    if "arb_totals" in s:
        a = s["arb_totals"]
        if a["butterfly_viol"] or a["calendar_viol"]:
            out.append(f"静态套利：蝶式违反 {a['butterfly_viol']} 处（{a['snap_with_bfly']} 个快照），日历违反 {a['calendar_viol']} 点（{a['snap_with_cal']} 个快照）。")
    if "low_r2" in s and len(s["low_r2"]):
        out.append(f"r² < {cfg['expiries']['r2_threshold']} 的曲线 {len(s['low_r2'])} 条，最低 {s['low_r2'].r_squared.iloc[0]:.3f}；`vol_curve_check` 不能作为质量依据。")
    if "switches" in s:
        sw = s["switches"]
        nr = int((sw.kind != "roll").sum())
        if nr:
            out.append(f"30 天括号组合的非正常切换 {nr} 次（共 {len(sw)} 次），来自到期日缺失或新挂牌。")
    u = r.get("und")
    if u is not None:
        c = u["checks"].set_index("check")
        out.append(f"标的：缺失 {int(c.loc['grid_missing_hours', 'n_viol'])} 小时，重复 {int(c.loc['grid_duplicate_hours', 'n_viol'])}；"
                   f"日内 open≠上一根 close {int(c.loc['open_eq_prev_close_intraday', 'n_viol'])}，跨日 {int(c.loc['open_eq_prev_close_cross_day', 'n_viol'])}；"
                   f"跳跃 {len(u['jumps'])}，零收益片段 {len(u['zero_runs'])}。")
    return out
