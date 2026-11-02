# deribit_vrp

BTC 期权与标的小时数据的审计和清洗，为 crypto VRP 研究（比较不同 IV、RV、VRP 定义）准备一个与具体定义无关的研究面板。

## 目录结构

| 目录 | 作用 |
|---|---|
| `config/` | `audit.yaml`（审计阈值）、`clean.yaml`（清洗规则、标记阈值、vendor_check 分类、默认研究过滤） |
| `src/audit/` | 审计：只描述原始数据，不修改、不清洗 |
| `src/clean/` | 清洗：生成研究面板；`filters.py` 是研究过滤的唯一入口 |
| `src/build_info.py` | 为每个输出目录写 `_build_info.json` |
| `scripts/` | 入口脚本：`run_audit.py`、`build_panel.py` |
| `tests/` | pytest 单元测试 |
| `notebooks/` | `01_audit_and_cleaning_explained.ipynb`：审计与清洗的说明；`00.ipynb`、`notes.ipynb` 为个人笔记 |
| `data/raw/` | 原始数据，**只读**（不在 git 中） |
| `data/audit/` | 审计输出（不在 git 中，可重新生成） |
| `data/processed/` | 研究面板（不在 git 中，可重新生成） |

## 数据流程

```
data/raw/ ──► scripts/run_audit.py   ──► data/audit/       （审计：明细表、audit_summary.md、图）
          └─► scripts/build_panel.py ──► data/processed/   （清洗：研究面板、clean_report.md）
```

面板文件：`surface_curves.parquet`（每条曲线一行）、`surface_points.parquet`（每个挂牌行权价一行）、
`surface_calendar_pairs.parquet`（同一快照内所有到期日组合的日历检查）、`underlying_hourly.parquet`、
`hour_of_week_clock.parquet`（vol-time 的 168 小时权重，EDT / EST 两套）。

## 清洗原则：只删除非真实观测，其余只加标记

- **删除**（不是真实观测）：
  - 1a 前向填充复制出来的曲线；
  - 2a 曲线内比最新快照早超过阈值的残留行；
  - 2b atm 行只保留最新的一组 C/P。
- **标记**（真实观测，保留在面板里）：
  - 1b `flag_late_snapshot`：快照年龄超过阈值；
  - 3a `flag_short_tenor`：日历 tenor 小于阈值；
  - 3b `flag_low_r2`：r² 小于阈值；
  - 3c `vendor_check` / `vendor_check_class`：供应商的 `vol_curve_check` 及其分类（fit_quality / value_screen / ambiguous）。
  - 另有 `iv_floor`（IV = 0.03 的下限点）、蝶式和日历套利的违反数。

`IV_RV_DIFF` 等 value_screen 是按 IV 的数值筛选的，删掉会截掉 VRP 的尾部，所以默认保留。

## 研究代码如何筛选曲线

**必须**通过 `src/clean/filters.py` 的 `apply_research_filter()` 筛选，不允许自己另写过滤条件：

```python
from clean.filters import apply_research_filter
curves = apply_research_filter(curves)                                # 默认：config 中的 research_default_filter
curves = apply_research_filter(curves, {"exclude_ambiguous": True})   # 敏感性检验：单独打开或关闭某一项
```

面板不保存"是否通过"的列，过滤结果每次现场计算，不会随配置变化而过时。

## 时间约定

- 所有时间戳都是 **UTC**；`timestamp` 是小时 bar 的**开始**时刻，`obs_time` 是该曲线实际的快照时间。
- tau 按 **365.25 天/年**：`tau_cal = (expiry − obs_time) / 365.25 天`；RV 年化统一用 365.25 × 24 小时/年。
- **2025-06-28 00:00 UTC 起，原始 `tau_years` 改为加权的 vol-time 时钟**（`clock_regime = vol_time`，此时 `tau_vol = tau_years`）。
- 面板只保存总方差 `w = IV² × tau_years` 和两种 τ（`tau_cal`、`tau_vol`），**不保存原始 IV**；需要 IV 时由研究代码明确选择时钟换算。

## 重新生成输出与运行测试

```bash
python scripts/run_audit.py                                                   # data/audit/
python scripts/run_audit.py --dates 2025-04-24 --out data/audit/sample_2025-04-24
python scripts/build_panel.py                                                 # data/processed/
python scripts/build_panel.py --dates 2025-04-24,2025-06-27,2025-06-28 --out data/processed/sample_boundary
python -m pytest -q tests
```

两个脚本都支持 `--start`、`--end`、`--dates`、`--out`、`--config`。

## `_build_info.json`

每次生成输出时，都会在输出目录里写一个 `_build_info.json`，记录 git 提交哈希（以及工作区是否 dirty）、配置文件的路径和 sha1、
生成时间（UTC）、运行参数、输出文件清单及每个文件的 sha1。用它判断一份输出是由哪个版本的代码、哪份配置生成的；
工作区 dirty 时脚本会打印警告，但不会中止。

## Notebook

`notebooks/01_audit_and_cleaning_explained.ipynb` 以 notebook 本身为准，直接在 Jupyter 里编辑。它只读取数据和配置、不写任何数据文件；
修改后用下面的命令重新执行，确认所有 assert 通过：

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_audit_and_cleaning_explained.ipynb
```
