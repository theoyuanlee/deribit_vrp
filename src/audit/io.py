"""File-layer helpers: config, directory inventory, JSON reports, parquet reads.

Everything here only reads from the raw tree.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

DATASET_DIRS = {"options": "options_daily", "underlyings": "underlyings_daily"}
KINDS = {"parquet": "*_60min.parquet", "manifest": "*_60min.manifest.json", "report": "*_60min.report.json"}


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def date_range(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]


def day_dir(root: Path, dataset: str, asset: str, date: str) -> Path:
    return Path(root) / DATASET_DIRS[dataset] / f"asset={asset}" / f"date={date}"


def present_dates(root: Path, dataset: str, asset: str) -> list[str]:
    base = Path(root) / DATASET_DIRS[dataset] / f"asset={asset}"
    return sorted(p.name.split("=", 1)[1] for p in base.glob("date=*") if p.is_dir())


def day_files(root: Path, dataset: str, asset: str, date: str) -> dict[str, list[Path]]:
    d = day_dir(root, dataset, asset, date)
    return {k: sorted(d.glob(pat)) if d.is_dir() else [] for k, pat in KINDS.items()}


def inventory(root: Path, dataset: str, asset: str, dates: list[str]) -> pd.DataFrame:
    """One row per expected date (plus unexpected extra dirs): which of the three files exist."""
    present = set(present_dates(root, dataset, asset))
    rows = []
    for date in sorted(set(dates) | (present - set(dates) if len(dates) > 1 else set())):
        files = day_files(root, dataset, asset, date)
        d = day_dir(root, dataset, asset, date)
        n_other = len([p for p in d.iterdir() if p not in sum(files.values(), [])]) if d.is_dir() else 0
        row = {"dataset": dataset, "date": date, "expected": date in dates, "has_dir": d.is_dir(), "n_other_files": n_other}
        for k, paths in files.items():
            row[f"n_{k}"] = len(paths)
        row["complete"] = all(row[f"n_{k}"] == 1 for k in KINDS)
        rows.append(row)
    return pd.DataFrame(rows)


def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def flatten_report(rep: dict, dataset: str, date: str) -> dict:
    """Scalars kept as-is; nested dicts flattened with a prefix; lists reduced to length + joined text."""
    out = {"dataset": dataset, "date": date}
    for k, v in rep.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                out[f"{k}.{k2}"] = v2
        elif isinstance(v, list):
            out[f"n_{k}"] = len(v)
            out[k] = ",".join(map(str, v))
        else:
            out[k] = v
    return out


def check_manifest(man: dict, dataset: str, date: str) -> dict:
    start = pd.Timestamp(date, tz="UTC")
    return {
        "dataset": dataset,
        "date": date,
        "manifest_date_ok": man.get("date") == date,
        "manifest_start_ok": pd.Timestamp(man.get("start")) == start if man.get("start") else False,
        "manifest_end_ok": pd.Timestamp(man.get("end_exclusive")) == start + pd.Timedelta("1D")
        if man.get("end_exclusive") else False,
        "manifest_freq": man.get("freq"),
        "manifest_data_type": man.get("data_type"),
    }


def crosscheck(rep: dict, actual: dict, dataset: str, date: str) -> list[dict]:
    """Compare report fields with values recomputed from the parquet. Returns long-format rows."""
    rows = []
    for field, act in actual.items():
        if field not in rep:
            rows.append({"dataset": dataset, "date": date, "field": field, "report": None, "actual": str(act), "match": False})
            continue
        r = rep[field]
        if isinstance(act, pd.Timestamp):
            ok = pd.Timestamp(r) == act if r is not None else False
        elif isinstance(act, float) or isinstance(r, float):
            ok = r is not None and abs(float(r) - float(act)) <= 1e-9 * max(1.0, abs(float(act)))
        else:
            ok = r == act
        rows.append({"dataset": dataset, "date": date, "field": field, "report": str(r), "actual": str(act), "match": bool(ok)})
    return rows


def read_day(root: Path, dataset: str, asset: str, date: str):
    """Return (df, report, manifest); each is None when the file is absent or ambiguous."""
    files = day_files(root, dataset, asset, date)
    df = pd.read_parquet(files["parquet"][0]) if len(files["parquet"]) == 1 else None
    rep = read_json(files["report"][0]) if len(files["report"]) == 1 else None
    man = read_json(files["manifest"][0]) if len(files["manifest"]) == 1 else None
    return df, rep, man
