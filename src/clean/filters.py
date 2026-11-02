"""The single place where research code decides which panel curves to use.

The panel stores only raw flag columns. Which flags exclude a curve is configuration
(`research_default_filter` in config/clean.yaml) and is evaluated here, at use time, so a
stored "pass" column can never go stale. Research code must call apply_research_filter()
instead of writing its own conditions.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "clean.yaml"
VENDOR_CLASSES = ("GOOD", "fit_quality", "value_screen", "ambiguous")

# filter item -> function returning True for curves that the item would exclude
FILTER_ITEMS = {
    "exclude_late_snapshot": lambda c: c["flag_late_snapshot"],
    "exclude_short_tenor": lambda c: c["flag_short_tenor"],
    "exclude_low_r2": lambda c: c["flag_low_r2"],
    "exclude_fit_quality": lambda c: c["vendor_check_class"] == "fit_quality",
    "exclude_value_screen": lambda c: c["vendor_check_class"] == "value_screen",
    "exclude_ambiguous": lambda c: c["vendor_check_class"] == "ambiguous",
}


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def vendor_class_map(cfg: dict) -> dict[str, str]:
    """vol_curve_check value -> class, validated against VENDOR_CLASSES and for duplicates."""
    classes = cfg["vendor_check_classes"]
    unknown_classes = set(classes) - set(VENDOR_CLASSES)
    if unknown_classes:
        raise ValueError(f"unknown vendor_check classes in config: {sorted(unknown_classes)}")
    mapping = {}
    for cls, values in classes.items():
        for v in values:
            if v in mapping:
                raise ValueError(f"vol_curve_check value {v!r} listed in both {mapping[v]!r} and {cls!r}")
            mapping[v] = cls
    return mapping


def classify_vendor_check(values: pd.Series, cfg: dict) -> pd.Series:
    """Map raw vol_curve_check values to classes; any value missing from the config is an error."""
    mapping = vendor_class_map(cfg)
    unknown = sorted(set(values.dropna().astype(str)) - set(mapping))
    if unknown or values.isna().any():
        raise ValueError(f"vol_curve_check values not in config vendor_check_classes: {unknown}"
                         + (" (and missing values)" if values.isna().any() else ""))
    return values.astype(str).map(mapping)


def resolve_options(cfg: dict, overrides: dict | None = None) -> dict[str, bool]:
    opts = dict(cfg["research_default_filter"])
    for source, d in [("config research_default_filter", opts), ("overrides", overrides or {})]:
        bad = set(d) - set(FILTER_ITEMS)
        if bad:
            raise KeyError(f"unknown filter item(s) in {source}: {sorted(bad)}; known: {sorted(FILTER_ITEMS)}")
    missing = set(FILTER_ITEMS) - set(opts)
    if missing:
        raise KeyError(f"config research_default_filter is missing: {sorted(missing)}")
    opts.update(overrides or {})
    for k, v in opts.items():
        if not isinstance(v, bool):
            raise TypeError(f"filter item {k!r} must be true/false, got {v!r}")
    return opts


def research_mask(curves: pd.DataFrame, overrides: dict | None = None, cfg: dict | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """Boolean keep-mask plus a per-item table: curves each item flags on its own, and curves it removes
    incrementally (in FILTER_ITEMS order, after the earlier enabled items)."""
    cfg = cfg or load_config()
    opts = resolve_options(cfg, overrides)
    keep = pd.Series(True, index=curves.index)
    rows = []
    for item, hit in FILTER_ITEMS.items():
        flagged = hit(curves).astype(bool)
        removed = int((flagged & keep).sum()) if opts[item] else 0
        if opts[item]:
            keep &= ~flagged
        rows.append({"item": item, "enabled": opts[item], "curves_flagged": int(flagged.sum()), "curves_removed": removed})
    return keep, pd.DataFrame(rows)


def apply_research_filter(curves: pd.DataFrame, overrides: dict | None = None, cfg: dict | None = None,
                          verbose: bool = True) -> pd.DataFrame:
    """Return the curves passing the research filter (config default, optionally overridden item by item),
    e.g. apply_research_filter(curves, {"exclude_value_screen": True}). Prints what each item removed."""
    keep, table = research_mask(curves, overrides, cfg)
    if verbose:
        print(f"apply_research_filter: {len(curves):,} curves in")
        for r in table.itertuples():
            state = "on " if r.enabled else "off"
            print(f"  [{state}] {r.item:<22} flags {r.curves_flagged:>7,}  removes {r.curves_removed:>7,}")
        print(f"  -> {int(keep.sum()):,} curves kept")
    return curves[keep].copy()
