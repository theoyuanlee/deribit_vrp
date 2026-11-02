"""clean.filters: default research filter, overrides, unknown vendor values, curve/point key correspondence."""
import copy
from pathlib import Path

import pandas as pd
import pytest

from audit import io
from audit import options as aopt
from clean import filters, panel

ROOT = Path(__file__).resolve().parents[1]
CFG = filters.load_config()
KEYS = ["timestamp", "expiry"]


def toy_curves() -> pd.DataFrame:
    """One curve per situation: clean, each flag alone, and each vendor class."""
    rows = [
        ("clean", False, False, False, "GOOD"),
        ("late", True, False, False, "GOOD"),
        ("short", False, True, False, "GOOD"),
        ("low_r2", False, False, True, "GOOD"),
        ("fit_quality", False, False, False, "EARLY_EXIT_INSUFFICIENT_STRIKES"),
        ("value_screen", False, False, False, "IV_RV_DIFF"),
        ("ambiguous", False, False, False, "BAD_SHAPE"),
        ("late_and_value_screen", True, False, False, "VOL_SPIKE"),
    ]
    c = pd.DataFrame(rows, columns=["name", "flag_late_snapshot", "flag_short_tenor", "flag_low_r2", "vendor_check"])
    c["vendor_check_class"] = filters.classify_vendor_check(c.vendor_check, CFG)
    return c


def test_default_filter_behaviour():
    kept = filters.apply_research_filter(toy_curves(), verbose=False)
    assert set(kept.name) == {"clean", "value_screen", "ambiguous"}


def test_default_filter_matches_config():
    assert set(CFG["research_default_filter"]) == set(filters.FILTER_ITEMS)
    opts = filters.resolve_options(CFG)
    assert opts["exclude_value_screen"] is False and opts["exclude_ambiguous"] is False


@pytest.mark.parametrize("overrides,expected", [
    ({"exclude_value_screen": True}, {"clean", "ambiguous"}),
    ({"exclude_ambiguous": True}, {"clean", "value_screen"}),
    ({"exclude_late_snapshot": False}, {"clean", "late", "value_screen", "ambiguous", "late_and_value_screen"}),
    ({"exclude_fit_quality": False, "exclude_low_r2": False}, {"clean", "low_r2", "fit_quality", "value_screen", "ambiguous"}),
])
def test_overrides(overrides, expected):
    kept = filters.apply_research_filter(toy_curves(), overrides, verbose=False)
    assert set(kept.name) == expected


def test_overrides_do_not_change_config():
    before = copy.deepcopy(CFG["research_default_filter"])
    filters.apply_research_filter(toy_curves(), {"exclude_value_screen": True}, cfg=CFG, verbose=False)
    assert CFG["research_default_filter"] == before


def test_bad_overrides_raise():
    with pytest.raises(KeyError):
        filters.apply_research_filter(toy_curves(), {"exclude_everything": True}, verbose=False)
    with pytest.raises(TypeError):
        filters.apply_research_filter(toy_curves(), {"exclude_value_screen": "yes"}, verbose=False)


def test_prints_removed_counts(capsys):
    filters.apply_research_filter(toy_curves())
    out = capsys.readouterr().out
    assert "exclude_late_snapshot" in out and "-> 3 curves kept" in out


def test_mask_counts_are_incremental():
    keep, table = filters.research_mask(toy_curves(), cfg=CFG)
    t = table.set_index("item")
    assert t.loc["exclude_late_snapshot", "curves_flagged"] == 2 and t.loc["exclude_late_snapshot", "curves_removed"] == 2
    assert t.loc["exclude_value_screen", "curves_flagged"] == 2 and t.loc["exclude_value_screen", "curves_removed"] == 0
    assert keep.sum() == 3 and t.curves_removed.sum() == len(keep) - keep.sum()


def test_unknown_vendor_value_raises():
    with pytest.raises(ValueError, match="NEW_CHECK"):
        filters.classify_vendor_check(pd.Series(["GOOD", "NEW_CHECK"]), CFG)


def test_duplicate_vendor_value_in_config_raises():
    cfg = copy.deepcopy(CFG)
    cfg["vendor_check_classes"]["ambiguous"] = ["BAD_SHAPE", "IV_RV_DIFF"]
    with pytest.raises(ValueError, match="IV_RV_DIFF"):
        filters.vendor_class_map(cfg)


def raw_day(date: str):
    df, _, _ = io.read_day(ROOT / CFG["paths"]["raw_root"], "options", CFG["asset"], date)
    if df is None:
        pytest.skip("raw data not available")
    df = aopt.prepare(df)
    es, _ = aopt.expiry_snapshots(df, aopt.new_state(), panel.audit_cfg(CFG))
    return df, es


def test_unknown_vendor_value_aborts_panel_build():
    df, es = raw_day("2025-06-28")
    df["vol_curve_check"] = df["vol_curve_check"].astype(str)
    first = df.loc[df.index[0], KEYS]
    in_curve = (df.timestamp == first.timestamp) & (df.expiry == first.expiry)
    df.loc[in_curve, "vol_curve_check"] = "NEW_CHECK"           # whole curve, so the constancy check passes
    with pytest.raises(ValueError, match="NEW_CHECK"):
        panel.clean_day(df, es, "2025-06-28", CFG)


def test_vendor_value_not_constant_within_curve_raises():
    df, es = raw_day("2025-06-28")
    df["vol_curve_check"] = df["vol_curve_check"].astype(str)
    df.loc[df.index[0], "vol_curve_check"] = "IV_RV_DIFF"        # one row of one curve differs
    with pytest.raises(ValueError, match="not constant"):
        panel.clean_day(df, es, "2025-06-28", CFG)


@pytest.mark.parametrize("date", ["2025-04-24", "2025-06-28"])
def test_curve_and_point_keys_correspond(date):
    df, es = raw_day(date)
    curves, points, pairs, _ = panel.clean_day(df, es, date, CFG)
    curve_keys = set(map(tuple, curves[KEYS].to_numpy()))
    point_keys = set(map(tuple, points[KEYS].drop_duplicates().to_numpy()))
    assert point_keys <= curve_keys                              # every point belongs to a curve
    no_points = curves[[k not in point_keys for k in map(tuple, curves[KEYS].to_numpy())]]
    assert (no_points.n_strikes == 0).all()                      # curves without points have no listed strikes
    assert not curves.duplicated(KEYS).any() and not curves.duplicated(["obs_time", "expiry"]).any()
    assert (points.groupby(KEYS).size().reindex(curves.set_index(KEYS).index).fillna(0).to_numpy()
            == curves.n_strikes.to_numpy()).all()
