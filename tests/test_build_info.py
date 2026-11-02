"""build_info: file list with sha1, stale-file detection, git state."""
import hashlib
import json
import os
import time
from pathlib import Path

import build_info

ROOT = Path(__file__).resolve().parents[1]


def test_write_lists_files_with_sha1_and_marks_stale(tmp_path):
    old = tmp_path / "old.parquet"
    old.write_bytes(b"old")
    past = time.time() - 3600
    os.utime(old, (past, past))
    started = build_info.now_utc()
    (tmp_path / "new.md").write_text("new")
    (tmp_path / "sub").mkdir()                                   # subdirectories are separate outputs
    info = build_info.write(tmp_path, ROOT, "scripts/x.py", ROOT / "config" / "clean.yaml", {"dates": "2025-04-24"}, started)
    on_disk = json.loads((tmp_path / build_info.FILENAME).read_text())
    assert on_disk == info
    assert set(info["files"]) == {"old.parquet", "new.md"}
    assert info["files"]["new.md"] == hashlib.sha1(b"new").hexdigest()
    assert info["stale_files_not_written_this_run"] == ["old.parquet"]
    assert info["config_path"] == "config/clean.yaml" and len(info["config_sha1"]) == 40
    assert info["args"] == {"dates": "2025-04-24"}


def test_git_state_reports_commit():
    state = build_info.git_state(ROOT)
    assert state["commit"] is None or len(state["commit"]) == 40
    assert state["dirty"] in (True, False, None)
