"""_build_info.json: record which code, config and arguments produced an output directory."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

FILENAME = "_build_info.json"


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state(root: Path) -> dict:
    """Commit hash and whether tracked/untracked (non-ignored) files differ from it."""
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout.strip()
    try:
        commit = git("rev-parse", "HEAD")
        changes = [line for line in git("status", "--porcelain").splitlines() if line]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None, "dirty_files": []}
    return {"commit": commit, "dirty": bool(changes), "dirty_files": [c[3:] for c in changes]}


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def write(out_dir: Path, root: Path, script: str, config_path: Path, args: dict, started: dt.datetime) -> dict:
    """Write out_dir/_build_info.json. Lists the files directly in out_dir (subdirectories are separate
    outputs with their own build info); files not rewritten during this run are listed as stale.
    A dirty working tree only triggers a warning."""
    out_dir, config_path = Path(out_dir), Path(config_path)
    state = git_state(root)
    if state["dirty"]:
        print(f"WARNING: working tree is dirty; {out_dir / FILENAME} records commit {state['commit']} + "
              f"uncommitted changes: {state['dirty_files']}", file=sys.stderr)
    files, stale = {}, []
    for p in sorted(out_dir.iterdir()):
        if not p.is_file() or p.name == FILENAME or p.name.startswith("."):
            continue
        files[p.name] = sha1(p)
        if dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc) < started:
            stale.append(p.name)
    info = {
        "script": script,
        "git_commit": state["commit"],
        "git_dirty": state["dirty"],
        "git_dirty_files": state["dirty_files"],
        "config_path": str(config_path.relative_to(root)) if config_path.is_relative_to(root) else str(config_path),
        "config_sha1": sha1(config_path),
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": now_utc().isoformat(timespec="seconds"),
        "args": args,
        "files": files,
        "stale_files_not_written_this_run": stale,
    }
    (out_dir / FILENAME).write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    return info
