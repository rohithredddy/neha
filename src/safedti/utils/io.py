"""Results writer: every run produces one JSON in results/ (CLAUDE.md rule 7)."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


def git_commit() -> str:
    """Return the current commit hash, suffixed with '-dirty' if the tree has changes."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_result(
    name: str,
    config: dict[str, Any],
    seed: int,
    metrics: dict[str, Any],
    wall_time_s: float,
    out_dir: str | Path = "results",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a run record to ``{out_dir}/{name}.json``.

    Args:
        name: Run name, e.g. ``"ib_id_safedti_0"``.
        config: Full resolved config.
        seed: Seed used.
        metrics: Metric name -> value. Must come from an actual computation.
        wall_time_s: Wall-clock duration of the run in seconds.
        out_dir: Destination directory.
        extra: Optional additional fields (e.g. data-card version).

    Returns:
        Path of the written file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "name": name,
        "git_commit": git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "seed": seed,
        "config": config,
        "metrics": metrics,
        "wall_time_s": wall_time_s,
        **(extra or {}),
    }
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=False), encoding="utf-8")
    return path
