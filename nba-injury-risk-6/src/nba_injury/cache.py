"""Local file cache — the day-one habit from BUILD_PLAN Stage 0.

Rule: every external pull writes to data/raw/ and is NEVER re-fetched if present.
This single habit avoids the multi-hour re-pull pain in Stage 2.

Usage:
    from nba_injury.cache import cached_json, RAW, PROCESSED

    data = cached_json("player_career_2544.json", fetch_fn=lambda: pull_from_api())
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

# Resolve project paths relative to this file so scripts work from any CWD.
_PKG_ROOT = Path(__file__).resolve().parents[2]
RAW = _PKG_ROOT / "data" / "raw"
PROCESSED = _PKG_ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)


def cached_json(name: str, fetch_fn: Callable[[], Any], *, refresh: bool = False) -> Any:
    """Return JSON from data/raw/<name>, fetching+caching only if absent.

    fetch_fn is called ONLY on a cache miss (or refresh=True). It must return a
    JSON-serializable object. The result is written to data/raw/<name>.
    """
    path = RAW / name
    if path.exists() and not refresh:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    result = fetch_fn()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)  # atomic write — never leave a half-written cache file
    return result


def cached_text(name: str, fetch_fn: Callable[[], str], *, refresh: bool = False) -> str:
    """Same contract as cached_json but for raw text/HTML payloads."""
    path = RAW / name
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")
    result = fetch_fn()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(result, encoding="utf-8")
    tmp.replace(path)
    return result


def raw_path(name: str) -> Path:
    return RAW / name


def processed_path(name: str) -> Path:
    return PROCESSED / name
