"""Stage 2 — shared nba_api access layer: rate-limited, retry-with-backoff,
resume-from-cache. Every endpoint pull in Stage 2 goes through here.

Design (per BUILD_PLAN: "treat this as an overnight job, not a quick run"):
  - POLITE_DELAY between every live request (nba_api throttles aggressively).
  - Exponential backoff with jitter on timeout / 429 / 5xx.
  - Each distinct pull is cached by a stable key under data/raw/; a cache hit
    skips the network entirely, so re-running resumes instead of re-pulling.

nba_api is imported lazily so the module loads (and tests run) even when the
package or network isn't available.
"""
from __future__ import annotations

import random
import time
from typing import Any, Callable

from nba_injury.cache import cached_json

# nba_api is sensitive to request rate. These defaults are deliberately gentle.
POLITE_DELAY = 0.7           # seconds between live calls
MAX_RETRIES = 5
BACKOFF_BASE = 1.5           # seconds; grows ~BASE * 2**attempt
BACKOFF_CAP = 60.0
REQUEST_TIMEOUT = 60         # nba_api endpoints can be slow

# Module-level pacing clock so spacing holds across many calls in one run.
_last_call_ts = 0.0


def _pace() -> None:
    global _last_call_ts
    elapsed = time.monotonic() - _last_call_ts
    if elapsed < POLITE_DELAY:
        time.sleep(POLITE_DELAY - elapsed)
    _last_call_ts = time.monotonic()


def with_retry(fn: Callable[[], Any], *, what: str = "request") -> Any:
    """Call fn() with pacing + exponential backoff. Raises on final failure."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        _pace()
        try:
            return fn()
        except Exception as exc:  # nba_api raises a grab-bag of errors/timeouts
            last_exc = exc
            wait = min(BACKOFF_CAP, BACKOFF_BASE * (2 ** attempt))
            wait += random.uniform(0, wait * 0.3)  # jitter to avoid lockstep
            print(f"[client] {what}: attempt {attempt + 1}/{MAX_RETRIES} failed "
                  f"({type(exc).__name__}: {exc}); backing off {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"[client] {what}: exhausted {MAX_RETRIES} retries") from last_exc


def cached_endpoint(cache_name: str, build_fn: Callable[[], Any]) -> Any:
    """Fetch-or-load a single endpoint result as normalized dict, cached.

    build_fn must construct the nba_api endpoint object and return a
    JSON-serializable normalized dict (i.e. call .get_normalized_dict()).
    On a cache hit, build_fn is never called -> no network, instant resume.
    """
    return cached_json(cache_name, lambda: with_retry(build_fn, what=cache_name))


# 10-season window decided in the vision doc (~2015-16 .. 2024-25).
SEASONS = [
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
]
