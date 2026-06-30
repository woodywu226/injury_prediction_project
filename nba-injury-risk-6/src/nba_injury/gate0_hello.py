"""Gate 0 check (BUILD_PLAN Stage 0): pull ONE player's career stats and cache it.

This is the trivial end-to-end smoke test. It confirms three things:
  1. The package imports and the cache layer works.
  2. nba_api (the Stage 2 backbone dependency) is reachable from your network.
  3. The write-once cache pattern functions.

Run:  python -m nba_injury.gate0_hello
"""
from __future__ import annotations

import sys

from nba_injury.cache import cached_json, RAW

# Nikola Jokic — a stable, well-known player_id, good canary.
PLAYER_ID = 203999


def _fetch_career() -> dict:
    """Pull career stats from nba_api. Imported lazily so the module loads even
    when nba_api isn't installed yet (e.g. inspecting the repo)."""
    try:
        from nba_api.stats.endpoints import playercareerstats
    except ImportError:
        raise SystemExit(
            "nba_api not installed. Run: pip install nba_api\n"
            "(Left out of requirements.txt Stage 0 pin intentionally? No — add it "
            "when you run this. It's the Stage 2 backbone.)"
        )
    career = playercareerstats.PlayerCareerStats(player_id=PLAYER_ID, timeout=30)
    return career.get_normalized_dict()


def main() -> int:
    print(f"[gate0] fetching career stats for player_id={PLAYER_ID} ...")
    try:
        data = cached_json(f"career_{PLAYER_ID}.json", _fetch_career)
    except Exception as exc:  # noqa: BLE001 — we want the human-readable reason
        print(f"[gate0] FAILED to reach nba_api: {exc}", file=sys.stderr)
        print(
            "[gate0] If this is a network/proxy block, run this on a machine with "
            "open egress to stats.nba.com. Gate 0 is NOT passed until this succeeds.",
            file=sys.stderr,
        )
        return 1

    seasons = data.get("SeasonTotalsRegularSeason", [])
    print(f"[gate0] OK — cached {len(seasons)} regular-season rows to "
          f"{RAW / f'career_{PLAYER_ID}.json'}")
    if seasons:
        first, last = seasons[0]["SEASON_ID"], seasons[-1]["SEASON_ID"]
        print(f"[gate0] season span: {first} .. {last}")
    print("[gate0] GATE 0 PASSED ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
