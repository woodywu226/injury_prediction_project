"""Stage 2 — the feature puller. Resumable, rate-limited, cached.

Pulls, per (player, season) in the manifest:
  TIER 1 (always-available backbone, every game, full 10 seasons):
    - player game logs (minutes, points, rebounds, FTA, etc.)  -> playergamelog
    - advanced/usage per game                                   -> via boxscore-
      style season splits where available; usage/pace approximated from team +
      player splits at the season level as a Tier-1 fallback.
  TIER 2 (tracking-dependent enrichment, ~2015-16+, NOT every game):
    - speed/distance, drives, contested shots                   -> player tracking
      endpoints, pulled per season with an EXPLICIT availability record.

Key design points straight from the build plan:
  - Everything caches to data/raw/; a re-run skips already-pulled keys (resume).
  - Tier-2 availability is recorded EXPLICITLY per (player, season): present vs
    absent, never silently dropped. This feeds the modeled-missingness design.
  - Treat as an overnight job. Progress prints + safe to Ctrl-C and resume.

Run:
  python -m nba_injury.pull_features            # full run (overnight)
  python -m nba_injury.pull_features --limit 5  # smoke test a few players
"""
from __future__ import annotations

import argparse
import sys

from nba_injury.cache import cached_json, raw_path, RAW
from nba_injury.nba_client import cached_endpoint
from nba_injury.rosters import build_manifest


# ----------------------------- Tier 1 ---------------------------------------
def pull_gamelog(player_id: int, season: str) -> dict:
    """Per-game log for a player-season. Tier-1 backbone."""
    from nba_api.stats.endpoints import playergamelog

    def build():
        ep = playergamelog.PlayerGameLog(
            player_id=player_id, season=season, season_type_all_star="Regular Season",
            timeout=60,
        )
        return ep.get_normalized_dict()

    return cached_endpoint(f"gamelog_{player_id}_{season}.json", build)


def pull_advanced(player_id: int, season: str) -> dict:
    """Season-level advanced/usage splits (usage rate, pace context). Tier-1.

    Per-game advanced box scores exist but are far heavier to pull; the vision
    doc's Tier-1 spine only needs usage/pace context, so we take the season
    dashboard split here and aggregate per-week in Stage 3.
    """
    from nba_api.stats.endpoints import playerdashboardbyyearoveryear

    def build():
        ep = playerdashboardbyyearoveryear.PlayerDashboardByYearOverYear(
            player_id=player_id, season=season, per_mode_detailed="PerGame",
            measure_type_detailed="Advanced", timeout=60,
        )
        return ep.get_normalized_dict()

    return cached_endpoint(f"advanced_{player_id}_{season}.json", build)


# ----------------------------- Tier 2 ---------------------------------------
# Tracking measure types we want. Each pulled per season at the player level.
TRACKING_MEASURES = {
    "speed_distance": "SpeedDistance",
    "drives": "Drives",
    "defense": "Defense",          # contested shots, rim protection
    "rebounding": "Rebounding",    # rebound chances
    "possessions": "Possessions",  # touches, time of possession
}


def pull_tracking(player_id: int, season: str, measure_key: str) -> dict | None:
    """Per-season player tracking for one measure type. Tier-2 (may be absent).

    Returns the normalized dict on success, or None if the endpoint yields no
    rows for this player-season (recorded as 'absent' by the caller).
    """
    from nba_api.stats.endpoints import playerdashptshots  # noqa: F401  (import guard)
    from nba_api.stats.endpoints import leaguedashptstats

    measure = TRACKING_MEASURES[measure_key]

    def build():
        ep = leaguedashptstats.LeagueDashPtStats(
            season=season, season_type_all_star="Regular Season",
            pt_measure_type=measure, player_or_team="Player", timeout=60,
        )
        return ep.get_normalized_dict()

    data = cached_endpoint(f"track_{measure_key}_{season}.json", build)
    # leaguedashptstats returns ALL players for the season in one call; we cache
    # the whole season once and filter to the player here.
    rows = data.get("LeagueDashPtStats", [])
    mine = [r for r in rows if r.get("PLAYER_ID") == player_id]
    return {"measure": measure, "rows": mine} if mine else None


# ------------------------- availability map ---------------------------------
def record_tracking_availability(player_id: int, season: str) -> dict:
    """Pull every tracking measure for a player-season and record presence.

    Returns {measure_key: bool} availability. The per-season tracking endpoint
    is cached once and reused across players, so this is cheap after the first
    player in a season.
    """
    avail = {}
    for key in TRACKING_MEASURES:
        try:
            result = pull_tracking(player_id, season, key)
            avail[key] = result is not None
        except Exception as exc:  # noqa: BLE001
            print(f"[pull] tracking {key} {season} p{player_id} failed: {exc}",
                  file=sys.stderr)
            avail[key] = False
    return avail


# ------------------------------- driver -------------------------------------
def run(limit: int | None = None) -> dict:
    manifest = build_manifest()
    if limit:
        manifest = manifest[:limit]
    total = len(manifest)
    print(f"[pull] manifest: {total} (player,season) pairs")

    availability: dict[str, dict] = {}
    done = 0
    for entry in manifest:
        pid, season = entry["player_id"], entry["season"]
        tag = f"{entry['player_name']} ({season})"
        try:
            pull_gamelog(pid, season)        # Tier-1
            pull_advanced(pid, season)       # Tier-1
            avail = record_tracking_availability(pid, season)  # Tier-2 + map
            availability[f"{pid}_{season}"] = {
                "player_id": pid, "season": season, **avail,
            }
        except Exception as exc:  # noqa: BLE001
            print(f"[pull] FAILED {tag}: {exc}", file=sys.stderr)
            availability[f"{pid}_{season}"] = {
                "player_id": pid, "season": season, "error": str(exc),
            }
        done += 1
        if done % 25 == 0 or done == total:
            print(f"[pull] {done}/{total} done ({tag})")

    # Persist the explicit availability map — the Gate-2 deliverable.
    cached_json("tracking_availability.json", lambda: availability, refresh=True)
    print(f"[pull] availability map -> {raw_path('tracking_availability.json')}")
    return availability


def main() -> int:
    ap = argparse.ArgumentParser(description="Pull Tier-1/Tier-2 features.")
    ap.add_argument("--limit", type=int, default=None,
                    help="only pull first N manifest entries (smoke test)")
    args = ap.parse_args()
    try:
        run(limit=args.limit)
    except KeyboardInterrupt:
        print("\n[pull] interrupted — safe to resume; cached pulls are kept.")
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"[pull] FAILED: {exc}", file=sys.stderr)
        print("[pull] If this is a proxy/network block, run on a machine with "
              "open egress to stats.nba.com.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
