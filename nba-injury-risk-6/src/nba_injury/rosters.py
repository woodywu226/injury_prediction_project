"""Stage 2 — resolve WHICH players to pull, per season.

We don't want to pull all-time players; we want everyone who logged a game in
each of the 10 seasons. `commonallplayers` (with a season param) returns the
league's player list with their active-season span, which lets us scope the
per-player game-log pulls to real participants.

Output cache: data/raw/roster_<season>.json (one per season), plus a combined
data/raw/pull_manifest.json listing every (player_id, player_name, season) the
puller should fetch. The manifest is the resumable work list.
"""
from __future__ import annotations

from nba_injury.cache import cached_json, raw_path
from nba_injury.nba_client import SEASONS, cached_endpoint


def _fetch_season_roster(season: str):
    from nba_api.stats.endpoints import commonallplayers

    def build():
        ep = commonallplayers.CommonAllPlayers(
            is_only_current_season=0, league_id="00", season=season, timeout=60,
        )
        return ep.get_normalized_dict()

    return cached_endpoint(f"roster_{season}.json", build)


def _season_active(row: dict, season: str) -> bool:
    """commonallplayers gives FROM_YEAR/TO_YEAR; keep players whose span covers
    this season's starting year (e.g. '2015-16' -> 2015)."""
    start_year = int(season[:4])
    try:
        frm = int(row.get("FROM_YEAR") or 0)
        to = int(row.get("TO_YEAR") or 0)
    except (TypeError, ValueError):
        return True  # keep on parse failure; better a spurious pull than a gap
    return frm <= start_year <= to


def build_manifest(refresh: bool = False) -> list[dict]:
    """Return the full pull list: one entry per (player, season)."""
    def build():
        manifest: list[dict] = []
        seen = set()
        for season in SEASONS:
            data = _fetch_season_roster(season)
            players = data.get("CommonAllPlayers", [])
            for p in players:
                if not _season_active(p, season):
                    continue
                pid = p.get("PERSON_ID")
                key = (pid, season)
                if pid is None or key in seen:
                    continue
                seen.add(key)
                manifest.append({
                    "player_id": pid,
                    "player_name": p.get("DISPLAY_FIRST_LAST", ""),
                    "season": season,
                })
        return manifest

    return cached_json("pull_manifest.json", build, refresh=refresh)


def manifest_summary(manifest: list[dict]) -> dict:
    by_season: dict[str, int] = {}
    players = set()
    for m in manifest:
        by_season[m["season"]] = by_season.get(m["season"], 0) + 1
        players.add(m["player_id"])
    return {
        "total_pairs": len(manifest),
        "unique_players": len(players),
        "by_season": dict(sorted(by_season.items())),
    }


if __name__ == "__main__":
    mani = build_manifest()
    summ = manifest_summary(mani)
    print(f"[roster] manifest cached -> {raw_path('pull_manifest.json')}")
    print(f"[roster] {summ['total_pairs']} (player,season) pairs, "
          f"{summ['unique_players']} unique players")
    for s, n in summ["by_season"].items():
        print(f"   {s}: {n} players")
