"""Generate a REALISTIC synthetic Stage-2 cache to exercise the pull -> Gate-2
pipeline where stats.nba.com isn't reachable (sandboxes).

DEV/TEST FIXTURE ONLY — not real data. Mirrors the nba_api normalized-dict
schema for gamelog / advanced / tracking and writes the same cache files the
real puller would, plus a manifest and availability map, so gate2_report runs
end to end. Tracking is made deliberately PARTIAL (absent in 2015-16 for some
measures, sparse for low-minute players) to prove the availability map works.

Run:  python -m nba_injury.make_synthetic_cache
"""
from __future__ import annotations

import random

from nba_injury.cache import cached_json, RAW
from nba_injury.nba_client import SEASONS
from nba_injury.pull_features import TRACKING_MEASURES

random.seed(7)

N_PLAYERS = 60


def _manifest():
    manifest = []
    for pid in range(1000, 1000 + N_PLAYERS):
        # each player active in a contiguous run of seasons
        first = random.randint(0, 4)
        last = random.randint(first, len(SEASONS) - 1)
        for s in SEASONS[first:last + 1]:
            manifest.append({"player_id": pid,
                             "player_name": f"Player{pid}", "season": s})
    return manifest


def _gamelog(pid, season):
    n_games = random.randint(20, 75)
    rows = []
    for g in range(n_games):
        rows.append({
            "Player_ID": pid, "SEASON_ID": season, "Game_ID": f"{season}{g:04d}",
            "GAME_DATE": f"{season[:4]}-11-{(g % 27) + 1:02d}",
            "MIN": random.randint(8, 38), "PTS": random.randint(0, 35),
            "REB": random.randint(0, 14), "FTA": random.randint(0, 12),
        })
    return {"PlayerGameLog": rows}


def _advanced(pid, season):
    return {"OverallPlayerDashboard": [{
        "PLAYER_ID": pid, "USG_PCT": round(random.uniform(0.10, 0.34), 3),
        "PACE": round(random.uniform(95, 104), 1),
    }]}


def _tracking_season(season, measure_key, pids):
    """One per-season tracking blob (cached once per season+measure)."""
    # Make some measures unavailable in the earliest season to prove the map.
    if season == "2015-16" and measure_key in ("possessions", "defense"):
        return {f"LeagueDashPtStats": []}  # absent
    rows = []
    for pid in pids:
        if random.random() < 0.12:  # sparse: some players lack tracking rows
            continue
        rows.append({"PLAYER_ID": pid, "MEASURE": measure_key,
                     "VALUE": round(random.uniform(1, 20), 2)})
    return {"LeagueDashPtStats": rows}


def main():
    manifest = cached_json("pull_manifest.json", _manifest, refresh=True)
    pids_by_season = {}
    for m in manifest:
        pids_by_season.setdefault(m["season"], set()).add(m["player_id"])

    # Tier-1 per (player,season)
    for m in manifest:
        pid, s = m["player_id"], m["season"]
        cached_json(f"gamelog_{pid}_{s}.json", lambda pid=pid, s=s: _gamelog(pid, s))
        cached_json(f"advanced_{pid}_{s}.json", lambda pid=pid, s=s: _advanced(pid, s))

    # Tier-2 per (season,measure), cached once
    for s in SEASONS:
        pids = sorted(pids_by_season.get(s, []))
        if not pids:
            continue
        for key in TRACKING_MEASURES:
            cached_json(f"track_{key}_{s}.json",
                        lambda s=s, key=key, pids=pids: _tracking_season(s, key, pids))

    # Availability map (what the real puller derives)
    avail = {}
    for m in manifest:
        pid, s = m["player_id"], m["season"]
        rec = {"player_id": pid, "season": s}
        for key in TRACKING_MEASURES:
            blob = cached_json(f"track_{key}_{s}.json", lambda: {"LeagueDashPtStats": []})
            rows = blob.get("LeagueDashPtStats", [])
            rec[key] = any(r.get("PLAYER_ID") == pid for r in rows)
        avail[f"{pid}_{s}"] = rec
    cached_json("tracking_availability.json", lambda: avail, refresh=True)

    print(f"[synthetic-cache] wrote cache for {len(manifest)} pairs -> {RAW}")
    print("[synthetic-cache] WARNING: dev fixture, NOT real data.")


if __name__ == "__main__":
    main()
