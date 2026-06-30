"""Build a COHERENT synthetic world to exercise the Stage-3 person-period
builder + Gate-3 end to end where the network isn't reachable.

Unlike the per-stage fixtures (which use independent id/name schemes), this one
makes the Stage-2 feature cache and the Stage-1 injury episodes share the SAME
players, so the event<->episode alignment actually connects and Gate-3's checks
are meaningfully exercised.

DEV/TEST FIXTURE ONLY — not real data. Writes:
  data/raw/gamelog_<pid>_<season>.json
  data/raw/advanced_<pid>_<season>.json
  data/raw/track_<measure>_<season>.json
  data/raw/tracking_availability.json
  data/raw/pull_manifest.json
  data/processed/injury_episodes.csv

Run:  python -m nba_injury.make_synthetic_world
"""
from __future__ import annotations

import csv
import json
import random
from datetime import date, timedelta

from nba_injury.cache import RAW, processed_path, cached_json
from nba_injury.nba_client import SEASONS
from nba_injury.pull_features import TRACKING_MEASURES

random.seed(11)

N_PLAYERS = 40
SEASON_START_MONTH = 10
SEASON_GAMES = 60  # synthetic; real seasons ~82 but 60 keeps the fixture light


def _season_year(season: str) -> int:
    return int(season.split("-")[0])


def _gen_games(pid: int, season: str) -> list[dict]:
    """Generate ~SEASON_GAMES game rows spread across the season."""
    yr = _season_year(season)
    start = date(yr, SEASON_START_MONTH, random.randint(18, 28))
    rows = []
    d = start
    n = SEASON_GAMES + random.randint(-10, 10)
    for _ in range(max(20, n)):
        d += timedelta(days=random.choice([1, 2, 2, 3, 3, 4]))
        if d.month in (7, 8, 9):  # past season end
            break
        rows.append({
            "PLAYER_ID": pid,
            "GAME_DATE": d.isoformat(),
            "MIN": round(random.uniform(8, 38), 1),
            "PTS": random.randint(0, 35),
            "REB": random.randint(0, 15),
            "FTA": random.randint(0, 12),
        })
    return rows


def main():
    # clear stale per-stage fixtures so the world is internally consistent
    for f in RAW.glob("gamelog_*.json"):
        f.unlink()
    for f in RAW.glob("advanced_*.json"):
        f.unlink()
    for f in RAW.glob("track_*.json"):
        f.unlink()

    pids = list(range(2000, 2000 + N_PLAYERS))
    manifest = []
    availability = {}

    # each player active in a contiguous run of seasons
    player_seasons: dict[int, list[str]] = {}
    for pid in pids:
        first = random.randint(0, 5)
        last = random.randint(first, len(SEASONS) - 1)
        seasons = SEASONS[first:last + 1]
        player_seasons[pid] = seasons
        for s in seasons:
            manifest.append({"player_id": pid, "player_name": f"Player{pid}",
                             "season": s})
            cached_json(f"gamelog_{pid}_{s}.json",
                        lambda pid=pid, s=s: {"PlayerGameLog": _gen_games(pid, s)})
            cached_json(f"advanced_{pid}_{s}.json",
                        lambda: {"OverallPlayerDashboard": [{
                            "USG_PCT": round(random.uniform(0.12, 0.34), 3),
                            "PACE": round(random.uniform(95, 104), 1)}]})
            avail = {}
            for m in TRACKING_MEASURES:
                # tracking absent in 2015-16 for some measures; sparse for low mins
                present = not (s == "2015-16" and m in ("defense", "possessions"))
                present = present and (random.random() > 0.1)
                avail[m] = present
            availability[f"{pid}_{s}"] = {"player_id": pid, "season": s, **avail}

    # per-season tracking files: one row per player present that season
    for m in TRACKING_MEASURES:
        for s in SEASONS:
            rows = []
            for pid in pids:
                if s in player_seasons[pid] and availability[f"{pid}_{s}"][m]:
                    rows.append({"PLAYER_ID": pid, "VALUE": round(random.uniform(1, 50), 2)})
            cached_json(f"track_{m}_{s}.json", lambda rows=rows: {f"PlayerTracking": rows})

    cached_json("pull_manifest.json", lambda: manifest)
    cached_json("tracking_availability.json", lambda: availability)

    # injuries: for each player, a few episodes whose start dates fall on real
    # game weeks of seasons they actually played -> alignment will connect.
    ep_rows = []
    for pid in pids:
        for s in player_seasons[pid]:
            if random.random() < 0.5:
                games = json.load(open(RAW / f"gamelog_{pid}_{s}.json"))["PlayerGameLog"]
                if not games:
                    continue
                g = random.choice(games)
                start = date.fromisoformat(g["GAME_DATE"])
                closes = random.random() < 0.8
                end = (start + timedelta(days=random.randint(7, 40))) if closes else None
                cat = random.choice(["lower_limb_soft_tissue", "knee_ligament",
                                     "back", "achilles", "illness"])
                ep_rows.append({
                    "player": f"Player{pid}",
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat() if end else "",
                    "category": cat,
                    "severe_tail": cat == "achilles",
                    "days_out": (end - start).days if end else "",
                    "raw_notes": "synthetic",
                })

    out = processed_path("injury_episodes.csv")
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["player", "start_date", "end_date",
                                           "category", "severe_tail", "days_out",
                                           "raw_notes"])
        w.writeheader()
        w.writerows(ep_rows)

    print(f"[world] {N_PLAYERS} players, {len(manifest)} player-seasons, "
          f"{len(ep_rows)} injury episodes")
    print("[world] WARNING: synthetic dev fixture, NOT real data.")


if __name__ == "__main__":
    main()
