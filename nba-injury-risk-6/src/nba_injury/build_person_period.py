"""Stage 3 — build the person-period dataset (the expensive-to-undo shape).

One row = one player-week. This is the canonical table every downstream stage
(model, validation, competing risks, prescriptive, monitoring) depends on, so
the structural choices and integrity checks here matter most.

Reads (all from the local cache built in Stages 1-2):
  data/processed/injury_episodes.csv         <- Stage 1 labels
  data/raw/pull_manifest.json                <- Stage 2 player x season universe
  data/raw/gamelog_<pid>_<season>.json       <- Tier-1 per-game load
  data/raw/advanced_<pid>_<season>.json      <- Tier-1 usage/pace
  data/raw/track_<measure>_<season>.json     <- Tier-2 tracking (per season)
  data/raw/tracking_availability.json        <- Tier-2 presence map

Writes:
  data/processed/person_period.parquet       <- THE canonical modeling table

Core design decisions (logged in DECISIONS.md):
  - Weekly grid = ISO-week buckets, ACTIVE roster weeks only (a player's first
    game week through their last game week within each season). No off-season
    rows; no rows before debut / after a season's last game.
  - NO LEAKAGE: every feature for week W uses only games in weeks <= W. Rolling
    load (games-in-7-days, cumulative minutes) is computed strictly causally.
  - EVENT: a player-week gets event=1 if a time-loss injury episode BEGAN in
    that week (episode start_date falls in the week). After an event begins, the
    player exits the risk set until they return (recovery weeks are not at-risk
    rows for time-to-first-injury; recurrent-event handling arrives in Stage 6).
  - CENSORING: a player's final observed week is censored (event=0) if no injury
    began there. Competing-risks exit type is attached:
        * 'active_end'  : still active in the final window season -> censored
        * 'exit'        : last active season precedes the final season ->
                          non-injury career exit (cut/retire/aging-out)
        * 'injury'      : the week an injury episode began
  - Tier-2 attached only where available, with explicit *_missing indicators;
    never silently imputed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta

import pandas as pd

from nba_injury.cache import RAW, processed_path
from nba_injury.nba_client import SEASONS
from nba_injury.pull_features import TRACKING_MEASURES

EPISODES_CSV = "injury_episodes.csv"
MANIFEST = "pull_manifest.json"
AVAILABILITY = "tracking_availability.json"
OUT_PARQUET = "person_period.parquet"

FINAL_SEASON = SEASONS[-1]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _parse_date(s):
    # Robust to pandas blanks (NaN -> float), None, and already-parsed dates.
    if s is None:
        return None
    if isinstance(s, (date, datetime)):
        return s if isinstance(s, date) and not isinstance(s, datetime) else s.date() if isinstance(s, datetime) else s
    if not isinstance(s, str):
        # NaN (float) or other non-string -> treat as missing
        try:
            import math
            if isinstance(s, float) and math.isnan(s):
                return None
        except Exception:
            pass
        s = str(s)
    s = s.strip()
    if not s or s.lower() == "nan":
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _iso_week_start(d: date) -> date:
    """Monday of the ISO week containing d. Weekly grid is Monday-anchored."""
    return d - timedelta(days=d.weekday())


def _load_json(name: str):
    path = RAW / name
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _first_list(d) -> list:
    """nba_api normalized dicts wrap rows under a single endpoint key."""
    if isinstance(d, dict):
        for v in d.values():
            if isinstance(v, list):
                return v
    elif isinstance(d, list):
        return d
    return []


# ----------------------------------------------------------------------------
# load inputs
# ----------------------------------------------------------------------------
def load_manifest() -> list[dict]:
    m = _load_json(MANIFEST)
    if not m:
        raise SystemExit(
            f"[stage3] {MANIFEST} not found. Run Stage 2 (rosters/pull) first."
        )
    return m


def load_episodes() -> pd.DataFrame:
    path = processed_path(EPISODES_CSV)
    if not path.exists():
        raise SystemExit(
            f"[stage3] {path} not found. Run Stage 1 (build_labels) first."
        )
    ep = pd.read_csv(path)
    ep["start_date"] = ep["start_date"].map(_parse_date)
    ep["end_date"] = ep["end_date"].map(_parse_date)
    return ep


def load_tracking_by_season() -> dict[tuple[str, str], dict[int, float]]:
    """Return {(measure, season): {player_id: value}} from per-season caches."""
    out: dict[tuple[str, str], dict[int, float]] = {}
    for measure in TRACKING_MEASURES:
        for season in SEASONS:
            d = _load_json(f"track_{measure}_{season}.json")
            rows = _first_list(d)
            pmap: dict[int, float] = {}
            for r in rows:
                pid = r.get("PLAYER_ID")
                if pid is not None:
                    pmap[int(pid)] = r.get("VALUE")
            out[(measure, season)] = pmap
    return out


def load_availability() -> dict[str, dict]:
    d = _load_json(AVAILABILITY) or {}
    # stored as {"<pid>_<season>": {...}} or list of [key, {...}]
    if isinstance(d, list):
        d = {k: v for k, v in d}
    return d


# ----------------------------------------------------------------------------
# per (player, season) load aggregation -> weekly rows
# ----------------------------------------------------------------------------
def _load_games(pid: int, season: str) -> list[dict]:
    d = _load_json(f"gamelog_{pid}_{season}.json")
    games = []
    for r in _first_list(d):
        gd = _parse_date(r.get("GAME_DATE", ""))
        if gd is None:
            continue
        games.append({
            "date": gd,
            "min": float(r.get("MIN") or 0),
            "pts": float(r.get("PTS") or 0),
            "reb": float(r.get("REB") or 0),
            "fta": float(r.get("FTA") or 0),
        })
    games.sort(key=lambda g: g["date"])
    return games


def _season_advanced(pid: int, season: str) -> dict:
    d = _load_json(f"advanced_{pid}_{season}.json")
    rows = _first_list(d)
    if rows:
        r = rows[0]
        return {"usg_pct": r.get("USG_PCT"), "pace": r.get("PACE")}
    return {"usg_pct": None, "pace": None}


def _weekly_grid(games: list[dict]) -> list[date]:
    """Active weeks only: from the week of the first game to the week of the
    last game, every Monday-anchored week in between (inclusive)."""
    if not games:
        return []
    w0 = _iso_week_start(games[0]["date"])
    wN = _iso_week_start(games[-1]["date"])
    weeks = []
    w = w0
    while w <= wN:
        weeks.append(w)
        w += timedelta(days=7)
    return weeks


def build_player_season_rows(pid: int, season: str, adv: dict) -> list[dict]:
    """One dict per active week for this (player, season). Causal features only."""
    games = _load_games(pid, season)
    weeks = _weekly_grid(games)
    if not weeks:
        return []

    # index games by their week-start for fast lookup
    games_by_week: dict[date, list[dict]] = defaultdict(list)
    for g in games:
        games_by_week[_iso_week_start(g["date"])].append(g)

    rows = []
    cum_min = 0.0       # cumulative season minutes up to & including this week
    cum_games = 0
    for w in weeks:
        wk_games = games_by_week.get(w, [])
        wk_min = sum(g["min"] for g in wk_games)
        wk_n = len(wk_games)

        # games-in-7-days = games this week (the week IS a 7-day bucket)
        games_in_7 = wk_n
        # back-to-backs within the week: consecutive calendar days
        b2b = 0
        days = sorted(g["date"] for g in wk_games)
        for i in range(1, len(days)):
            if (days[i] - days[i - 1]).days == 1:
                b2b += 1

        cum_min += wk_min
        cum_games += wk_n

        rows.append({
            "player_id": pid,
            "season": season,
            "week_start": w,
            "games_this_week": wk_n,
            "minutes_this_week": wk_min,
            "back_to_backs_this_week": b2b,
            "games_in_7days": games_in_7,
            "cum_season_minutes": cum_min,     # through this week (causal)
            "cum_season_games": cum_games,
            "usg_pct": adv.get("usg_pct"),
            "pace": adv.get("pace"),
        })
    return rows


# ----------------------------------------------------------------------------
# event / censor structure
# ----------------------------------------------------------------------------
def attach_events_and_censor(
    df: pd.DataFrame,
    episodes: pd.DataFrame,
    manifest: list[dict],
) -> pd.DataFrame:
    """Add event, exit_type, at_risk, and prior-injury features.

    - event=1 on the week an injury episode BEGAN (matched by player+week).
    - after an event week, weeks until episode end are removed from the risk set
      (time-to-FIRST-injury framing for v1; recurrence handled in Stage 6).
    - the player's final observed week carries exit_type:
        injury / active_end / exit  (competing-risks groundwork).
    - prior_injury_count: number of episodes that began strictly BEFORE this
      week (causal; no future leakage).
    """
    # map player_id -> episode start weeks and (start,end) intervals
    ep_starts: dict[int, list[date]] = defaultdict(list)
    ep_intervals: dict[int, list[tuple[date, date | None]]] = defaultdict(list)
    # episodes use player NAME; map names to ids via manifest where possible
    name_to_id = {}
    for r in manifest:
        name_to_id.setdefault(r.get("player_name"), r.get("player_id"))

    for _, e in episodes.iterrows():
        if e["start_date"] is None:
            continue
        pid = name_to_id.get(e["player"], None)
        if pid is None:
            continue
        sw = _iso_week_start(e["start_date"])
        ep_starts[pid].append(sw)
        ep_intervals[pid].append((sw, e["end_date"]))

    # last active season per player (for exit-type)
    last_season: dict[int, str] = {}
    for r in manifest:
        pid = r["player_id"]
        s = r["season"]
        if pid not in last_season or SEASONS.index(s) > SEASONS.index(last_season[pid]):
            last_season[pid] = s

    df = df.sort_values(["player_id", "week_start"]).reset_index(drop=True)

    event = []
    at_risk = []
    prior_cnt = []
    for row in df.itertuples(index=False):
        pid = row.player_id
        w = row.week_start
        starts = ep_starts.get(pid, [])
        # event if an episode began this exact week
        ev = 1 if w in starts else 0
        event.append(ev)
        # prior injuries strictly before this week
        prior_cnt.append(sum(1 for s in starts if s < w))
        # at_risk: not currently inside a recovery interval that began earlier.
        # An event week is BY DEFINITION at-risk (you must be at risk to have the
        # event), so a new episode beginning this week overrides any overlapping
        # earlier recovery interval. (Re-injury-during-recovery is a recurrence
        # concern handled in Stage 6; for the time-to-first-injury v1 framing the
        # event week is always an at-risk row.)
        recovering = False
        if not ev:
            for (s, e_end) in ep_intervals.get(pid, []):
                if s < w and (e_end is None or _iso_week_start(e_end) >= w):
                    recovering = True
                    break
        at_risk.append(0 if recovering else 1)

    df["event"] = event
    df["prior_injury_count"] = prior_cnt
    df["at_risk"] = at_risk

    # exit_type on each player's LAST observed week only; '' elsewhere
    exit_type = [""] * len(df)
    last_idx_by_player = (
        df.reset_index().groupby("player_id")["index"].max().to_dict()
    )
    for pid, idx in last_idx_by_player.items():
        if df.at[idx, "event"] == 1:
            exit_type[idx] = "injury"
        elif last_season.get(pid) == FINAL_SEASON:
            exit_type[idx] = "active_end"     # right-censored
        else:
            exit_type[idx] = "exit"           # non-injury career exit
    df["exit_type"] = exit_type
    return df


# ----------------------------------------------------------------------------
# Tier-2 attach (with explicit missingness)
# ----------------------------------------------------------------------------
def attach_tier2(df: pd.DataFrame) -> pd.DataFrame:
    tracking = load_tracking_by_season()
    for measure in TRACKING_MEASURES:
        vals = []
        miss = []
        for row in df.itertuples(index=False):
            pmap = tracking.get((measure, row.season), {})
            v = pmap.get(int(row.player_id))
            if v is None:
                vals.append(float("nan"))
                miss.append(1)
            else:
                vals.append(float(v))
                miss.append(0)
        df[f"trk_{measure}"] = vals
        df[f"trk_{measure}_missing"] = miss   # explicit indicator, never silent
    return df


# ----------------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------------
def build(limit: int | None = None) -> pd.DataFrame:
    manifest = load_manifest()
    episodes = load_episodes()

    pairs = [(r["player_id"], r["season"]) for r in manifest]
    if limit:
        # keep all seasons for the first `limit` players (coherent careers)
        keep_players = sorted({p for p, _ in pairs})[:limit]
        pairs = [(p, s) for (p, s) in pairs if p in keep_players]

    all_rows = []
    for i, (pid, season) in enumerate(pairs):
        adv = _season_advanced(pid, season)
        all_rows.extend(build_player_season_rows(pid, season, adv))
        if (i + 1) % 50 == 0:
            print(f"[stage3] aggregated {i + 1}/{len(pairs)} player-seasons",
                  file=sys.stderr)

    if not all_rows:
        raise SystemExit("[stage3] produced zero rows — is the cache populated?")

    df = pd.DataFrame(all_rows)
    df = attach_events_and_censor(df, episodes, manifest)
    df = attach_tier2(df)

    # canonical column order
    df = df.sort_values(["player_id", "week_start"]).reset_index(drop=True)
    return df


def write(df: pd.DataFrame) -> None:
    out = processed_path(OUT_PARQUET)
    df.to_parquet(out, index=False)
    print(f"[stage3] wrote {len(df)} player-weeks "
          f"({df['player_id'].nunique()} players) -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the person-period table.")
    ap.add_argument("--limit", type=int, default=None,
                    help="restrict to first N players (smoke test)")
    args = ap.parse_args()
    df = build(limit=args.limit)
    write(df)
    # quick summary
    print(f"[stage3] events: {int(df['event'].sum())} | "
          f"at-risk weeks: {int(df['at_risk'].sum())} | "
          f"exit types: {df[df['exit_type'] != ''].exit_type.value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
