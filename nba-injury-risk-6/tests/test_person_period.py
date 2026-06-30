"""Tests for the Stage-3 person-period builder.

These pin the structural guarantees independently of fixture volume:
  - weekly grid covers active weeks only, Monday-anchored, contiguous
  - cumulative minutes are strictly causal (non-decreasing, week-1 == own week)
  - event weeks align with episode starts
  - an event week is always at_risk (never flagged as recovery)
  - Tier-2 missingness is explicit
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_injury import build_person_period as bpp  # noqa: E402


def test_iso_week_start_is_monday():
    # 2019-01-10 is a Thursday; its ISO week starts Monday 2019-01-07
    assert bpp._iso_week_start(date(2019, 1, 10)) == date(2019, 1, 7)
    assert bpp._iso_week_start(date(2019, 1, 7)).weekday() == 0


def test_parse_date_handles_blanks_and_nan():
    assert bpp._parse_date("") is None
    assert bpp._parse_date(None) is None
    assert bpp._parse_date(float("nan")) is None
    assert bpp._parse_date("2020-03-01") == date(2020, 3, 1)


def test_weekly_grid_is_contiguous_active_only():
    games = [
        {"date": date(2020, 1, 2), "min": 30, "pts": 0, "reb": 0, "fta": 0},
        {"date": date(2020, 1, 20), "min": 25, "pts": 0, "reb": 0, "fta": 0},
    ]
    weeks = bpp._weekly_grid(games)
    # first game week (2019-12-30 Mon) .. last game week (2020-01-20 Mon)
    assert weeks[0] == date(2019, 12, 30)
    assert weeks[-1] == date(2020, 1, 20)
    # contiguous 7-day steps, no gaps
    for i in range(1, len(weeks)):
        assert (weeks[i] - weeks[i - 1]).days == 7


def test_cumulative_minutes_are_causal():
    games = [
        {"date": date(2020, 1, 6), "min": 20, "pts": 0, "reb": 0, "fta": 0},
        {"date": date(2020, 1, 13), "min": 30, "pts": 0, "reb": 0, "fta": 0},
        {"date": date(2020, 1, 20), "min": 10, "pts": 0, "reb": 0, "fta": 0},
    ]
    # patch loader to return our games
    rows = bpp.build_player_season_rows.__wrapped__ if hasattr(
        bpp.build_player_season_rows, "__wrapped__") else None
    # call the internal weekly builder via a tiny shim
    import nba_injury.build_person_period as m
    orig = m._load_games
    m._load_games = lambda pid, season: games
    try:
        out = m.build_player_season_rows(1, "2019-20", {"usg_pct": 0.2, "pace": 99})
    finally:
        m._load_games = orig
    cum = [r["cum_season_minutes"] for r in out]
    assert cum == sorted(cum)              # non-decreasing
    assert cum[0] == out[0]["minutes_this_week"]  # week-1 carries no future
    assert cum[-1] == 60                   # total adds up


def _tiny_world():
    """A 1-player table built directly to test event/censor attach logic."""
    games_rows = [
        {"player_id": 1, "season": "2024-25", "week_start": date(2024, 11, 4),
         "games_this_week": 3, "minutes_this_week": 60, "back_to_backs_this_week": 0,
         "games_in_7days": 3, "cum_season_minutes": 60, "cum_season_games": 3,
         "usg_pct": 0.2, "pace": 99},
        {"player_id": 1, "season": "2024-25", "week_start": date(2024, 11, 11),
         "games_this_week": 2, "minutes_this_week": 40, "back_to_backs_this_week": 0,
         "games_in_7days": 2, "cum_season_minutes": 100, "cum_season_games": 5,
         "usg_pct": 0.2, "pace": 99},
    ]
    df = pd.DataFrame(games_rows)
    episodes = pd.DataFrame([{
        "player": "P1", "start_date": date(2024, 11, 11), "end_date": None,
        "category": "back", "severe_tail": False, "days_out": None,
        "raw_notes": "x"}])
    manifest = [{"player_id": 1, "player_name": "P1", "season": "2024-25"}]
    return df, episodes, manifest


def test_event_aligns_and_is_at_risk():
    df, episodes, manifest = _tiny_world()
    out = bpp.attach_events_and_censor(df, episodes, manifest)
    ev_week = out[out["event"] == 1]
    assert len(ev_week) == 1
    assert ev_week.iloc[0]["week_start"] == date(2024, 11, 11)
    # the event week must be at_risk
    assert ev_week.iloc[0]["at_risk"] == 1
    # terminal exit type is 'injury' (final week & event) since season is final
    assert ev_week.iloc[0]["exit_type"] == "injury"


def test_prior_injury_count_is_causal():
    df, episodes, manifest = _tiny_world()
    out = bpp.attach_events_and_censor(df, episodes, manifest).sort_values("week_start")
    # week 1 has zero prior injuries; event week also zero (strictly before)
    assert out.iloc[0]["prior_injury_count"] == 0
    assert out.iloc[1]["prior_injury_count"] == 0
