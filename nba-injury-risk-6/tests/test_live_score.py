"""Tests for the Stage-10 live monitoring job.

Pin the job's contract:
  - demo mode scores players and writes a well-formed record
  - real mode degrades gracefully when the live source is unreachable
  - the season-string helper is correct
  - back-to-back counting is correct
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_injury import live_score as ls  # noqa: E402


def test_current_season_string_format():
    s = ls._current_season_str()
    # format YYYY-YY
    assert len(s) == 7 and s[4] == "-"
    assert s[:4].isdigit() and s[5:].isdigit()


def test_count_back_to_backs():
    days = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 4), date(2024, 1, 5)]
    # consecutive pairs: (1->2) and (4->5) = 2 back-to-backs
    assert ls._count_b2b(days) == 2
    assert ls._count_b2b([date(2024, 1, 1)]) == 0
    assert ls._count_b2b([]) == 0


def test_real_mode_degrades_gracefully_without_network(monkeypatch, tmp_path):
    # ensure a person-period table exists for the model; build a tiny synthetic one
    # by pointing the job at a fabricated table via load_table monkeypatch
    import numpy as np
    from nba_injury import live_score

    df = pd.DataFrame({
        "player_id": list(range(1, 61)) * 2,
        "season": ["2015-16"] * 60 + ["2016-17"] * 60,
        "week_start": [date(2016, 1, 4)] * 120,
        "event": ([0, 0, 0, 0, 0, 0, 0, 0, 0, 1] * 12), "at_risk": [1] * 120,
        **{f: np.random.RandomState(0).uniform(1, 30, 120) for f in
           ["games_this_week", "minutes_this_week", "back_to_backs_this_week",
            "games_in_7days", "cum_season_minutes", "cum_season_games",
            "usg_pct", "pace", "prior_injury_count",
            "trk_speed_distance", "trk_drives", "trk_defense",
            "trk_rebounding", "trk_possessions"]},
    })
    monkeypatch.setattr(live_score, "load_table", lambda: df)
    monkeypatch.setattr(live_score, "LIVE_DIR", tmp_path)
    # nba_api isn't installed in CI test env -> fetch must fail gracefully
    rec = live_score.run(demo=False)
    assert rec["status"] == "fetch_failed"
    assert "detail" in rec


def test_demo_mode_writes_record(monkeypatch, tmp_path):
    import numpy as np
    from nba_injury import live_score

    df = pd.DataFrame({
        "player_id": list(range(1, 61)) * 2,
        "season": ["2015-16"] * 60 + ["2016-17"] * 60,
        "week_start": [date(2016, 1, 4)] * 120,
        "event": ([0, 0, 0, 0, 0, 0, 0, 0, 0, 1] * 12), "at_risk": [1] * 120,
        **{f: np.random.RandomState(1).uniform(1, 30, 120) for f in
           ["games_this_week", "minutes_this_week", "back_to_backs_this_week",
            "games_in_7days", "cum_season_minutes", "cum_season_games",
            "usg_pct", "pace", "prior_injury_count",
            "trk_speed_distance", "trk_drives", "trk_defense",
            "trk_rebounding", "trk_possessions"]},
    })
    monkeypatch.setattr(live_score, "load_table", lambda: df)
    monkeypatch.setattr(live_score, "LIVE_DIR", tmp_path)
    rec = live_score.run(demo=True)
    assert rec["status"] == "ok"
    assert rec["source"] == "synthetic-demo"
    assert rec["n_scored"] > 0
    assert 0 <= rec["mean_hazard"] <= 1
    assert (tmp_path / "latest.json").exists()
    assert (tmp_path / "history.jsonl").exists()
    assert len(rec["top_players"]) > 0
