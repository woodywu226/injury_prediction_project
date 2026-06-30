"""Tests for the Stage-5 gauntlet.

Two things must hold:
  - the structural causality checks correctly PASS clean data and FAIL leaky data
  - the stylistic-comparables machinery ranks an injured high-load player above
    its healthy low-load look-alikes (and the cohort excludes injured players)
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_injury import audit_leakage as al  # noqa: E402
from nba_injury import stylistic_comparables as sc  # noqa: E402


def _clean_player_season(pid, season_year, n_weeks=20, minutes=30):
    rows = []
    cum = 0.0
    cg = 0
    w = date(season_year, 11, 4)
    for i in range(n_weeks):
        cum += minutes
        cg += 2
        rows.append({
            "player_id": pid, "season": f"{season_year}-{str(season_year+1)[2:]}",
            "week_start": w, "games_this_week": 2, "minutes_this_week": minutes,
            "back_to_backs_this_week": 0, "games_in_7days": 2,
            "cum_season_minutes": cum, "cum_season_games": cg,
            "usg_pct": 0.2, "pace": 99, "prior_injury_count": 0,
            "event": 0, "at_risk": 1, "exit_type": "",
        })
        w += timedelta(days=7)
    return rows


def test_cumulative_causality_passes_clean():
    df = pd.DataFrame(_clean_player_season(1, 2018))
    ok, _ = al.check_cumulative_causality(df)
    assert ok


def test_cumulative_causality_catches_leak():
    # plant a leak: cumulative minutes DROP mid-season (impossible if causal)
    rows = _clean_player_season(1, 2018)
    rows[10]["cum_season_minutes"] = 0.0  # a future-aware/corrupt value
    df = pd.DataFrame(rows)
    ok, detail = al.check_cumulative_causality(df)
    assert not ok
    assert "non-causal" in detail


def test_prior_injury_monotonic_catches_decrease():
    rows = _clean_player_season(1, 2018)
    for i, r in enumerate(rows):
        r["prior_injury_count"] = 1 if i < 5 else 0  # decreases -> leak/corruption
    df = pd.DataFrame(rows)
    ok, _ = al.check_prior_injury_monotonic(df)
    assert not ok


def _two_style_world():
    """Index = high-load injured player; comparables = healthy low-load and a
    healthy high-load look-alike. The cohort must exclude injured players."""
    rows = []
    # index player: high minutes, has an event
    r = _clean_player_season(100, 2018, minutes=38)
    r[10]["event"] = 1
    rows += r
    # healthy high-load look-alike (similar style, NO event)
    rows += _clean_player_season(101, 2018, minutes=37)
    # healthy low-load players
    rows += _clean_player_season(102, 2018, minutes=12)
    rows += _clean_player_season(103, 2018, minutes=10)
    # an injured high-load player who must NOT be selected as a comparable
    r2 = _clean_player_season(104, 2018, minutes=39)
    r2[5]["event"] = 1
    rows += r2
    return pd.DataFrame(rows)


def test_comparable_cohort_excludes_injured():
    df = _two_style_world()
    _, healthy = sc.build_comparable_cohort(df, index_player=100, k=10)
    assert 104 not in healthy           # injured look-alike excluded
    assert 100 not in healthy           # index itself excluded
    assert all(df[df.player_id == p]["event"].sum() == 0 for p in healthy)


def test_nearest_comparable_is_style_match():
    df = _two_style_world()
    _, healthy = sc.build_comparable_cohort(df, index_player=100, k=1)
    # nearest healthy by style to the high-load index should be the high-load one
    assert healthy[0] == 101


def test_style_profile_uses_only_pre_window():
    df = _two_style_world()
    cutoff = date(2018, 11, 4) + timedelta(days=7 * 5)
    prof = sc._player_style_profile(df, 100, before_week=cutoff)
    # only 5 weeks before cutoff are averaged -> cum_season_minutes reflects <= wk5
    assert prof is not None
    # the max cum minutes pre-cutoff is 5*38=190; mean must be <= that
    idx = sc.STYLE_FEATURES.index("cum_season_minutes")
    assert prof[idx] <= 190 + 1e-6
