"""Tests for the Stage-4 discrete-time hazard model.

Pins the two properties that matter most for credibility:
  - the temporal split never puts a test-season week into training (no leakage)
  - only at-risk weeks enter the model
and basic metric plumbing.
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_injury import model_hazard as mh  # noqa: E402
from nba_injury.nba_client import SEASONS  # noqa: E402


def _toy_table(n_per_season=40):
    """Build a small person-period-shaped table spanning several seasons."""
    rng = np.random.default_rng(0)
    rows = []
    pid = 1
    for s in SEASONS:
        for i in range(n_per_season):
            minutes = rng.uniform(5, 40)
            # event probability rises with minutes (signal to recover)
            ev = int(rng.random() < (minutes / 400))
            rows.append({
                "player_id": pid, "season": s,
                "week_start": date(int(s[:4]), 11, 1),
                "games_this_week": rng.integers(1, 5),
                "minutes_this_week": minutes,
                "back_to_backs_this_week": 0,
                "games_in_7days": rng.integers(1, 5),
                "cum_season_minutes": minutes * rng.integers(1, 10),
                "cum_season_games": rng.integers(1, 30),
                "usg_pct": rng.uniform(0.1, 0.3), "pace": rng.uniform(95, 104),
                "prior_injury_count": rng.integers(0, 3),
                "trk_speed_distance": rng.uniform(1, 50),
                "trk_drives": rng.uniform(1, 50),
                # partially missing (like real Tier-2 coverage), not 100% NaN
                "trk_defense": (np.nan if rng.random() < 0.4 else rng.uniform(1, 50)),
                "trk_rebounding": rng.uniform(1, 50),
                "trk_possessions": rng.uniform(1, 50),
                "event": ev, "at_risk": 1, "exit_type": "",
            })
            pid += 1
    return pd.DataFrame(rows)


def test_temporal_split_has_no_season_overlap():
    df = _toy_table()
    train, test, train_seasons, test_seasons = mh.temporal_split(df, n_test_seasons=3)
    assert set(train_seasons).isdisjoint(set(test_seasons))
    # every test season is strictly later than every train season
    assert max(SEASONS.index(s) for s in train_seasons) < \
           min(SEASONS.index(s) for s in test_seasons)


def test_split_keeps_only_at_risk():
    df = _toy_table()
    df.loc[df.index[:10], "at_risk"] = 0  # mark some recovery weeks
    train, test, *_ = mh.temporal_split(df)
    assert (train["at_risk"] == 1).all()
    assert (test["at_risk"] == 1).all()


def test_hgb_tolerates_nan_tier2():
    df = _toy_table()
    train, test, *_ = mh.temporal_split(df)
    # should not raise despite trk_defense being all-NaN
    _, metrics, _ = mh.fit_hgb(train, test)
    assert metrics.model == "hgb_tier1_2"
    assert metrics.n_test > 0


def test_metrics_are_finite_when_events_present():
    df = _toy_table(n_per_season=80)
    train, test, *_ = mh.temporal_split(df)
    if test["event"].sum() > 0:
        _, m, _ = mh.fit_logistic(train, test)
        assert 0.0 <= m.auc_pr <= 1.0
        assert 0.0 <= m.brier <= 1.0
