"""Tests for the Stage-8 monitoring / assurance layer.

Pin the statistical methods and the drift-vs-degradation logic:
  - PSI/KS are ~0 for identical distributions and large for shifted ones
  - the verifiable-event demo separates drift-without-degradation from
    drift-with-degradation
  - late-label two-mode returns both an estimate and an actual
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_injury import monitoring as mon  # noqa: E402


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    # same distribution, different sample -> PSI near 0
    y = rng.normal(0, 1, 5000)
    assert mon.psi(x, y) < 0.05


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    y = rng.normal(2, 1, 5000)  # big mean shift
    assert mon.psi(x, y) > 0.25


def test_ks_zero_for_identical_large_for_shifted():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 4000)
    y_same = rng.normal(0, 1, 4000)
    y_shift = rng.normal(3, 1, 4000)
    assert mon.ks_stat(x, y_same) < 0.1
    assert mon.ks_stat(x, y_shift) > 0.5


def test_psi_handles_too_few_points():
    assert np.isnan(mon.psi(np.array([1.0, 2.0]), np.array([1.0])))


def _scored_world(degrade: bool):
    """Build a scored at-risk frame spanning pre-COVID + COVID seasons.
    If degrade=True, the model's predictions during COVID are made useless."""
    rng = np.random.default_rng(2)
    rows = []
    seasons = ["2017-18", "2018-19", "2019-20", "2020-21"]
    for s in seasons:
        covid = s in mon.COVID_SEASONS
        for i in range(400):
            # covariate shift during COVID: usg_pct distribution moves
            usg = rng.uniform(0.25, 0.4) if covid else rng.uniform(0.1, 0.25)
            y = int(rng.random() < 0.05)
            if covid and degrade:
                pred = rng.random()              # useless predictions
            else:
                pred = 0.04 + 0.3 * y + rng.normal(0, 0.02)  # informative
            rows.append({"season": s, "event": y, "usg_pct": usg,
                         "minutes_this_week": rng.uniform(5, 40),
                         "_pred": np.clip(pred, 0, 1)})
    return pd.DataFrame(rows)


def test_demo_drift_without_degradation():
    df = _scored_world(degrade=False)
    res = mon.verifiable_event_demo(df, ["usg_pct", "minutes_this_week"])
    assert "usg_pct" in res["drifted_features"]      # covariate drifted
    assert "WITHOUT DEGRADATION" in res["verdict"]   # but model held up


def test_demo_drift_with_degradation():
    df = _scored_world(degrade=True)
    res = mon.verifiable_event_demo(df, ["usg_pct", "minutes_this_week"])
    assert "usg_pct" in res["drifted_features"]
    assert "WITH DEGRADATION" in res["verdict"]      # model actually got worse


def test_late_label_returns_estimate_and_actual():
    df = _scored_world(degrade=False)
    rows = mon.estimated_vs_actual_performance(df)
    assert len(rows) >= 2
    for r in rows:
        assert "estimated_brier_prelabels" in r
        assert "actual_brier_postlabels" in r
        assert "estimate_gap" in r


def test_calibration_drift_tracks_windows():
    df = _scored_world(degrade=False)
    rows = mon.calibration_drift(df)
    assert len(rows) == 4
    assert all("reliability_gap" in r for r in rows)
