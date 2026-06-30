"""Tests for the Stage-7 prescriptive core.

The most important guarantees are about DISCIPLINE, not numbers:
  - the counterfactual guard REFUSES non-modifiable features (no overclaim)
  - the standing caveat is non-empty and attached
  - modifiable / non-modifiable sets are disjoint
  - a counterfactual on a modifiable feature actually moves the hazard
  - observational validation returns a verdict (support OR honest null)
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_injury import prescriptive as pr  # noqa: E402


def test_partition_is_disjoint_and_nonempty():
    mod = set(pr.MODIFIABLE_FEATURES)
    fixed = set(pr.NON_MODIFIABLE_FEATURES)
    assert mod and fixed
    assert mod.isdisjoint(fixed)


def test_standing_caveat_nonempty_and_clear():
    assert len(pr.STANDING_CAVEAT) > 50
    low = pr.STANDING_CAVEAT.lower()
    assert "not advice" in low or "not a promise" in low
    assert "model" in low


def _toy_fit():
    """Tiny trained logistic pipeline + a representative row."""
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    feats = pr.MODIFIABLE_FEATURES + pr.NON_MODIFIABLE_FEATURES
    rng = np.random.default_rng(0)
    n = 400
    data = {f: rng.uniform(1, 40, n) for f in feats}
    df = pd.DataFrame(data)
    # event rises with minutes_this_week so a lever has an effect to find
    pmin = df["minutes_this_week"] / df["minutes_this_week"].max()
    df["event"] = (rng.random(n) < pmin * 0.3).astype(int)
    clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf.fit(df[feats].to_numpy(float), df["event"].to_numpy(int))
    return clf, feats, df.iloc[0]


def test_counterfactual_refuses_fixed_feature():
    clf, feats, row = _toy_fit()
    fixed = pr.NON_MODIFIABLE_FEATURES[0]
    res = pr.counterfactual_hazard(clf, row, feats, fixed, 0.0)
    assert "error" in res                      # MUST refuse
    assert "overclaim" in res["error"].lower()


def test_counterfactual_allows_modifiable_feature():
    clf, feats, row = _toy_fit()
    lever = "minutes_this_week"
    res = pr.counterfactual_hazard(clf, row, feats, lever, float(row[lever]) * 0.5)
    assert "error" not in res
    assert "hazard_delta" in res
    assert res["framing"]                       # framing string present
    # reducing minutes should not increase the hazard estimate (monotone-ish)
    # (not a hard assert on sign — just that it returns a finite delta)
    assert np.isfinite(res["hazard_delta"])


def test_counterfactual_changes_hazard():
    clf, feats, row = _toy_fit()
    res = pr.counterfactual_hazard(clf, row, feats, "minutes_this_week",
                                   float(row["minutes_this_week"]) * 0.2)
    assert res["hazard_base"] != res["hazard_counterfactual"]


def test_observational_validation_returns_verdict():
    # build a person-period-shaped frame with at_risk + a modifiable lever
    rng = np.random.default_rng(1)
    n = 600
    df = pd.DataFrame({
        "player_id": rng.integers(1, 50, n),
        "at_risk": 1,
        "usg_pct": rng.uniform(0.1, 0.3, n),
        "cum_season_minutes": rng.uniform(100, 2000, n),
        "minutes_this_week": rng.uniform(5, 40, n),
        "event": rng.integers(0, 2, n),
    })
    res = pr.validate_lever_observationally(df, "minutes_this_week")
    assert "verdict" in res
    assert "caveat" in res


def test_validation_rejects_non_modifiable():
    df = pd.DataFrame({"at_risk": [1], "prior_injury_count": [2], "event": [0]})
    res = pr.validate_lever_observationally(df, "prior_injury_count")
    assert "error" in res
