"""Tests for Stage-6 competing risks + recurrent events.

Pin the LOGIC (not the noisy fixture's exact metrics):
  - cumulative_incidence is coherent: CIF_injury + CIF_exit + S == 1, monotone
  - cause coding assigns injury/exit/censored correctly
  - the recurrent table adds a causal weeks_since_last_injury
  - the PyDTS wrapper never raises (returns a status dict)
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_injury import model_competing_risks as cr  # noqa: E402


def test_cif_is_coherent_and_sums_to_one():
    # constant small hazards over a horizon -> CIFs + survival == 1
    h_i = np.full(100, 0.02)
    h_e = np.full(100, 0.01)
    out = cr.cumulative_incidence(h_i, h_e, horizon_weeks=26)
    total = out["cif_injury_final"] + out["cif_exit_final"] + out["survival_final"]
    assert abs(total - 1.0) < 1e-9
    assert 0 <= out["cif_injury_final"] <= 1
    assert 0 <= out["survival_final"] <= 1
    # injury hazard double the exit hazard -> injury CIF should exceed exit CIF
    assert out["cif_injury_final"] > out["cif_exit_final"]


def test_cif_zero_hazard_means_full_survival():
    out = cr.cumulative_incidence(np.zeros(10), np.zeros(10), horizon_weeks=26)
    assert out["survival_final"] == 1.0
    assert out["cif_injury_final"] == 0.0


def test_cif_longer_horizon_increases_incidence():
    h_i = np.full(50, 0.03)
    h_e = np.full(50, 0.02)
    short = cr.cumulative_incidence(h_i, h_e, horizon_weeks=5)
    long = cr.cumulative_incidence(h_i, h_e, horizon_weeks=40)
    assert long["cif_injury_final"] > short["cif_injury_final"]
    assert long["survival_final"] < short["survival_final"]


def _toy_table():
    rows = []
    w = date(2024, 11, 4)
    # player 1: an injury week then continues
    for i in range(6):
        rows.append({"player_id": 1, "season": "2024-25", "week_start": w,
                     "event": 1 if i == 2 else 0, "at_risk": 1,
                     "exit_type": "" if i < 5 else "active_end"})
        w += timedelta(days=7)
    # player 2: non-injury exit on the terminal week
    w = date(2024, 11, 4)
    for i in range(4):
        rows.append({"player_id": 2, "season": "2024-25", "week_start": w,
                     "event": 0, "at_risk": 1,
                     "exit_type": "" if i < 3 else "exit"})
        w += timedelta(days=7)
    return pd.DataFrame(rows)


def test_code_causes_assigns_injury_and_exit():
    df = cr.code_causes(_toy_table())
    p1 = df[df.player_id == 1].sort_values("week_start")
    p2 = df[df.player_id == 2].sort_values("week_start")
    # player 1's 3rd week is an injury cause
    assert p1.iloc[2]["cause"] == cr.CAUSE_INJURY
    # player 2's terminal week is a non-injury exit cause
    assert p2.iloc[-1]["cause"] == cr.CAUSE_EXIT
    # everything else is censored (0)
    assert (p1.iloc[[0, 1, 3, 4, 5]]["cause"] == 0).all()


def test_recurrent_table_adds_weeks_since_last_injury():
    df = cr.build_recurrent_table(_toy_table())
    assert "weeks_since_last_injury" in df.columns
    p1 = df[df.player_id == 1].sort_values("week_start").reset_index(drop=True)
    # before any injury -> NaN; after the week-2 injury, the gap grows
    assert np.isnan(p1.loc[0, "weeks_since_last_injury"])
    # week index 3 is 1 week after the injury at index 2
    assert p1.loc[3, "weeks_since_last_injury"] == 1.0
    assert p1.loc[4, "weeks_since_last_injury"] == 2.0


def test_pydts_wrapper_never_raises():
    # whatever PyDTS does internally, the wrapper must return a status dict
    train = cr.code_causes(cr.build_recurrent_table(_toy_table()))
    res = cr._try_pydts(train, train, ["weeks_since_last_injury"])
    assert isinstance(res, dict)
    assert res.get("status") in {"fit_ok", "failed", "unavailable"}
