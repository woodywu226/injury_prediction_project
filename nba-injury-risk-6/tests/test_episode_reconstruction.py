"""Test episode reconstruction on a tiny, controlled snapshot.

Verifies the OUT->IN pairing, right-open episodes, load-management filtering,
and ambiguous handling — independent of any real or synthetic data volume.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nba_injury import build_labels  # noqa: E402


def _rows(records):
    """records: list of (date, acquired, relinquished, notes)."""
    return [{"date": d, "team": "XXX", "acquired": a,
             "relinquished": r, "notes": n} for (d, a, r, n) in records]


def test_basic_out_in_pairing():
    rows = _rows([
        ("2020-01-01", "", "• Player A", "sprained left ankle"),
        ("2020-01-15", "• Player A", "", "returned to lineup"),
    ])
    eps, amb, n_out, total = build_labels.reconstruct_episodes(rows)
    assert len(eps) == 1
    e = eps[0]
    assert e.category == "lower_limb_soft_tissue"
    assert e.start_date == "2020-01-01"
    assert e.end_date == "2020-01-15"
    assert e.days_out == 14


def test_right_open_episode_when_never_returns():
    rows = _rows([
        ("2020-03-01", "", "• Player B", "torn Achilles (out for season)"),
    ])
    eps, *_ = build_labels.reconstruct_episodes(rows)
    assert len(eps) == 1
    assert eps[0].end_date is None
    assert eps[0].days_out is None
    assert eps[0].severe_tail is True


def test_load_management_not_counted():
    rows = _rows([
        ("2020-02-01", "", "• Player C", "rest (DNP)"),
        ("2020-02-02", "• Player C", "", "returned to lineup"),
    ])
    eps, amb, n_out, total = build_labels.reconstruct_episodes(rows)
    assert len(eps) == 0  # rest is not an injury episode


def test_ambiguous_counted_separately():
    rows = _rows([
        ("2020-02-01", "", "• Player D", "roster move xyz unspecified"),
    ])
    eps, amb, n_out, total = build_labels.reconstruct_episodes(rows)
    assert len(eps) == 0
    assert sum(amb.values()) == 1


def test_two_consecutive_outs_closes_first_as_right_open():
    rows = _rows([
        ("2020-01-01", "", "• Player E", "sprained ankle"),
        ("2020-02-01", "", "• Player E", "lower back spasms"),  # new out, no in
        ("2020-02-20", "• Player E", "", "returned to lineup"),
    ])
    eps, *_ = build_labels.reconstruct_episodes(rows)
    assert len(eps) == 2
    cats = {e.category for e in eps}
    assert cats == {"lower_limb_soft_tissue", "back"}
    # first (ankle) is right-open, second (back) closes
    ankle = next(e for e in eps if e.category == "lower_limb_soft_tissue")
    back = next(e for e in eps if e.category == "back")
    assert ankle.end_date is None
    assert back.end_date == "2020-02-20"
