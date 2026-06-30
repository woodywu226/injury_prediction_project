"""Stage 4 — simplest correct model, end to end (discrete-time hazard).

A discrete-time hazard model is just a binary classifier on the person-period
table: for each at-risk player-week, predict P(a time-loss injury BEGINS this
week). We start with the simplest honest models and validate them the only way
that doesn't lie to itself — temporally.

Two models, by design (per BUILD_PLAN):
  1. Logistic regression  — interpretable baseline. Needs imputation + scaling.
  2. HistGradientBoosting — handles Tier-2 NaN missingness natively (no impute).

Validation:
  - STRICT TEMPORAL split: train on earlier seasons, test on later seasons.
    Never a random split — a random split leaks the future into the past and is
    the #1 way these projects flatter themselves.
  - Only AT-RISK weeks enter the model (at_risk == 1). Recovery weeks are not
    part of the time-to-first-injury risk set.

Honest metrics for rare events (NOT accuracy):
  - AUC-PR (average precision) — the headline for imbalanced events.
  - ROC-AUC — secondary.
  - Brier score + calibration curve data — are probabilities trustworthy?
  - Per-injury-type recall breakdown (attached separately in the report).
  - Baselines to beat: league-average-hazard (constant) and age+minutes-only.

Run:
  python -m nba_injury.model_hazard                 # train + temporal eval
  python -m nba_injury.model_hazard --report-json reports/stage4_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from nba_injury.cache import processed_path
from nba_injury.nba_client import SEASONS

PERSON_PERIOD = "person_period.parquet"

# Features available strictly before each week (causal). Tier-2 trk_* may be NaN;
# only HistGradientBoosting consumes those natively.
TIER1_FEATURES = [
    "games_this_week", "minutes_this_week", "back_to_backs_this_week",
    "games_in_7days", "cum_season_minutes", "cum_season_games",
    "usg_pct", "pace", "prior_injury_count",
]
TIER2_FEATURES = [
    "trk_speed_distance", "trk_drives", "trk_defense",
    "trk_rebounding", "trk_possessions",
]
# The "simple honest baseline" feature set the full model must beat.
MINIMAL_FEATURES = ["cum_season_minutes", "games_in_7days"]

TARGET = "event"


@dataclass
class Metrics:
    model: str
    n_train: int
    n_test: int
    train_pos_rate: float
    test_pos_rate: float
    auc_pr: float
    roc_auc: float
    brier: float
    # lift over the constant league-average-hazard baseline (AUC-PR ratio)
    auc_pr_lift_vs_constant: float


def load_table() -> pd.DataFrame:
    path = processed_path(PERSON_PERIOD)
    if not path.exists():
        raise SystemExit(
            f"[stage4] {path} not found. Run Stage 3 (build_person_period) first."
        )
    return pd.read_parquet(path)


def temporal_split(df: pd.DataFrame, n_test_seasons: int = 3):
    """Train on earlier seasons, test on the last `n_test_seasons`.

    Returns (train_df, test_df, train_seasons, test_seasons). Only at-risk weeks
    are kept — those are the rows in the time-to-first-injury risk set.
    """
    at_risk = df[df["at_risk"] == 1].copy()
    seasons_present = [s for s in SEASONS if s in set(at_risk["season"])]
    if len(seasons_present) <= n_test_seasons:
        # tiny fixtures: hold out just the final season
        n_test_seasons = max(1, len(seasons_present) - 1)
    test_seasons = seasons_present[-n_test_seasons:]
    train_seasons = seasons_present[:-n_test_seasons]
    train = at_risk[at_risk["season"].isin(train_seasons)]
    test = at_risk[at_risk["season"].isin(test_seasons)]
    return train, test, train_seasons, test_seasons


def _xy(df: pd.DataFrame, features: list[str]):
    # np.ascontiguousarray guarantees a writable, C-contiguous copy. Some
    # numpy/sklearn combos pass a read-only view into HistGradientBoosting's
    # sliding-window binning, which then raises "assignment destination is
    # read-only"; a contiguous copy sidesteps that without changing results.
    X = np.ascontiguousarray(df[features].to_numpy(dtype=float))
    y = np.ascontiguousarray(df[TARGET].to_numpy(dtype=int))
    return X, y


def _eval(name, y_test, p_test, y_train) -> Metrics:
    from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

    pos = float(np.mean(y_test)) if len(y_test) else 0.0
    # constant league-average-hazard baseline = predict train positive rate
    base_rate = float(np.mean(y_train)) if len(y_train) else 0.0
    const_ap = pos  # AUC-PR of a constant predictor ~= positive prevalence
    ap = average_precision_score(y_test, p_test) if pos > 0 else float("nan")
    try:
        roc = roc_auc_score(y_test, p_test) if 0 < pos < 1 else float("nan")
    except ValueError:
        roc = float("nan")
    brier = brier_score_loss(y_test, p_test)
    lift = (ap / const_ap) if (const_ap and not np.isnan(ap)) else float("nan")
    return Metrics(
        model=name, n_train=len(y_train), n_test=len(y_test),
        train_pos_rate=base_rate, test_pos_rate=pos,
        auc_pr=ap, roc_auc=roc, brier=brier,
        auc_pr_lift_vs_constant=lift,
    )


def fit_logistic(train, test):
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    feats = TIER1_FEATURES  # LR can't take NaN; Tier-1 only (mostly complete)
    Xtr, ytr = _xy(train, feats)
    Xte, yte = _xy(test, feats)
    clf = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return clf, _eval("logistic_tier1", yte, p, ytr), (yte, p)


def fit_hgb(train, test):
    from sklearn.ensemble import HistGradientBoostingClassifier

    feats = TIER1_FEATURES + TIER2_FEATURES  # HGB handles NaN natively
    # Drop any column that is entirely NaN in train: HGB can't bin a 100%-missing
    # feature (it carries no signal anyway). On real data the Tier-2 availability
    # map guarantees partial coverage, so this rarely fires — but it makes the
    # pipeline robust to a fully-absent tracking measure in a small slice.
    usable = [f for f in feats if not train[f].isna().all()]
    Xtr, ytr = _xy(train, usable)
    Xte, yte = _xy(test, usable)
    clf = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300,
        l2_regularization=1.0, class_weight="balanced", random_state=0,
    )
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return clf, _eval("hgb_tier1_2", yte, p, ytr), (yte, p)


def fit_minimal_baseline(train, test):
    """age+minutes-style trivial baseline the real models must beat."""
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression

    Xtr, ytr = _xy(train, MINIMAL_FEATURES)
    Xte, yte = _xy(test, MINIMAL_FEATURES)
    clf = make_pipeline(
        SimpleImputer(strategy="median"),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return clf, _eval("minimal_baseline", yte, p, ytr), (yte, p)


def calibration_points(y_true, p, n_bins=10):
    from sklearn.calibration import calibration_curve
    try:
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="quantile")
        return [{"mean_pred": float(a), "frac_pos": float(b)}
                for a, b in zip(mean_pred, frac_pos)]
    except Exception:
        return []


def run(report_json: str | None = None) -> dict:
    df = load_table()
    train, test, train_seasons, test_seasons = temporal_split(df)
    print(f"[stage4] temporal split: train {train_seasons} ({len(train)} wks) "
          f"-> test {test_seasons} ({len(test)} wks)", file=sys.stderr)
    if train["event"].sum() == 0 or test["event"].sum() == 0:
        print("[stage4] WARNING: a split has zero events — fixture too small "
              "for a meaningful eval (real data will be far larger).",
              file=sys.stderr)

    results = []
    _, m_base, _ = fit_minimal_baseline(train, test)
    _, m_lr, lr_pred = fit_logistic(train, test)
    _, m_hgb, hgb_pred = fit_hgb(train, test)
    results = [m_base, m_lr, m_hgb]

    out = {
        "train_seasons": train_seasons,
        "test_seasons": test_seasons,
        "n_train_weeks": len(train),
        "n_test_weeks": len(test),
        "models": [asdict(m) for m in results],
        "calibration": {
            "logistic_tier1": calibration_points(*lr_pred),
            "hgb_tier1_2": calibration_points(*hgb_pred),
        },
    }

    # console summary
    print("\n" + "=" * 70)
    print("STAGE 4 — DISCRETE-TIME HAZARD (temporal validation)")
    print("=" * 70)
    print(f"{'model':<20}{'AUC-PR':>9}{'ROC-AUC':>9}{'Brier':>9}"
          f"{'lift×base':>11}")
    for m in results:
        print(f"{m.model:<20}{m.auc_pr:>9.4f}{m.roc_auc:>9.4f}"
              f"{m.brier:>9.4f}{m.auc_pr_lift_vs_constant:>11.2f}")
    print("=" * 70)

    if report_json:
        from nba_injury.cache import _PKG_ROOT  # repo root
        target = _PKG_ROOT / report_json
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w") as fh:
            json.dump(out, fh, indent=2)
        print(f"[stage4] wrote metrics -> {target}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 4 discrete-time hazard model.")
    ap.add_argument("--report-json", default=None,
                    help="path (repo-relative) to write metrics JSON, "
                         "e.g. reports/stage4_metrics.json")
    args = ap.parse_args()
    run(report_json=args.report_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
