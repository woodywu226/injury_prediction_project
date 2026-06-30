"""Stage 6 — competing risks + recurrent events (the graduate-level depth).

Two limitations of the Stage-4 model this stage removes:

  1. COMPETING RISKS. Stage 4 treats every non-injury week as "censored." But a
     player can also leave the at-risk pool for a NON-injury reason (cut, retire,
     age out). Treating that exit as ordinary censoring biases longevity
     estimates. Competing-risks modeling estimates CAUSE-SPECIFIC hazards:
        - h_injury(t)  = P(injury begins at week t | survived, at risk)
        - h_exit(t)    = P(non-injury career exit at week t | survived, at risk)
     and combines them into a cumulative incidence function (CIF) per cause that
     correctly accounts for the competing event.

  2. RECURRENT EVENTS. Stage 4 is time-to-FIRST-injury. Real careers have many
     injuries, and prior injury changes future risk. We move to a recurrent-event
     framing: every at-risk week is in play (after recovery a player re-enters the
     risk set), with prior-injury burden as a covariate.

Primary implementation = scikit-learn (always runs; the sanity anchor):
  - cause-specific discrete-time hazards = two binary classifiers on the
    person-period table, one per cause, sharing the temporal-validation split.
  - CIF computed from the cause-specific hazards on the test horizon.
Optional cross-check = PyDTS TwoStagesFitter (academic competing-risks package;
  wired in defensively — if its rough edges bite, we log and fall back).

Recurrent events: build_recurrent_table() relaxes the Stage-3 "first injury only"
restriction so post-recovery weeks re-enter the risk set, carrying prior-injury
burden. The Stage-4 models then run on this expanded table for comparison.

Run:
  python -m nba_injury.model_competing_risks            # sklearn anchor + CIF
  python -m nba_injury.model_competing_risks --pydts    # also try PyDTS
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from nba_injury.cache import processed_path
from nba_injury.model_hazard import (
    load_table, temporal_split, TIER1_FEATURES, TIER2_FEATURES,
)

PERSON_PERIOD = "person_period.parquet"

# Cause codes for competing risks (0 = censored / no event this week).
CAUSE_INJURY = 1
CAUSE_EXIT = 2


# ----------------------------------------------------------------------------
# recurrent-event table
# ----------------------------------------------------------------------------
def build_recurrent_table(df: pd.DataFrame) -> pd.DataFrame:
    """Recurrent-event framing: keep ALL at-risk weeks (post-recovery weeks
    re-enter the risk set). The person-period table already encodes at_risk and
    recovery; here we simply DON'T collapse to first-injury. prior_injury_count
    (already causal) carries the recurrent burden.

    The Stage-3 table is already recurrent-ready for at_risk==1 weeks, because
    weeks inside a recovery interval are at_risk==0 and excluded by the model.
    This function exists to make the framing explicit and to add a
    'weeks_since_last_injury' covariate (causal) that recurrent models want.
    """
    df = df.sort_values(["player_id", "week_start"]).reset_index(drop=True)
    weeks_since = np.full(len(df), np.nan)
    last_injury_week: dict[int, object] = {}
    for i, row in enumerate(df.itertuples(index=False)):
        pid = row.player_id
        lw = last_injury_week.get(pid)
        if lw is not None:
            weeks_since[i] = (row.week_start - lw).days / 7.0
        if row.event == 1:
            last_injury_week[pid] = row.week_start
    df = df.copy()
    df["weeks_since_last_injury"] = weeks_since
    return df


# ----------------------------------------------------------------------------
# competing-risks cause coding
# ----------------------------------------------------------------------------
def code_causes(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `cause` column: CAUSE_INJURY on injury-onset weeks, CAUSE_EXIT on a
    player's terminal week when exit_type == 'exit' (non-injury career exit),
    else 0 (no event / censored this week)."""
    df = df.copy()
    cause = np.zeros(len(df), dtype=int)
    cause[df["event"].to_numpy() == 1] = CAUSE_INJURY
    # non-injury exit lands on the terminal week flagged exit_type == 'exit'
    exit_mask = (df["exit_type"].to_numpy() == "exit")
    # don't overwrite an injury week (injury takes precedence if both somehow)
    cause[(exit_mask) & (cause == 0)] = CAUSE_EXIT
    df["cause"] = cause
    return df


# ----------------------------------------------------------------------------
# scikit-learn cause-specific hazards (the anchor)
# ----------------------------------------------------------------------------
def _fit_cause_specific(train, test, cause_code, features):
    """One classifier for a given cause: P(this cause's event at week t).

    For competing-risks CIF we need CALIBRATED probabilities, not just good
    ranking. class_weight='balanced' improves AUC-PR but badly miscalibrates the
    hazards (it inflates them to ~0.5), which makes the per-step CIF accumulation
    explode. So we fit WITHOUT balanced weighting and rely on the natural
    prevalence, giving honest low hazards. (Ranking quality is still reported via
    AUC-PR; the model is used for its probabilities here, not a threshold.)
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

    ytr = (train["cause"].to_numpy() == cause_code).astype(int)
    yte = (test["cause"].to_numpy() == cause_code).astype(int)
    clf = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000),  # no balanced weighting -> calibrated
    )
    clf.fit(train[features].to_numpy(float), ytr)
    p = clf.predict_proba(test[features].to_numpy(float))[:, 1]

    pos = yte.mean()
    ap = average_precision_score(yte, p) if pos > 0 else float("nan")
    try:
        roc = roc_auc_score(yte, p) if 0 < pos < 1 else float("nan")
    except ValueError:
        roc = float("nan")
    brier = brier_score_loss(yte, p)
    return clf, p, {"cause": cause_code, "pos_rate": float(pos),
                    "auc_pr": float(ap), "roc_auc": float(roc),
                    "brier": float(brier)}


def cumulative_incidence(h_injury, h_exit, horizon_weeks: int = 26):
    """Discrete-time cumulative incidence over a fixed horizon.

    A coherent CIF needs a TIME horizon, not an accumulation across all
    heterogeneous test rows. We use the population MEAN per-week cause-specific
    hazard as the representative discrete-time hazard, then propagate it over
    `horizon_weeks` (≈ one active NBA season):

        CIF_cause(T) = sum_{t=1..T} h_cause * S(t-1),  S(t) = prod (1 - h_total)

    This yields CIF_injury + CIF_exit + S(T) ≈ 1, with the competing exit risk
    correctly subtracted from the injury incidence (the Stage-4 bias removed).
    """
    h_i = float(np.clip(np.mean(h_injury), 0, 1)) if len(h_injury) else 0.0
    h_e = float(np.clip(np.mean(h_exit), 0, 1)) if len(h_exit) else 0.0
    h_total = min(1.0, h_i + h_e)

    S = 1.0
    cif_i = 0.0
    cif_e = 0.0
    for _ in range(horizon_weeks):
        cif_i += h_i * S
        cif_e += h_e * S
        S *= (1 - h_total)
    return {"cif_injury_final": cif_i, "cif_exit_final": cif_e,
            "survival_final": S, "horizon_weeks": horizon_weeks,
            "mean_h_injury": h_i, "mean_h_exit": h_e}


def run(use_pydts: bool = False) -> dict:
    df = load_table()
    df = build_recurrent_table(df)
    df = code_causes(df)

    train, test, train_seasons, test_seasons = temporal_split(df)
    feats = TIER1_FEATURES + ["weeks_since_last_injury"]
    feats_hgb = feats + TIER2_FEATURES  # available to the gbm cross-check

    print("=" * 70)
    print("STAGE 6 — COMPETING RISKS + RECURRENT EVENTS")
    print(f"   train {train_seasons} -> test {test_seasons}")
    print(f"   recurrent at-risk weeks: train {len(train)}, test {len(test)}")
    print("=" * 70)

    n_inj = int((train["cause"] == CAUSE_INJURY).sum())
    n_exit = int((train["cause"] == CAUSE_EXIT).sum())
    print(f"train events: injury={n_inj}, non-injury exit={n_exit}")
    if n_inj == 0 or n_exit == 0:
        print("WARNING: a cause has zero training events — competing-risks "
              "estimates will be unstable on this fixture.", file=sys.stderr)

    out = {"train_seasons": train_seasons, "test_seasons": test_seasons}

    # cause-specific hazards (the sklearn anchor)
    _, p_inj, m_inj = _fit_cause_specific(train, test, CAUSE_INJURY, feats)
    _, p_exit, m_exit = _fit_cause_specific(train, test, CAUSE_EXIT, feats)
    out["cause_specific"] = {"injury": m_inj, "exit": m_exit}

    print("\ncause-specific hazards (logistic anchor):")
    for name, m in (("injury", m_inj), ("exit", m_exit)):
        print(f"   {name:<8} AUC-PR={m['auc_pr']:.4f}  ROC-AUC={m['roc_auc']:.4f}"
              f"  Brier={m['brier']:.4f}  prevalence={m['pos_rate']:.4f}")

    cif = cumulative_incidence(p_inj, p_exit)
    out["cif"] = cif
    print(f"\ncumulative incidence over the test horizon:")
    print(f"   CIF(injury)={cif['cif_injury_final']:.3f}  "
          f"CIF(exit)={cif['cif_exit_final']:.3f}  "
          f"survival={cif['survival_final']:.3f}")
    print("   (competing-risks CIFs + survival sum toward 1; the exit cause is no"
          "\n    longer mislabeled as ordinary censoring — the Stage-4 bias removed.)")

    # what the depth buys vs Stage 4: the exit hazard is now modeled, not hidden
    out["delta_vs_stage4"] = {
        "non_injury_exit_modeled": True,
        "recurrent_framing": True,
        "added_covariate": "weeks_since_last_injury",
    }

    if use_pydts:
        out["pydts"] = _try_pydts(train, test, feats)

    print("=" * 70)
    return out


# ----------------------------------------------------------------------------
# optional PyDTS cross-check (defensive — academic package, rough edges)
# ----------------------------------------------------------------------------
def _try_pydts(train, test, features) -> dict:
    """Attempt a PyDTS TwoStagesFitter competing-risks fit. Returns a status
    dict; never raises (the build plan warns about this package's rough edges)."""
    try:
        from pydts.fitters import TwoStagesFitter
    except Exception as e:  # noqa: BLE001
        return {"status": "unavailable", "detail": str(e)}

    try:
        # PyDTS expects a person-period frame with pid, X (discrete time index),
        # J (event type; 0 = censored), and covariates. Adapt our table.
        adapted = _adapt_for_pydts(train, features)
        fitter = TwoStagesFitter()
        fitter.fit(df=adapted, covariates=features,
                   event_type_col="J", duration_col="X", pid_col="pid",
                   verbose=0)
        return {"status": "fit_ok",
                "note": "PyDTS competing-risks fit succeeded as cross-check"}
    except Exception as e:  # noqa: BLE001
        return {"status": "failed",
                "detail": f"{type(e).__name__}: {e}",
                "note": "fell back to the sklearn anchor (expected per build plan)"}


def _adapt_for_pydts(df: pd.DataFrame, features) -> pd.DataFrame:
    """Map our person-period rows to PyDTS's (pid, X, J, covariates) schema.

    X = within-player discrete time step (1-indexed week order). J = cause code
    (0 censored, 1 injury, 2 exit). Each player's series is truncated at their
    first event (PyDTS models time-to-event with competing causes)."""
    rows = []
    for pid, g in df.groupby("player_id"):
        g = g.sort_values("week_start").reset_index(drop=True)
        for t, row in enumerate(g.itertuples(index=False), start=1):
            j = int(row.cause)
            rec = {"pid": int(pid), "X": t, "J": j}
            for f in features:
                rec[f] = getattr(row, f)
            rows.append(rec)
            if j != 0:  # stop at first event for time-to-event framing
                break
    adapted = pd.DataFrame(rows)
    # PyDTS wants no NaNs in covariates; median-fill defensively
    for f in features:
        if adapted[f].isna().any():
            adapted[f] = adapted[f].fillna(adapted[f].median())
    return adapted


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 6 competing risks + recurrent.")
    ap.add_argument("--pydts", action="store_true",
                    help="also attempt the PyDTS competing-risks cross-check")
    args = ap.parse_args()
    run(use_pydts=args.pydts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
