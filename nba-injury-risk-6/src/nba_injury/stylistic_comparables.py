"""Stage 5 — stylistic-comparables check (the hindsight-bias guard).

The trap (vision §7): it's trivial to build a model that "would have predicted
Derrick Rose" because the outcome is known. The real test is whether the model
flags an index case using ONLY pre-injury data WITHOUT also flagging the many
stylistically-similar players who stayed healthy.

Method:
  1. Define an index case by its PRE-INJURY style profile (a feature vector:
     workload density, usage, athletic-load proxies, mass/position if available).
     For real data these are Rose (c. 2011) and Zion (current); the function
     takes any (player_id, as-of week) so it's reusable.
  2. Build the "stylistic comparables who stayed healthy" cohort: players whose
     pre-period style vector is nearest (standardized Euclidean / cosine) to the
     index case but who had NO time-loss injury in the comparison horizon.
  3. Score everyone with the temporally-trained model using ONLY pre-injury-week
     features. Ask: does the index case's modeled hazard sit ABOVE its healthy
     look-alikes? Quantify with:
        - index-case percentile within the comparable cohort's hazard distribution
        - AUC of (model hazard) separating the injured index group from the
          healthy comparables (across multiple index cases, if given)
  4. Report honestly: if the model only separates them by raw style (i.e. the
     comparables score just as high), that's the hindsight-bias failure and we
     SAY SO.

This module is data-shape-driven, not hardcoded to specific stars, so it runs on
synthetic worlds in dev and on the real table in production. Supply real index
players via --index-player on the CLI once the real table exists.

Run:  python -m nba_injury.stylistic_comparables
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from nba_injury.model_hazard import (
    load_table, temporal_split, TIER1_FEATURES, TIER2_FEATURES,
)

# Style = the slowly-varying "what kind of player" signal, distinct from the
# acute weekly load. We average these over a player's pre-period to get a profile.
STYLE_FEATURES = [
    "usg_pct", "pace", "minutes_this_week", "games_in_7days",
    "back_to_backs_this_week", "cum_season_minutes",
]


def _player_style_profile(df: pd.DataFrame, player_id: int,
                          before_week=None) -> np.ndarray | None:
    """Mean style vector for a player using only weeks strictly before
    `before_week` (or all weeks if None)."""
    g = df[df["player_id"] == player_id]
    if before_week is not None:
        g = g[g["week_start"] < before_week]
    if g.empty:
        return None
    return g[STYLE_FEATURES].mean(numeric_only=True).to_numpy(dtype=float)


def _had_injury(df: pd.DataFrame, player_id: int) -> bool:
    return bool(df[df["player_id"] == player_id]["event"].sum() > 0)


def build_comparable_cohort(
    df: pd.DataFrame, index_player: int, k: int = 15, before_week=None,
) -> tuple[np.ndarray, list[int]]:
    """Return (index_profile, [healthy comparable player_ids]) — the k nearest
    healthy players to the index case in standardized style space."""
    idx_profile = _player_style_profile(df, index_player, before_week)
    if idx_profile is None:
        raise ValueError(f"no pre-period data for index player {index_player}")

    # candidate pool = all OTHER players with a usable profile
    profiles = {}
    for pid in df["player_id"].unique():
        if pid == index_player:
            continue
        prof = _player_style_profile(df, pid, before_week)
        if prof is not None and not np.any(np.isnan(prof)):
            profiles[pid] = prof
    if not profiles:
        return idx_profile, []

    pid_list = list(profiles)
    P = np.vstack([profiles[p] for p in pid_list])
    # standardize across the candidate pool so no single feature dominates
    mu, sd = P.mean(0), P.std(0) + 1e-9
    Pz = (P - mu) / sd
    iz = (idx_profile - mu) / sd
    d = np.linalg.norm(Pz - iz, axis=1)

    order = np.argsort(d)
    healthy = [pid_list[i] for i in order if not _had_injury(df, pid_list[i])]
    return idx_profile, healthy[:k]


def _model_hazard_for_players(df: pd.DataFrame, player_ids: list[int]) -> dict[int, float]:
    """Train the temporal model, return each player's MAX modeled pre-event
    hazard over their at-risk weeks (peak risk the model ever assigned)."""
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    train, test, _, _ = temporal_split(df)
    feats = TIER1_FEATURES
    clf = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    clf.fit(train[feats].to_numpy(float), train["event"].to_numpy(int))

    at_risk = df[df["at_risk"] == 1]
    out = {}
    for pid in player_ids:
        g = at_risk[at_risk["player_id"] == pid]
        if g.empty:
            out[pid] = float("nan")
            continue
        p = clf.predict_proba(g[feats].to_numpy(float))[:, 1]
        out[pid] = float(np.max(p))
    return out


def evaluate_index_case(df: pd.DataFrame, index_player: int, k: int = 15) -> dict:
    """Core Stage-5 check for one index case."""
    idx_profile, healthy = build_comparable_cohort(df, index_player, k=k)
    if not healthy:
        return {"index_player": index_player, "n_comparables": 0,
                "verdict": "insufficient comparables"}

    hz = _model_hazard_for_players(df, [index_player] + healthy)
    idx_h = hz.get(index_player, float("nan"))
    comp_h = np.array([hz[p] for p in healthy if hz[p] == hz[p]])  # drop NaN

    if np.isnan(idx_h) or comp_h.size == 0:
        return {"index_player": index_player, "n_comparables": len(healthy),
                "verdict": "unscorable (no at-risk weeks)"}

    # percentile of the index case within the healthy comparables' hazards
    pct = float((comp_h < idx_h).mean())
    sep = float(idx_h - np.median(comp_h))
    return {
        "index_player": index_player,
        "index_had_injury": _had_injury(df, index_player),
        "n_comparables": len(healthy),
        "index_max_hazard": idx_h,
        "comparable_median_hazard": float(np.median(comp_h)),
        "comparable_p90_hazard": float(np.percentile(comp_h, 90)),
        "index_percentile_vs_comparables": pct,
        "separation": sep,
        "verdict": (
            "separates (index above healthy look-alikes)" if pct >= 0.75 else
            "WEAK — index not clearly above stylistic comparables "
            "(possible hindsight bias / style-only signal)"
        ),
    }


def pick_default_index_cases(df: pd.DataFrame, n: int = 2) -> list[int]:
    """When no real index player is supplied (e.g. synthetic dev), pick injured
    players with the richest pre-period as stand-ins so the check runs."""
    injured = [p for p in df["player_id"].unique() if _had_injury(df, p)]
    injured.sort(key=lambda p: -len(df[df["player_id"] == p]))
    return injured[:n]


def run(index_players: list[int] | None = None, k: int = 15) -> dict:
    df = load_table()
    if not index_players:
        index_players = pick_default_index_cases(df)
        note = "(synthetic stand-ins; supply --index-player for Rose/Zion on real data)"
    else:
        note = ""

    print("=" * 70)
    print("STAGE 5 — STYLISTIC-COMPARABLES CHECK")
    print(f"   index cases: {index_players} {note}")
    print("=" * 70)

    results = []
    for pid in index_players:
        try:
            r = evaluate_index_case(df, pid, k=k)
        except ValueError as e:
            r = {"index_player": pid, "verdict": f"error: {e}"}
        results.append(r)
        print(f"\nindex player {pid}:")
        for key in ("n_comparables", "index_max_hazard",
                    "comparable_median_hazard", "comparable_p90_hazard",
                    "index_percentile_vs_comparables", "verdict"):
            if key in r:
                v = r[key]
                v = f"{v:.4f}" if isinstance(v, float) else v
                print(f"   {key:<32} {v}")

    return {"index_cases": results}


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 5 stylistic-comparables check.")
    ap.add_argument("--index-player", type=int, action="append", default=None,
                    help="player_id of an index case (repeatable); "
                         "real Rose/Zion ids on the real table")
    ap.add_argument("-k", type=int, default=15, help="comparable cohort size")
    args = ap.parse_args()
    run(index_players=args.index_player, k=args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
