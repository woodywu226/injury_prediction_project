"""Stage 7 — the prescriptive / causal core (the heart of the project).

This is the highest-overclaim-risk stage. Causal humility is non-negotiable AND
the credibility strength. Three jobs, each with a hard boundary:

  JOB 1 — INTERPRETATION (descriptive, solid ground).
    SHAP explains WHY THE MODEL assigns a player's risk — "what the model
    attributes risk to," NEVER "what causes injury in the world." No world-claim
    beyond the model.

  JOB 3 — MODIFIABLE / NON-MODIFIABLE PARTITION (what makes it prescriptive).
    Features split into ACTIONABLE (minutes, games-in-7, back-to-backs, usage,
    drive-load, rest patterns) vs FIXED (age, height, mass, position). Levers are
    surfaced ONLY on the actionable subset; fixed features are context. This is
    also where the public-data ceiling is honored: only in-game style/load levers
    are prescribable — NOT training/recovery (not in the data).

  JOB 2 — COUNTERFACTUAL HAZARD (the dangerous leap; explicit humility).
    Alter ONE modifiable feature, hold the rest fixed, read the model's predicted
    hazard delta. ALWAYS framed as "predicted risk IF this feature differed, all
    else equal, under learned associations" — never a promise the intervention
    works. Strengthened with a propensity-style comparison of similar-profile
    players who actually differed on the lever.

  VALIDATION — does a model-implied lever track LOWER REALIZED injury among
    comparable-profile players who actually had lower lever values? If yes →
    observational support; if no → report honestly. This turns "SHAP + a
    recommendation" into "a hypothesis with observational support and stated
    limits."

Every prescriptive output carries the STANDING_CAVEAT. The audience is team
performance / sports-science staff: outputs SURFACE HYPOTHESES for expert
judgment, never autonomous advice.

Run:
  python -m nba_injury.prescriptive                 # full prescriptive pass
  python -m nba_injury.prescriptive --player <id>   # one player's levers
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

# The standing caveat wired into EVERY prescriptive output.
STANDING_CAVEAT = (
    "MODEL-IMPLIED HYPOTHESIS, NOT ADVICE. This surfaces associations the model "
    "considers actionable, for trained performance/sports-science staff to weigh "
    "against everything the model cannot see (private biomechanical, training, "
    "and medical data). SHAP attributions explain the MODEL, not real-world "
    "causation. Counterfactuals are 'all-else-equal under learned associations', "
    "not promises that an intervention reduces injury."
)

# JOB 3: the modifiable / non-modifiable partition.
# ACTIONABLE = in-game style/load levers staff can influence via rotation/usage.
MODIFIABLE_FEATURES = [
    "minutes_this_week", "games_this_week", "games_in_7days",
    "back_to_backs_this_week", "usg_pct", "cum_season_minutes",
    "trk_drives", "trk_speed_distance",
]
# FIXED = context only; never surfaced as a lever.
NON_MODIFIABLE_FEATURES = [
    "prior_injury_count",   # history: context, not a lever (can't un-injure)
    "pace",                 # team-level, not a per-player rest lever
    "trk_defense", "trk_rebounding", "trk_possessions",
]
# (age/height/mass live in the bbref bio join; included as fixed context when
#  present. They are deliberately NOT in the lever set.)


def _fit_model(train, features):
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    clf = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    clf.fit(train[features].to_numpy(float), train["event"].to_numpy(int))
    return clf


# ----------------------------------------------------------------------------
# JOB 1 — SHAP interpretation (descriptive)
# ----------------------------------------------------------------------------
def shap_attributions(clf, X: np.ndarray, features: list[str]) -> dict:
    """Mean |SHAP| per feature = "what the model attributes risk to." Falls back
    to standardized logistic coefficients if SHAP is unavailable (same ranking
    for a linear model; SHAP adds per-instance detail, not a different story)."""
    try:
        import shap
        # LinearExplainer is exact + fast for the logistic pipeline. We explain
        # the final estimator on the transformed features.
        pre = clf[:-1]
        est = clf[-1]
        Xt = pre.transform(X)
        explainer = shap.LinearExplainer(est, Xt)
        sv = explainer.shap_values(Xt)
        mean_abs = np.abs(sv).mean(axis=0)
        method = "shap_linear"
    except Exception as e:  # noqa: BLE001
        # transparent fallback: |standardized coefficient|
        est = clf[-1]
        mean_abs = np.abs(est.coef_[0])
        method = f"coef_fallback ({type(e).__name__})"
    order = np.argsort(mean_abs)[::-1]
    return {
        "method": method,
        "ranking": [(features[i], float(mean_abs[i])) for i in order],
    }


# ----------------------------------------------------------------------------
# JOB 2 — counterfactual hazard (model-implied, all-else-equal)
# ----------------------------------------------------------------------------
def counterfactual_hazard(clf, row: pd.Series, features: list[str],
                          lever: str, new_value: float) -> dict:
    """Predicted hazard if `lever` were `new_value`, all else equal.

    HARD GUARD: refuses to counterfactual a non-modifiable feature — surfacing a
    lever on a fixed attribute (age, prior injuries) would be an overclaim by
    construction.
    """
    if lever not in MODIFIABLE_FEATURES:
        return {"error": f"'{lever}' is non-modifiable; no lever surfaced "
                         f"(would be an overclaim)."}
    base = row[features].to_numpy(float).reshape(1, -1)
    p_base = float(clf.predict_proba(base)[0, 1])
    cf = base.copy()
    cf[0, features.index(lever)] = new_value
    p_cf = float(clf.predict_proba(cf)[0, 1])
    return {
        "lever": lever,
        "from_value": float(row[lever]),
        "to_value": float(new_value),
        "hazard_base": p_base,
        "hazard_counterfactual": p_cf,
        "hazard_delta": p_cf - p_base,
        "framing": ("predicted hazard IF this feature differed, all else equal, "
                    "under learned associations — NOT a promise the intervention "
                    "reduces injury"),
    }


# ----------------------------------------------------------------------------
# Observational validation of a lever (propensity-style)
# ----------------------------------------------------------------------------
def validate_lever_observationally(df: pd.DataFrame, lever: str,
                                   n_strata: int = 5) -> dict:
    """Do comparable-profile players with LOWER lever values actually show LOWER
    realized injury? Stratify by a coarse style profile, split each stratum at
    the lever median, compare realized event rates. Honest either way.

    This is observational, confounded, and we SAY SO — it's supporting evidence
    for a hypothesis, not a causal demonstration.
    """
    if lever not in df.columns or lever not in MODIFIABLE_FEATURES:
        return {"error": f"'{lever}' not a validatable modifiable lever"}
    at_risk = df[df["at_risk"] == 1].copy()
    at_risk = at_risk.dropna(subset=[lever])
    if at_risk.empty:
        return {"error": "no at-risk rows with this lever present"}

    # coarse comparable strata by usage + minutes profile (style proxy)
    prof_cols = [c for c in ("usg_pct", "cum_season_minutes") if c in at_risk]
    if prof_cols:
        prof = at_risk[prof_cols].fillna(at_risk[prof_cols].median())
        # quantile bins on the first profile col
        try:
            at_risk["_stratum"] = pd.qcut(prof[prof_cols[0]], q=n_strata,
                                          labels=False, duplicates="drop")
        except ValueError:
            at_risk["_stratum"] = 0
    else:
        at_risk["_stratum"] = 0

    rows = []
    for stratum, g in at_risk.groupby("_stratum"):
        med = g[lever].median()
        low = g[g[lever] <= med]
        high = g[g[lever] > med]
        if len(low) < 20 or len(high) < 20:
            continue
        rows.append({
            "stratum": int(stratum),
            "n_low": len(low), "n_high": len(high),
            "event_rate_low_lever": float(low["event"].mean()),
            "event_rate_high_lever": float(high["event"].mean()),
        })
    if not rows:
        return {"lever": lever, "verdict": "insufficient stratum sizes",
                "strata": []}

    # Across strata: does lower lever -> lower realized injury more often than not?
    supports = sum(1 for r in rows
                   if r["event_rate_low_lever"] < r["event_rate_high_lever"])
    total = len(rows)
    return {
        "lever": lever,
        "strata": rows,
        "strata_supporting_lever": supports,
        "strata_total": total,
        "verdict": (
            "observational support: lower lever tracks lower realized injury in "
            f"{supports}/{total} comparable strata"
            if supports > total / 2 else
            f"NO clear observational support ({supports}/{total} strata); the "
            "model-implied lever does NOT track realized injury here — report as a "
            "limitation, not a recommendation"
        ),
        "caveat": "observational + confounded; supporting evidence only, not causal",
    }


# ----------------------------------------------------------------------------
# per-player prescriptive narrative
# ----------------------------------------------------------------------------
def player_levers(df, clf, features, player_id: int) -> dict:
    at_risk = df[(df["player_id"] == player_id) & (df["at_risk"] == 1)]
    if at_risk.empty:
        return {"player_id": player_id, "error": "no at-risk weeks"}
    # use the player's highest-hazard week as the focus
    p = clf.predict_proba(at_risk[features].to_numpy(float))[:, 1]
    row = at_risk.iloc[int(np.argmax(p))]

    levers = []
    for lever in MODIFIABLE_FEATURES:
        if lever not in features or pd.isna(row[lever]):
            continue
        # counterfactual: reduce the lever by 20% (a plausible rotation change)
        cf = counterfactual_hazard(clf, row, features, lever, row[lever] * 0.8)
        if "error" not in cf:
            levers.append(cf)
    levers.sort(key=lambda c: c["hazard_delta"])  # most risk-reducing first
    return {
        "player_id": player_id,
        "focus_week": str(row["week_start"]),
        "modeled_hazard": float(np.max(p)),
        "top_modifiable_levers": levers[:3],
        "non_modifiable_context": {
            f: (float(row[f]) if f in row and not pd.isna(row[f]) else None)
            for f in NON_MODIFIABLE_FEATURES if f in features
        },
        "CAVEAT": STANDING_CAVEAT,
    }


def run(player_id: int | None = None) -> dict:
    df = load_table()
    features = TIER1_FEATURES + TIER2_FEATURES
    train, test, _, _ = temporal_split(df)
    clf = _fit_model(train, features)

    print("=" * 70)
    print("STAGE 7 — PRESCRIPTIVE / CAUSAL CORE")
    print("=" * 70)
    print("CAVEAT:", STANDING_CAVEAT)

    # JOB 1
    Xtr = train[features].to_numpy(float)
    attr = shap_attributions(clf, Xtr, features)
    print(f"\nJOB 1 — what the MODEL attributes risk to ({attr['method']}):")
    for f, v in attr["ranking"][:8]:
        tag = "actionable" if f in MODIFIABLE_FEATURES else "fixed/context"
        print(f"   {f:<26} {v:.4f}   [{tag}]")

    # JOB 3 — partition is structural; echo it
    print("\nJOB 3 — lever partition:")
    print(f"   modifiable (lever-able): {MODIFIABLE_FEATURES}")
    print(f"   fixed (context only):    {NON_MODIFIABLE_FEATURES}")

    # Validation of the top modifiable lever
    top_mod = next((f for f, _ in attr["ranking"] if f in MODIFIABLE_FEATURES),
                   None)
    val = validate_lever_observationally(df, top_mod) if top_mod else {}
    if val:
        print(f"\nVALIDATION of top modifiable lever '{top_mod}':")
        print(f"   {val.get('verdict')}")

    out = {"attributions": attr, "top_modifiable_lever": top_mod,
           "lever_validation": val, "caveat": STANDING_CAVEAT}

    # per-player narrative
    if player_id is None:
        # pick an injured player with the most at-risk weeks as a demo
        injured = df[df["event"] == 1]["player_id"]
        player_id = int(injured.iloc[0]) if not injured.empty else \
            int(df["player_id"].iloc[0])
    pl = player_levers(df, clf, features, player_id)
    out["player"] = pl
    print(f"\nPER-PLAYER LEVERS — player {player_id} "
          f"(week {pl.get('focus_week')}, hazard {pl.get('modeled_hazard', float('nan')):.4f}):")
    for cf in pl.get("top_modifiable_levers", []):
        print(f"   {cf['lever']:<22} {cf['from_value']:.1f} -> {cf['to_value']:.1f}"
              f"   Δhazard {cf['hazard_delta']:+.4f}")
    print("\n   (every lever above is a MODEL-IMPLIED HYPOTHESIS for staff review)")
    print("=" * 70)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 7 prescriptive core.")
    ap.add_argument("--player", type=int, default=None, help="player_id focus")
    args = ap.parse_args()
    run(player_id=args.player)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
