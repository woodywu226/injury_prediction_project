"""Stage 8 — monitoring / assurance layer (the clinical-governance signal).

The whole point (vision §17, §74): distinguish HARMLESS covariate drift from
PERFORMANCE-BREAKING drift, and refuse to over-trust sparse signal. A risk model
that silently degrades is worse than no model; this layer keeps it honest over
time.

Five capabilities, implemented natively (dependency-light, fully reproducible —
the build plan warns "don't let Evidently dashboards become the project"):

  1. COVARIATE DRIFT — PSI (population stability index) + KS per feature
     (univariate) plus a simple multivariate drift score (mean standardized
     shift). Reference = training-era distribution; analysis = a later window.

  2. SUBGROUP PERFORMANCE — AUC-PR / event-rate per subgroup (era, position-ish,
     age band where bio is joined, injury type) to catch localized degradation a
     global metric hides.

  3. CALIBRATION DRIFT — Brier + reliability gap per time window; flags when the
     model's probabilities stop matching reality.

  4. LATE-LABEL TWO-MODE — injuries are confirmed only after games are missed, so
     labels lag. Mode A (NannyML-style): ESTIMATE performance before labels using
     the model's own confidence (DLE-style expected-Brier). Mode B: retrospective
     CONFIRMATION once labels arrive. The gap between A and B is itself monitored.

  5. VERIFIABLE-EVENT DEMO — run the monitor across the timeline and show it
     reacting to the COVID seasons (2019-20 bubble, 2020-21 compressed), then
     check drift-vs-degradation honestly: did the schedule drift, and did the
     model actually get worse, or just see different inputs?

Optional Evidently cross-check is wired in defensively (degrades gracefully).

Run:
  python -m nba_injury.monitoring                 # full assurance pass
  python -m nba_injury.monitoring --evidently      # also try Evidently
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

# COVID-disrupted seasons sit inside the 10-season window — the drift demo anchor.
COVID_SEASONS = ["2019-20", "2020-21"]


# ----------------------------------------------------------------------------
# 1. covariate drift — PSI + KS
# ----------------------------------------------------------------------------
def psi(reference: np.ndarray, analysis: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. <0.1 negligible, 0.1-0.25 moderate, >0.25
    significant drift. Robust to empty bins via a small epsilon."""
    ref = reference[~np.isnan(reference)]
    ana = analysis[~np.isnan(analysis)]
    if len(ref) < 10 or len(ana) < 10:
        return float("nan")
    # quantile edges from the reference, then EXTEND to cover the analysis range
    # so shifted/disjoint analysis values still fall inside a bin (avoids NaN).
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    lo = min(edges[0], float(np.min(ana)))
    hi = max(edges[-1], float(np.max(ana)))
    edges = np.concatenate(([lo - 1e-9], edges[1:-1], [hi + 1e-9]))
    ref_hist, _ = np.histogram(ref, bins=edges)
    ana_hist, _ = np.histogram(ana, bins=edges)
    ref_pct = ref_hist / ref_hist.sum() + 1e-6
    ana_pct = ana_hist / ana_hist.sum() + 1e-6
    return float(np.sum((ana_pct - ref_pct) * np.log(ana_pct / ref_pct)))


def ks_stat(reference: np.ndarray, analysis: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic (max CDF gap). Dependency-free."""
    ref = np.sort(reference[~np.isnan(reference)])
    ana = np.sort(analysis[~np.isnan(analysis)])
    if len(ref) < 10 or len(ana) < 10:
        return float("nan")
    grid = np.concatenate([ref, ana])
    cdf_ref = np.searchsorted(ref, grid, side="right") / len(ref)
    cdf_ana = np.searchsorted(ana, grid, side="right") / len(ana)
    return float(np.max(np.abs(cdf_ref - cdf_ana)))


def covariate_drift(ref_df, ana_df, features) -> dict:
    out = {}
    for f in features:
        if f not in ref_df or f not in ana_df:
            continue
        out[f] = {
            "psi": psi(ref_df[f].to_numpy(float), ana_df[f].to_numpy(float)),
            "ks": ks_stat(ref_df[f].to_numpy(float), ana_df[f].to_numpy(float)),
        }
    # multivariate: mean standardized shift across features
    shifts = []
    for f in features:
        if f in ref_df and f in ana_df:
            r, a = ref_df[f].dropna(), ana_df[f].dropna()
            if len(r) > 10 and len(a) > 10 and r.std() > 0:
                shifts.append(abs(a.mean() - r.mean()) / (r.std() + 1e-9))
    out["_multivariate_mean_shift"] = float(np.mean(shifts)) if shifts else float("nan")
    return out


# ----------------------------------------------------------------------------
# 2. subgroup performance
# ----------------------------------------------------------------------------
def _ap(y, p):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")


def subgroup_performance(df, p_col="_pred", group_cols=("season",)) -> dict:
    out = {}
    for gc in group_cols:
        if gc not in df:
            continue
        rows = []
        for val, g in df.groupby(gc):
            y = g["event"].to_numpy(int)
            rows.append({
                "group": str(val), "n": len(g), "event_rate": float(y.mean()),
                "auc_pr": _ap(y, g[p_col].to_numpy(float)),
            })
        out[gc] = rows
    return out


# ----------------------------------------------------------------------------
# 3. calibration drift over time
# ----------------------------------------------------------------------------
def calibration_drift(df, p_col="_pred", time_col="season") -> list[dict]:
    from sklearn.metrics import brier_score_loss
    rows = []
    for val, g in df.groupby(time_col):
        y = g["event"].to_numpy(int)
        p = g[p_col].to_numpy(float)
        trivial = y.mean() * (1 - y.mean())
        rows.append({
            "window": str(val), "n": len(g), "event_rate": float(y.mean()),
            "brier": float(brier_score_loss(y, p)) if len(set(y)) > 0 else float("nan"),
            "trivial_brier": float(trivial),
            "reliability_gap": float(p.mean() - y.mean()),  # mean pred - observed
        })
    return rows


# ----------------------------------------------------------------------------
# 4. late-label two-mode (NannyML-style estimate vs retrospective confirm)
# ----------------------------------------------------------------------------
def estimated_vs_actual_performance(df, p_col="_pred", time_col="season") -> list[dict]:
    """Mode A (pre-label): estimate Brier from the model's own probabilities
    (expected Brier = mean p(1-p), the irreducible part under calibration).
    Mode B (post-label): the realized Brier. The gap signals over/under-confidence
    that pure label-lag would hide."""
    from sklearn.metrics import brier_score_loss
    rows = []
    for val, g in df.groupby(time_col):
        p = g[p_col].to_numpy(float)
        y = g["event"].to_numpy(int)
        est_brier = float(np.mean(p * (1 - p)))         # expected, label-free
        act_brier = float(brier_score_loss(y, p)) if len(g) else float("nan")
        rows.append({
            "window": str(val), "n": len(g),
            "estimated_brier_prelabels": est_brier,
            "actual_brier_postlabels": act_brier,
            "estimate_gap": act_brier - est_brier,
        })
    return rows


# ----------------------------------------------------------------------------
# 5. verifiable-event demo: COVID seasons drift vs degradation
# ----------------------------------------------------------------------------
def verifiable_event_demo(df, features, p_col="_pred") -> dict:
    """Reference = pre-COVID seasons; analysis = COVID seasons. Did inputs drift,
    and did the model actually DEGRADE, or just see different inputs?"""
    pre = df[~df["season"].isin(COVID_SEASONS)]
    covid = df[df["season"].isin(COVID_SEASONS)]
    if covid.empty or pre.empty:
        return {"status": "COVID seasons not both present in fixture"}

    drift = covariate_drift(pre, covid, features)
    # performance pre vs during
    pre_ap = _ap(pre["event"].to_numpy(int), pre[p_col].to_numpy(float))
    covid_ap = _ap(covid["event"].to_numpy(int), covid[p_col].to_numpy(float))

    # a feature counts as drifted if EITHER PSI>0.1 OR KS>0.2 (multi-test, like
    # Evidently — no single metric's edge case can silently hide real drift)
    def _drifted(d):
        psi_v, ks_v = d.get("psi"), d.get("ks")
        psi_hit = psi_v == psi_v and psi_v > 0.1
        ks_hit = ks_v == ks_v and ks_v > 0.2
        return psi_hit or ks_hit
    drifted_feats = [f for f in features
                     if f in drift and isinstance(drift[f], dict)
                     and _drifted(drift[f])]
    degraded = (pre_ap == pre_ap and covid_ap == covid_ap and covid_ap < pre_ap * 0.8)

    return {
        "drifted_features": drifted_feats,
        "multivariate_shift": drift.get("_multivariate_mean_shift"),
        "auc_pr_pre_covid": pre_ap,
        "auc_pr_during_covid": covid_ap,
        "verdict": (
            "DRIFT WITHOUT DEGRADATION — inputs shifted (schedule density) but "
            "the model held up; harmless drift, no retrain triggered"
            if drifted_feats and not degraded else
            "DRIFT WITH DEGRADATION — inputs shifted AND performance dropped; "
            "this is the retrain/with-care signal"
            if drifted_feats and degraded else
            "no significant covariate drift detected across COVID seasons"
        ),
    }


# ----------------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------------
def _score(df, features):
    """Train on early seasons, score ALL at-risk weeks for monitoring.

    Produces TWO scores:
      _pred       — balanced-weight model (good ranking; used for drift/subgroup
                    AUC-PR where ranking is what matters).
      _pred_cal   — UNWEIGHTED model (calibrated probabilities; used for the
                    calibration-drift and late-label panels, which need honest
                    probabilities — the same balanced-vs-calibrated tradeoff
                    identified in Stage 6).
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    at_risk = df[df["at_risk"] == 1].copy()
    train, _, train_seasons, _ = temporal_split(df)
    ytr = train["event"].to_numpy(int)
    Xtr = train[features].to_numpy(float)

    rank = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    rank.fit(Xtr, ytr)
    at_risk["_pred"] = rank.predict_proba(at_risk[features].to_numpy(float))[:, 1]

    cal = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        LogisticRegression(max_iter=2000))  # unweighted -> calibrated
    cal.fit(Xtr, ytr)
    at_risk["_pred_cal"] = cal.predict_proba(at_risk[features].to_numpy(float))[:, 1]
    return at_risk, train_seasons


def run(use_evidently: bool = False) -> dict:
    df = load_table()
    features = TIER1_FEATURES + TIER2_FEATURES
    scored, train_seasons = _score(df, features)

    print("=" * 70)
    print("STAGE 8 — MONITORING / ASSURANCE")
    print("=" * 70)

    # reference = training era; analysis = post-training era
    ref = scored[scored["season"].isin(train_seasons)]
    ana = scored[~scored["season"].isin(train_seasons)]

    drift = covariate_drift(ref, ana, features) if not ana.empty else {}
    subperf = subgroup_performance(scored, group_cols=("season",))
    caldrift = calibration_drift(scored, p_col="_pred_cal")
    latelabel = estimated_vs_actual_performance(scored, p_col="_pred_cal")
    demo = verifiable_event_demo(scored, features)

    print("\n1) COVARIATE DRIFT (reference=train era -> analysis=later):")
    sig = [(f, d["psi"]) for f, d in drift.items()
           if f != "_multivariate_mean_shift" and d.get("psi") == d.get("psi")]
    for f, v in sorted(sig, key=lambda x: -x[1])[:6]:
        tag = "SIGNIFICANT" if v > 0.25 else "moderate" if v > 0.1 else "negligible"
        print(f"   {f:<26} PSI={v:.3f}  [{tag}]")
    print(f"   multivariate mean shift: {drift.get('_multivariate_mean_shift', float('nan')):.3f}")

    print("\n3) CALIBRATION DRIFT (Brier vs trivial, reliability gap):")
    for r in caldrift:
        flag = "" if r["brier"] != r["brier"] or r["brier"] <= r["trivial_brier"] else "  <-- worse than trivial"
        print(f"   {r['window']}: Brier={r['brier']:.4f} (trivial {r['trivial_brier']:.4f}) "
              f"relgap={r['reliability_gap']:+.4f}{flag}")

    print("\n4) LATE-LABEL (estimated pre-labels vs actual post-labels Brier):")
    for r in latelabel[:4]:
        print(f"   {r['window']}: est={r['estimated_brier_prelabels']:.4f} "
              f"actual={r['actual_brier_postlabels']:.4f} "
              f"gap={r['estimate_gap']:+.4f}")

    print("\n5) VERIFIABLE-EVENT DEMO (COVID seasons drift vs degradation):")
    print(f"   drifted features: {demo.get('drifted_features')}")
    print(f"   AUC-PR pre={demo.get('auc_pr_pre_covid')} "
          f"during={demo.get('auc_pr_during_covid')}")
    print(f"   VERDICT: {demo.get('verdict')}")

    out = {"drift": drift, "subgroup": subperf, "calibration_drift": caldrift,
           "late_label": latelabel, "verifiable_demo": demo}

    if use_evidently:
        out["evidently"] = _try_evidently(ref, ana, features)
        print(f"\nEvidently cross-check: {out['evidently'].get('status')}")

    print("=" * 70)
    return out


def _try_evidently(ref, ana, features) -> dict:
    """Optional Evidently drift report; never raises (version-fragile dep)."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except Exception as e:  # noqa: BLE001
        return {"status": "unavailable", "detail": str(e)}
    try:
        cols = [f for f in features if f in ref and f in ana]
        rep = Report(metrics=[DataDriftPreset()])
        rep.run(reference_data=ref[cols], current_data=ana[cols])
        return {"status": "ok", "note": "Evidently drift report generated"}
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "detail": f"{type(e).__name__}: {e}",
                "note": "native PSI/KS is the primary path (expected fallback)"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 8 monitoring / assurance.")
    ap.add_argument("--evidently", action="store_true",
                    help="also attempt the optional Evidently cross-check")
    args = ap.parse_args()
    run(use_evidently=args.evidently)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
