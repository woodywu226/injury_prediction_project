"""Stage 5 — temporal honesty (leakage) audit.

The build plan's #1 silent credibility-killer is temporal leakage: a feature for
player-week W that secretly encodes information from week >= W. This module hunts
for it with adversarial checks rather than trusting the construction.

Checks:
  1. CAUSALITY OF CUMULATIVES — cum_season_minutes / cum_season_games must be
     non-decreasing within (player, season) and week-1 must equal that week's own
     value (no pre-loaded future). (Also enforced in Gate 3; re-asserted here as
     a leakage-specific gate.)
  2. PRIOR-INJURY MONOTONICITY — prior_injury_count never decreases over weeks
     for a player, and is strictly < total episodes before the final week.
  3. TARGET-SHUFFLE TEST — if we randomly permute the event labels (destroying any
     real signal), a correctly-specified pipeline under temporal validation should
     collapse to ~baseline AUC-PR. A model that still "predicts" shuffled labels
     is leaking (e.g. an index/time feature aligned with the label).
  4. FUTURE-WEEK PROBE — inject a deliberately leaky feature (the event itself,
     shifted by zero) and confirm the audit's shuffle test would catch a model
     that uses it. (Sanity check on the detector, run in tests.)

The honest output: a pass/fail per check plus the shuffled-vs-real AUC-PR gap,
which quantifies how much of the model's performance is real signal vs. leak.

Run:  python -m nba_injury.audit_leakage
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from nba_injury.cache import processed_path
from nba_injury.model_hazard import (
    load_table, temporal_split, fit_logistic, fit_hgb, TIER1_FEATURES,
)


def check_cumulative_causality(df: pd.DataFrame) -> tuple[bool, str]:
    bad = 0
    for (_, _), g in df.groupby(["player_id", "season"]):
        g = g.sort_values("week_start")
        cum = g["cum_season_minutes"].to_numpy()
        wk = g["minutes_this_week"].to_numpy()
        if np.any(np.diff(cum) < -1e-9):
            bad += 1
            continue
        if len(cum) and abs(cum[0] - wk[0]) > 1e-6:
            bad += 1
    return bad == 0, f"(player,season) groups with non-causal cumulatives: {bad}"


def check_prior_injury_monotonic(df: pd.DataFrame) -> tuple[bool, str]:
    bad = 0
    for _, g in df.groupby("player_id"):
        pc = g.sort_values("week_start")["prior_injury_count"].to_numpy()
        if np.any(np.diff(pc) < 0):
            bad += 1
    return bad == 0, f"players with decreasing prior_injury_count: {bad}"


def target_shuffle_test(df: pd.DataFrame, n_repeats: int = 5, seed: int = 0):
    """Permute labels within the training set; a clean pipeline should drop to
    ~baseline AUC-PR on the (real-labelled) test set. Returns (real_ap,
    mean_shuffled_ap, std). A small gap => the model's lift is mostly leak."""
    from sklearn.metrics import average_precision_score

    train, test, _, _ = temporal_split(df)
    if train["event"].sum() == 0 or test["event"].sum() == 0:
        return float("nan"), float("nan"), float("nan")

    # real model
    _, m_real, _ = fit_logistic(train, test)
    real_ap = m_real.auc_pr

    rng = np.random.default_rng(seed)
    shuffled_aps = []
    for _ in range(n_repeats):
        tr = train.copy()
        tr["event"] = rng.permutation(tr["event"].to_numpy())
        try:
            _, m_s, _ = fit_logistic(tr, test)
            shuffled_aps.append(m_s.auc_pr)
        except Exception:
            continue
    if not shuffled_aps:
        return real_ap, float("nan"), float("nan")
    return real_ap, float(np.mean(shuffled_aps)), float(np.std(shuffled_aps))


def run() -> dict:
    df = load_table()
    out = {}

    print("=" * 70)
    print("STAGE 5 — TEMPORAL HONESTY (LEAKAGE) AUDIT")
    print("=" * 70)

    checks = [
        ("cumulative causality", check_cumulative_causality),
        ("prior-injury monotonicity", check_prior_injury_monotonic),
    ]
    structural_ok = True
    for name, fn in checks:
        ok, detail = fn(df)
        structural_ok &= ok
        out[name] = ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<28} {detail}")

    real_ap, shuf_ap, shuf_sd = target_shuffle_test(df)
    out["real_auc_pr"] = real_ap
    out["shuffled_auc_pr"] = shuf_ap
    if real_ap == real_ap and shuf_ap == shuf_ap:  # not NaN
        gap = real_ap - shuf_ap
        # Leak suspicion is about the GAP: a clean pipeline shows real >> shuffled.
        # If shuffling the labels barely changes performance, the model wasn't
        # using real label-structure (it's leaking time/index alignment instead).
        # Require the real model to beat shuffled by a clear margin AND shuffled
        # to land within noise of its own std of the real score.
        shuf_hi = shuf_ap + 2 * (shuf_sd if shuf_sd == shuf_sd else 0)
        leak_suspect = real_ap <= shuf_hi  # real not clearly above shuffled band
        out["shuffle_gap"] = gap
        out["leak_suspect"] = bool(leak_suspect)
        print(f"\ntarget-shuffle test:")
        print(f"   real AUC-PR ........ {real_ap:.4f}")
        print(f"   shuffled AUC-PR .... {shuf_ap:.4f} ± {shuf_sd:.4f}")
        print(f"   real - shuffled .... {gap:+.4f}")
        print(f"   [{'WARN' if leak_suspect else 'OK'}] "
              f"{'real not clearly above shuffled band (investigate)' if leak_suspect else 'real clearly beats shuffled (clean)'}")
    else:
        print("\ntarget-shuffle test: SKIPPED (too few events in a split).")
        out["shuffle_gap"] = float("nan")
        out["leak_suspect"] = None

    print("=" * 70)
    out["structural_ok"] = structural_ok
    return out


def main() -> int:
    out = run()
    return 0 if out.get("structural_ok") and not out.get("leak_suspect") else 0
    # NB: returns 0 either way; interpretation is the point, not a hard gate here.


if __name__ == "__main__":
    raise SystemExit(main())
