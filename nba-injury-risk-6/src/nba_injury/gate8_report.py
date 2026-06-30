"""Stage 8 — GATE 8 report.

Gate 8 (from BUILD_PLAN): the monitor distinguishes HARMLESS drift from
PERFORMANCE-BREAKING drift on at least one verifiable historical event, with a
clear narrative ("the schedule drifted here; the model did/didn't degrade, and
here's how we know").

Checks:
  - drift detection produces finite PSI/KS for the monitored features,
  - calibration drift is tracked per window,
  - the late-label two-mode design yields an estimate AND an actual per window,
  - the verifiable-event (COVID) demo renders a clear drift-vs-degradation verdict.

Run:  python -m nba_injury.gate8_report
"""
from __future__ import annotations

import sys

from nba_injury.monitoring import run, COVID_SEASONS


def main() -> int:
    out = run(use_evidently=False)

    print("\n" + "=" * 70)
    print("GATE 8 — MONITORING / ASSURANCE")
    print("=" * 70)

    checks = []

    # 1) drift produced finite metrics for at least some features
    drift = out.get("drift", {})
    finite_psi = [f for f, d in drift.items()
                  if f != "_multivariate_mean_shift"
                  and isinstance(d, dict) and d.get("psi") == d.get("psi")]
    checks.append(("covariate drift computed (PSI/KS finite)", len(finite_psi) > 0))

    # 2) calibration drift tracked per window
    cal = out.get("calibration_drift", [])
    checks.append(("calibration drift tracked per time window", len(cal) >= 2))

    # 3) late-label two-mode produced estimate + actual
    ll = out.get("late_label", [])
    ll_ok = len(ll) >= 2 and all("estimated_brier_prelabels" in r and
                                 "actual_brier_postlabels" in r for r in ll)
    checks.append(("late-label: estimated (pre) AND actual (post) per window", ll_ok))

    # 4) verifiable-event demo renders a clear verdict
    demo = out.get("verifiable_demo", {})
    verdict = demo.get("verdict", "")
    demo_ok = bool(verdict) and "status" not in demo  # demo actually ran
    checks.append(("verifiable-event (COVID) demo renders a verdict", demo_ok))

    all_ok = True
    for name, ok in checks:
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    print("\n--- the Gate-8 narrative ---")
    if demo_ok:
        print(f"Across the COVID seasons ({', '.join(COVID_SEASONS)}):")
        print(f"   drifted inputs : {demo.get('drifted_features')}")
        print(f"   AUC-PR pre={demo.get('auc_pr_pre_covid')} -> "
              f"during={demo.get('auc_pr_during_covid')}")
        print(f"   {demo.get('verdict')}")
        print("\nThis is the crux: the monitor SEPARATES 'inputs looked different'")
        print("from 'the model got worse' — drift is not automatically failure.")
    else:
        print("(COVID seasons not both present in this fixture; on the real")
        print(" 10-season table the demo anchors on 2019-20 / 2020-21.)")

    print("\n" + ("GATE 8 PASSED ✓ — the assurance layer distinguishes harmless\n"
                  "drift from degradation with a clear narrative. Proceed to Stage 9."
                  if all_ok else
                  "GATE 8 NOT PASSED — a monitoring capability is missing.\n"
                  "(On a small fixture the COVID demo may lack data; meaningful on"
                  " the real table.)"))
    print("=" * 70)
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
