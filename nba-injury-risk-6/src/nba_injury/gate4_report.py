"""Stage 4 — GATE 4 check.

Gate 4 (from BUILD_PLAN): the model trains, temporal validation runs cleanly,
and performance is MEANINGFULLY BETTER than a sensible baseline (league-average
hazard, or age+minutes-only). It does NOT need to be great yet — it needs to
show the features carry *some* signal beyond the trivial. If even the simple
model shows zero lift over baseline, pause and investigate before adding depth.

This runs the Stage-4 models and evaluates the gate condition. Reads only the
local person-period table (no network).

Run:  python -m nba_injury.gate4_report
"""
from __future__ import annotations

import sys

from nba_injury.model_hazard import run

# Gate threshold: the best real model must beat the minimal baseline's AUC-PR
# by a margin, AND beat the constant league-average-hazard predictor.
MIN_LIFT_VS_CONSTANT = 1.10   # >=10% better average-precision than prevalence
MIN_REL_GAIN_VS_MINIMAL = 0.05  # >=5% better AUC-PR than age+minutes-only


def main() -> int:
    out = run()
    models = {m["model"]: m for m in out["models"]}

    minimal = models.get("minimal_baseline", {})
    best_real = max(
        (m for k, m in models.items() if k != "minimal_baseline"),
        key=lambda m: (m["auc_pr"] if m["auc_pr"] == m["auc_pr"] else -1),
        default=None,
    )

    print("\n" + "=" * 70)
    print("GATE 4 — SIGNAL ABOVE BASELINE")
    print("=" * 70)

    if best_real is None or best_real["auc_pr"] != best_real["auc_pr"]:
        print("[FAIL] no valid AUC-PR (likely zero events in the test split — "
              "fixture too small). Re-run on real data.")
        print("=" * 70)
        return 2

    lift_const = best_real["auc_pr_lift_vs_constant"]
    rel_gain = (
        (best_real["auc_pr"] - minimal["auc_pr"]) / minimal["auc_pr"]
        if minimal.get("auc_pr") else float("nan")
    )

    cond_const = lift_const >= MIN_LIFT_VS_CONSTANT
    cond_minimal = rel_gain >= MIN_REL_GAIN_VS_MINIMAL

    print(f"best model ................. {best_real['model']}")
    print(f"AUC-PR ..................... {best_real['auc_pr']:.4f}")
    print(f"test prevalence (constant) . {best_real['test_pos_rate']:.4f}")
    print(f"[{'PASS' if cond_const else 'FAIL'}] lift vs constant hazard "
          f">= {MIN_LIFT_VS_CONSTANT:.2f}  (got {lift_const:.2f})")
    print(f"[{'PASS' if cond_minimal else 'FAIL'}] rel. AUC-PR gain vs "
          f"age+minutes baseline >= {MIN_REL_GAIN_VS_MINIMAL:.0%}  "
          f"(got {rel_gain:.0%})")

    passed = cond_const and cond_minimal
    print("\n" + ("GATE 4 PASSED ✓ — features carry signal. Proceed to Stage 5."
                  if passed else
                  "GATE 4 NOT PASSED — investigate features/labels before depth.\n"
                  "(On a tiny synthetic fixture this is expected; the gate is\n"
                  "meaningful only on the real person-period table.)"))
    print("=" * 70)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
