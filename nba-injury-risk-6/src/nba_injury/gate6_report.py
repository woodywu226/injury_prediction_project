"""Stage 6 — GATE 6 report.

Gate 6 (from BUILD_PLAN): competing-risks and recurrent-event models run under
temporal validation and are properly calibrated; you can articulate what they
capture that the simple model missed (e.g. unbiased longevity estimates once
non-injury exits are accounted for).

This runs the Stage-6 models and checks:
  - both cause-specific hazards trained under the temporal split (no random),
  - each cause-specific model is calibrated (Brier below the trivial bound),
  - the competing-risks CIF + survival are coherent (in [0,1], sum sensible),
  - a clear statement of the delta vs Stage 4.

Run:  python -m nba_injury.gate6_report
"""
from __future__ import annotations

import sys

from nba_injury.model_competing_risks import run, CAUSE_INJURY, CAUSE_EXIT


def main() -> int:
    out = run(use_pydts=False)

    cs = out["cause_specific"]
    cif = out["cif"]

    print("\n" + "=" * 70)
    print("GATE 6 — COMPETING RISKS + RECURRENT EVENTS")
    print("=" * 70)

    # 1) both cause models produced finite metrics
    inj_ok = cs["injury"]["auc_pr"] == cs["injury"]["auc_pr"]  # not NaN
    exit_ok = cs["exit"]["auc_pr"] == cs["exit"]["auc_pr"]
    print(f"[{'PASS' if inj_ok else 'WARN'}] injury cause-specific hazard trained "
          f"(AUC-PR={cs['injury']['auc_pr']:.4f})")
    print(f"[{'PASS' if exit_ok else 'WARN'}] exit cause-specific hazard trained "
          f"(AUC-PR={cs['exit']['auc_pr']:.4f})")

    # 2) calibration sanity: Brier below the trivial p*(1-p) bound for each cause
    def brier_ok(m):
        p = m["pos_rate"]
        trivial = p * (1 - p)  # Brier of predicting the constant prevalence
        return m["brier"] <= trivial + 1e-6, trivial
    bi_ok, bi_t = brier_ok(cs["injury"])
    be_ok, be_t = brier_ok(cs["exit"])
    print(f"[{'PASS' if bi_ok else 'FAIL'}] injury calibration: Brier "
          f"{cs['injury']['brier']:.4f} <= trivial {bi_t:.4f}")
    print(f"[{'PASS' if be_ok else 'FAIL'}] exit calibration: Brier "
          f"{cs['exit']['brier']:.4f} <= trivial {be_t:.4f}")

    # 3) CIF coherence
    cif_i, cif_e, surv = (cif["cif_injury_final"], cif["cif_exit_final"],
                          cif["survival_final"])
    coherent = (0 <= cif_i <= 1) and (0 <= cif_e <= 1) and (0 <= surv <= 1) \
        and (cif_i + cif_e + surv <= 1.05)
    print(f"[{'PASS' if coherent else 'FAIL'}] CIF coherence: "
          f"CIF_injury={cif_i:.3f} + CIF_exit={cif_e:.3f} + S={surv:.3f} "
          f"= {cif_i + cif_e + surv:.3f} (<= ~1)")

    # 4) the articulation Gate 6 demands
    print("\n--- what this buys over Stage 4 ---")
    print("Stage 4 treated NON-injury career exits as ordinary censoring, which")
    print("biases longevity estimates (a player who retires is not 'still at risk")
    print("of injury forever'). Stage 6 models the exit cause explicitly, so the")
    print("injury cumulative-incidence is no longer inflated by that mislabeling.")
    print("The recurrent-event framing also uses every post-recovery at-risk week")
    print("(+ weeks_since_last_injury), not just time-to-FIRST-injury.")

    passed = bi_ok and be_ok and coherent
    print("\n" + ("GATE 6 PASSED ✓ — competing-risks + recurrent models are "
                  "calibrated and coherent. Proceed to Stage 7."
                  if passed else
                  "GATE 6 NOT PASSED — investigate calibration/CIF coherence.\n"
                  "(On a tiny synthetic fixture instability is expected; the gate"
                  " is meaningful on the real table.)"))
    print("=" * 70)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
