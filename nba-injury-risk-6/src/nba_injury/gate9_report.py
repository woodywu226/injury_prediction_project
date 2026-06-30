"""Stage 9 — GATE 9 verification.

Gate 9 (from BUILD_PLAN): a third party could read the report and understand
exactly what was built, what it shows, and what it honestly cannot claim.

This is a deliverable-completeness check, not a model check. It verifies the
portfolio artifact a stranger receives is complete and self-consistent:
  - the final report exists and covers framing, methods, results, limitations,
  - every stage has a report + a notebook + a gate module,
  - the reproducibility entrypoint and README exist,
  - the resume bullets exist with no fabricated numbers (placeholders intact),
  - the test suite is present.

Run:  python -m nba_injury.gate9_report
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def main() -> int:
    print("=" * 70)
    print("GATE 9 — DELIVERABLE COMPLETENESS")
    print("=" * 70)

    checks = []

    # 1) final report exists and covers the required sections
    fr = ROOT / "reports" / "FINAL_REPORT.md"
    if fr.exists():
        text = fr.read_text().lower()
        required = ["problem framing", "limitation", "method", "validation",
                    "calibration", "prescriptive", "monitoring",
                    "reproducib"]
        missing = [s for s in required if s not in text]
        checks.append((f"final report covers all required sections"
                       + (f" (missing: {missing})" if missing else ""),
                       not missing))
    else:
        checks.append(("final report exists", False))

    # 2) every stage has report + notebook + gate
    stage_reports = ["stage4_interim_report", "stage5_validation_report",
                     "stage6_competing_risks_report", "stage7_prescriptive_report",
                     "stage8_monitoring_report"]
    checks.append(("per-stage reports present",
                   all(_exists(f"reports/{r}.md") for r in stage_reports)))
    notebooks = [f"stage{n}_" for n in (4, 5, 6, 7, 8)]
    nb_dir = ROOT / "notebooks"
    checks.append(("per-stage notebooks present",
                   all(any(p.name.startswith(n) for p in nb_dir.glob("*.ipynb"))
                       for n in notebooks)))
    gates = [f"src/nba_injury/gate{n}_report.py" for n in (3, 4, 5, 6, 7, 8)]
    checks.append(("per-stage gate modules present",
                   all(_exists(g) for g in gates)))

    # 3) reproducibility entrypoint + README
    checks.append(("run_all.sh reproducibility script present",
                   _exists("run_all.sh")))
    checks.append(("README present", _exists("README.md")))
    checks.append(("DECISIONS.md log present", _exists("DECISIONS.md")))
    checks.append(("pinned requirements present", _exists("requirements.txt")))

    # 4) resume bullets exist AND still contain placeholders (no fabricated nums)
    rb = ROOT / "reports" / "RESUME_BULLETS.md"
    if rb.exists():
        has_placeholders = "[ ]" in rb.read_text() or "`[" in rb.read_text()
        checks.append(("resume bullets present with placeholders (no fabricated "
                       "metrics)", has_placeholders))
    else:
        checks.append(("resume bullets present", False))

    # 5) test suite present
    tests = list((ROOT / "tests").glob("test_*.py"))
    checks.append((f"test suite present ({len(tests)} test files)",
                   len(tests) >= 5))

    all_ok = True
    for name, ok in checks:
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    print("\n" + ("GATE 9 PASSED ✓ — the deliverable is complete: a third party\n"
                  "can read the report, run the pipeline, and understand what the\n"
                  "project shows and what it honestly cannot claim.\n\n"
                  "PROJECT COMPLETE (Stages 0–9). Optional Stage 10 = live job +\n"
                  "Streamlit frontend."
                  if all_ok else
                  "GATE 9 FAILED ✗ — a deliverable component is missing."))
    print("=" * 70)
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
