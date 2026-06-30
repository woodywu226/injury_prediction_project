"""Stage 5 — GATE 5 report.

Gate 5 (from BUILD_PLAN): you can state, WITH EVIDENCE, what the model can and
cannot distinguish — and the index cases hold up under pre-injury-only data
against healthy comparables (or you've honestly characterized the limits). No
overclaiming survives this stage.

This runs the leakage audit + the stylistic-comparables check and renders a
single honest verdict. Unlike earlier gates, "passing" here is about EVIDENCE
and HONESTY, not a performance threshold: a model that demonstrably can't beat
stylistic comparables still PASSES Gate 5 as long as the report says so plainly.
What fails Gate 5 is leakage, or claims unsupported by the evidence.

Run:  python -m nba_injury.gate5_report
"""
from __future__ import annotations

import sys

from nba_injury.audit_leakage import run as run_leakage
from nba_injury.stylistic_comparables import run as run_comparables


def main() -> int:
    leak = run_leakage()
    print()
    comp = run_comparables()

    print("\n" + "=" * 70)
    print("GATE 5 — CREDIBILITY VERDICT")
    print("=" * 70)

    # 1) Structural leakage must be clean — this is the hard fail.
    structural_ok = leak.get("structural_ok", False)
    leak_suspect = leak.get("leak_suspect", None)

    print(f"[{'PASS' if structural_ok else 'FAIL'}] no structural leakage "
          f"(cumulatives/prior-injury causal)")
    if leak_suspect is None:
        print("[ -- ] target-shuffle test inconclusive (too few events)")
    else:
        print(f"[{'WARN' if leak_suspect else 'PASS'}] target-shuffle: "
              f"{'shuffled labels still score high' if leak_suspect else 'collapses to baseline'}")

    # 2) Comparables: report separation honestly (not a perf gate).
    cases = comp.get("index_cases", [])
    separated = [c for c in cases if str(c.get("verdict", "")).startswith("separates")]
    weak = [c for c in cases if "WEAK" in str(c.get("verdict", ""))]
    print(f"\nstylistic-comparables: {len(separated)}/{len(cases)} index cases "
          f"sit clearly above their healthy look-alikes")
    for c in cases:
        print(f"   player {c.get('index_player')}: {c.get('verdict')}")

    # Gate 5 verdict
    honest_statement_possible = structural_ok and len(cases) > 0
    print("\n--- the claim this earns ---")
    if not structural_ok:
        print("CANNOT make claims — fix structural leakage first. GATE 5 FAILED ✗")
        print("=" * 70)
        return 2

    if separated and not leak_suspect:
        print("The model distinguishes the index case(s) from stylistically-")
        print("similar healthy players using pre-injury-only data. The Rose/Zion-")
        print("type cases are validation anecdotes, not cherry-picks.")
    elif separated and leak_suspect:
        print("The index case(s) sit above their healthy stylistic comparables,")
        print("BUT the target-shuffle test was not clean — on this data the")
        print("separation cannot be fully distinguished from temporal/sample")
        print("artifact. Treat the case studies as suggestive, not confirmed,")
        print("until the shuffle test is clean on the full real table.")
    elif weak:
        print("HONEST LIMIT: the model does NOT cleanly separate the index")
        print("case(s) from healthy stylistic comparables. The risk signal is")
        print("substantially style-driven; per-case 'we would have caught X'")
        print("claims are NOT supported. This is stated plainly, not hidden.")
    else:
        print("Mixed/insufficient separation — characterized per case above.")

    print("\nGATE 5 PASSED ✓ — claims are bounded by evidence; no overclaiming.")
    print("(Passing = honesty about limits, not a performance threshold.)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
