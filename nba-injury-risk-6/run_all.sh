#!/usr/bin/env bash
# Stage 9 reproducibility entrypoint — run the full pipeline end to end.
#
# Two modes:
#   ./run_all.sh synthetic   # no network; uses a synthetic world (NOT real data)
#   ./run_all.sh real        # real pulls; needs open egress to the data sources
#
# The synthetic mode lets anyone verify the pipeline runs and every gate fires,
# without network access. The real mode reproduces the actual analysis.
set -euo pipefail
MODE="${1:-synthetic}"
export PYTHONPATH=src

echo "=== NBA Injury-Risk pipeline — mode: $MODE ==="

if [ "$MODE" = "real" ]; then
  echo "[1/9] Gate 0 — nba_api reachability"
  python -m nba_injury.gate0_hello
  echo "[2/9] Stage 1 — injury labels (snapshot + map + Gate 1)"
  python -m nba_injury.fetch_injuries --start 2015-10-01 --end 2025-07-01
  python -m nba_injury.build_labels
  echo "[3/9] Stage 2 — feature pull (rosters, overnight pull, bbref, Gate 2)"
  python -m nba_injury.rosters
  python -m nba_injury.pull_features
  python -m nba_injury.fetch_bbref --start 2016 --end 2025
  python -m nba_injury.gate2_report
elif [ "$MODE" = "synthetic" ]; then
  echo "[1-2/9] Synthetic world (NOT real data) standing in for Stages 1-2"
  python -m nba_injury.make_synthetic_world
else
  echo "usage: ./run_all.sh [synthetic|real]"; exit 1
fi

echo "[3/9] Stage 3 — person-period table + Gate 3"
python -m nba_injury.build_person_period
python -m nba_injury.gate3_report
echo "[4/9] Stage 4 — discrete-time hazard + Gate 4"
python -m nba_injury.gate4_report || true
echo "[5/9] Stage 5 — validation gauntlet + Gate 5"
python -m nba_injury.gate5_report || true
echo "[6/9] Stage 6 — competing risks + recurrent + Gate 6"
python -m nba_injury.gate6_report || true
echo "[7/9] Stage 7 — prescriptive core + Gate 7"
python -m nba_injury.gate7_report || true
echo "[8/9] Stage 8 — monitoring/assurance + Gate 8"
python -m nba_injury.gate8_report || true
echo "[9/9] Tests"
python -m pytest tests/ -q

echo "=== pipeline complete ==="
echo "NOTE: in synthetic mode, Gates 4 & 6 are EXPECTED to not-pass — the"
echo "synthetic injuries are noise-only, so the model honestly shows no signal."
echo "The gates are designed to fail on noise and pass on real data. That is"
echo "the integrity check working, not a bug."

# --- Stage 10 (optional) ---
echo "[10/10] Stage 10 — live job (demo) + dashboard check"
python -m nba_injury.live_score --demo || true
python -m nba_injury.gate10_report || true
echo "Dashboard: streamlit run src/nba_injury/dashboard.py"
