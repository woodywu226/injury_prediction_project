"""Stage 2 — GATE 2 coverage report.

Gate 2 (from BUILD_PLAN): local, cached, 10-season Tier-1 coverage with NO
silent gaps, and a clear Tier-2 availability map (you know exactly which
player-games/-seasons have tracking and which don't).

This audits the cache on disk (it does NOT hit the network) and reports:
  - per (player,season): is the Tier-1 gamelog cached and non-empty?
  - per season: how many players have each Tier-2 tracking measure present?
  - any manifest entries with missing Tier-1 files (the 'silent gap' check).

Run:  python -m nba_injury.gate2_report
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

from nba_injury.cache import RAW, cached_json
from nba_injury.nba_client import SEASONS
from nba_injury.rosters import build_manifest
from nba_injury.pull_features import TRACKING_MEASURES


def _load(name: str):
    path = RAW / name
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def audit() -> dict:
    manifest = build_manifest()
    tier1_missing = []
    tier1_empty = []
    tier1_ok = 0

    for entry in manifest:
        pid, season = entry["player_id"], entry["season"]
        gl = _load(f"gamelog_{pid}_{season}.json")
        if gl is None:
            tier1_missing.append((pid, season))
            continue
        rows = gl.get("PlayerGameLog", [])
        if not rows:
            tier1_empty.append((pid, season))  # rostered but logged 0 games — OK-ish
        else:
            tier1_ok += 1

    # Tier-2 availability: read the explicit map the puller wrote.
    avail = _load("tracking_availability.json") or {}
    per_season_track = defaultdict(lambda: defaultdict(int))
    per_season_total = defaultdict(int)
    for rec in avail.values():
        season = rec.get("season")
        if season is None:
            continue
        per_season_total[season] += 1
        for measure in TRACKING_MEASURES:
            if rec.get(measure):
                per_season_track[season][measure] += 1

    return {
        "manifest_total": len(manifest),
        "tier1_ok": tier1_ok,
        "tier1_missing": tier1_missing,
        "tier1_empty": tier1_empty,
        "per_season_track": per_season_track,
        "per_season_total": per_season_total,
    }


def report() -> bool:
    a = audit()
    print("\n" + "=" * 64)
    print("GATE 2 — FEATURE-PULL COVERAGE")
    print("=" * 64)
    print(f"manifest (player,season) pairs ... {a['manifest_total']}")
    print(f"Tier-1 gamelogs cached & non-empty {a['tier1_ok']}")
    print(f"Tier-1 rostered-but-zero-games .... {len(a['tier1_empty'])} "
          f"(acceptable: rostered, never played)")
    print(f"Tier-1 MISSING (silent-gap risk) .. {len(a['tier1_missing'])}")

    if a["tier1_missing"]:
        print("\n  first missing entries (re-run pull_features to fill):")
        for pid, s in a["tier1_missing"][:10]:
            print(f"    player {pid}  {s}")

    print("\nTier-2 tracking availability (players with measure present / total):")
    if not a["per_season_total"]:
        print("  (no availability map yet — run pull_features first)")
    else:
        header = "season    " + "".join(f"{m[:9]:>11}" for m in TRACKING_MEASURES)
        print("  " + header)
        for s in SEASONS:
            tot = a["per_season_total"].get(s, 0)
            if not tot:
                continue
            cells = "".join(
                f"{a['per_season_track'][s].get(m, 0):>5}/{tot:<5}"
                for m in TRACKING_MEASURES
            )
            print(f"  {s:<9} {cells}")

    # Gate condition: no Tier-1 gaps among manifest entries, and an availability
    # map exists covering the seasons.
    no_gaps = len(a["tier1_missing"]) == 0
    have_map = bool(a["per_season_total"])
    passed = no_gaps and have_map

    print("\n--- gate conditions ---")
    print(f"[{'PASS' if no_gaps else 'FAIL'}] no missing Tier-1 gamelogs")
    print(f"[{'PASS' if have_map else 'FAIL'}] Tier-2 availability map present")
    print("\n" + ("GATE 2 PASSED ✓ — Tier-1 complete, Tier-2 mapped. Proceed to "
                  "Stage 3." if passed else
                  "GATE 2 NOT YET — fill missing pulls / run the pull, then re-run."))
    print("=" * 64)
    return passed


if __name__ == "__main__":
    raise SystemExit(0 if report() else 2)
