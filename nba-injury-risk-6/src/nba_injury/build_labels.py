"""Stage 1 — reconstruct time-loss episodes and run the GATE 1 tally.

Reads the frozen snapshot (data/raw/injuries_snapshot.csv), applies the
reason-string mapping layer, reconstructs per-player time-loss episodes, and
prints the go/no-go tally that decides whether the predictive framing is viable.

Episode reconstruction:
  prosportstransactions logs each IL movement as a separate row. A player going
  OUT has the `relinquished` cell populated (and a reason in `notes`); coming
  BACK has `acquired` populated. We pair each 'out' with the next 'in' for the
  same player to get an episode (start_date, end_date-ish, category). Games
  missed are approximated from the out->in date gap until Stage 3 attaches the
  real schedule. Unclosed 'out' rows (still out at snapshot end) are kept as
  right-open episodes.

GATE 1 (go/no-go) thresholds, straight from BUILD_PLAN:
  - total time-loss episodes across the window in the THOUSANDS
  - each modeled category (Achilles excepted) has at least a few HUNDRED
  - ambiguous-string rate is tolerable (large majority classifiable)

Run:  python -m nba_injury.build_labels
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime

from nba_injury.cache import raw_path, processed_path
from nba_injury.reason_mapper import classify

SNAPSHOT = "injuries_snapshot.csv"

# Gate-1 thresholds (tunable, but these are the build-plan defaults).
GATE_TOTAL_MIN = 2000          # "in the thousands"
GATE_PER_CATEGORY_MIN = 200    # "at least a few hundred" (Achilles excepted)
GATE_AMBIGUOUS_MAX_RATE = 0.15 # "tolerable" unclassifiable remainder


@dataclass
class Episode:
    player: str
    start_date: str
    end_date: str | None      # None = still out at snapshot end (right-open)
    category: str
    severe_tail: bool
    days_out: int | None
    raw_notes: str


def _parse_date(s: str):
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def load_snapshot() -> list[dict]:
    path = raw_path(SNAPSHOT)
    if not path.exists():
        raise SystemExit(
            f"[labels] {path} not found. Run the fetch step first:\n"
            f"    python -m nba_injury.fetch_injuries\n"
            f"(That step needs network egress to prosportstransactions.com.)"
        )
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def reconstruct_episodes(rows: list[dict]) -> tuple[list[Episode], Counter, int, int]:
    """Pair OUT rows with the next IN row per player -> episodes.

    Returns (episodes, ambiguous_counter, total_injury_out_rows, total_rows).
    """
    # Bucket rows per player in date order.
    per_player: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        name = (r.get("relinquished") or r.get("acquired") or "").strip()
        # The player-name cell on pst is prefixed with bullets like "• Name".
        name = name.lstrip("•").strip()
        r["_player"] = name
        r["_date"] = _parse_date(r.get("date", ""))
        per_player[name].append(r)

    episodes: list[Episode] = []
    ambiguous = Counter()
    n_injury_out = 0
    total = len(rows)

    for name, prows in per_player.items():
        prows.sort(key=lambda x: (x["_date"] or datetime.max.date()))
        open_out: dict | None = None
        open_label = None
        for r in prows:
            is_out = bool((r.get("relinquished") or "").strip())
            is_in = bool((r.get("acquired") or "").strip())
            if is_out:
                label = classify(r.get("notes", ""))
                if label.ambiguous:
                    ambiguous[r.get("notes", "").strip()[:60] or "(blank)"] += 1
                    continue
                if not label.time_loss:
                    continue  # load-management / rest — not an injury episode
                n_injury_out += 1
                # If a previous out is still open, close it as right-open first.
                if open_out is not None:
                    episodes.append(_mk_episode(name, open_out, None, open_label))
                open_out, open_label = r, label
            elif is_in and open_out is not None:
                episodes.append(_mk_episode(name, open_out, r, open_label))
                open_out, open_label = None, None
        # Player still out at snapshot end.
        if open_out is not None:
            episodes.append(_mk_episode(name, open_out, None, open_label))

    return episodes, ambiguous, n_injury_out, total


def _mk_episode(name, out_row, in_row, label) -> Episode:
    start = out_row["_date"]
    end = in_row["_date"] if in_row else None
    days = (end - start).days if (start and end) else None
    return Episode(
        player=name,
        start_date=start.isoformat() if start else "",
        end_date=end.isoformat() if end else None,
        category=label.category,
        severe_tail=label.severe_tail,
        days_out=days,
        raw_notes=out_row.get("notes", "").strip(),
    )


def write_episodes(episodes: list[Episode]) -> None:
    out = processed_path("injury_episodes.csv")
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(episodes[0]).keys())
                           if episodes else
                           ["player", "start_date", "end_date", "category",
                            "severe_tail", "days_out", "raw_notes"])
        w.writeheader()
        for e in episodes:
            w.writerow(asdict(e))
    print(f"[labels] wrote {len(episodes)} episodes -> {out}")


def gate1_report(episodes, ambiguous, n_injury_out, total) -> bool:
    cat_counts = Counter(e.category for e in episodes)
    n_amb = sum(ambiguous.values())
    classifiable_denom = n_injury_out + n_amb
    amb_rate = (n_amb / classifiable_denom) if classifiable_denom else 0.0

    print("\n" + "=" * 64)
    print("GATE 1 — INJURY LABEL TALLY (go/no-go for the whole project)")
    print("=" * 64)
    print(f"raw snapshot rows ............ {total}")
    print(f"time-loss injury episodes .... {len(episodes)}")
    print(f"ambiguous strings (set aside)  {n_amb}  "
          f"({amb_rate:.1%} of injury-flavored rows)")
    print("\nepisodes per category:")
    for cat, n in cat_counts.most_common():
        print(f"   {cat:<26} {n}")
    severe = sum(1 for e in episodes if e.severe_tail)
    print(f"\nsevere-tail (Achilles) episodes: {severe}  "
          f"(spotlighted rare tail — NOT gated on count)")

    # --- evaluate the three gate conditions ---
    cond_total = len(episodes) >= GATE_TOTAL_MIN
    gated_cats = {c: n for c, n in cat_counts.items() if c != "achilles"}
    thin = {c: n for c, n in gated_cats.items() if n < GATE_PER_CATEGORY_MIN}
    cond_cats = len(thin) == 0
    cond_amb = amb_rate <= GATE_AMBIGUOUS_MAX_RATE

    print("\n--- gate conditions ---")
    print(f"[{'PASS' if cond_total else 'FAIL'}] total episodes >= {GATE_TOTAL_MIN}"
          f"  (got {len(episodes)})")
    print(f"[{'PASS' if cond_cats else 'FAIL'}] every category >= "
          f"{GATE_PER_CATEGORY_MIN} (Achilles excepted)"
          + ("" if cond_cats else f"  thin: {thin}"))
    print(f"[{'PASS' if cond_amb else 'FAIL'}] ambiguous rate <= "
          f"{GATE_AMBIGUOUS_MAX_RATE:.0%}  (got {amb_rate:.1%})")

    passed = cond_total and cond_cats and cond_amb
    print("\n" + ("GATE 1 PASSED ✓ — predictive framing is supportable."
                  if passed else
                  "GATE 1 FAILED ✗ — fall back to descriptive/epidemiological\n"
                  "framing (vision §4) OR enrich the mapping layer and re-run."))
    print("=" * 64)

    if ambiguous and not passed:
        print("\nTop unmatched strings to triage into reason_map.yaml:")
        for s, n in ambiguous.most_common(15):
            print(f"   {n:>4}  {s}")
    return passed


def main() -> int:
    rows = load_snapshot()
    episodes, ambiguous, n_injury_out, total = reconstruct_episodes(rows)
    if episodes:
        write_episodes(episodes)
    passed = gate1_report(episodes, ambiguous, n_injury_out, total)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
