"""Generate a REALISTIC synthetic injury snapshot to exercise the Stage-1
pipeline end-to-end where the live source isn't reachable (e.g. sandboxes).

This is a TEST/DEV fixture only — it is NOT real data and must never feed a
real Gate-1 decision. It reproduces the prosportstransactions schema and
realistic reason-string phrasing + plausible category frequencies, so you can
confirm episode reconstruction and the gate report work before running the
real fetch on an open network.

Run:  python -m nba_injury.make_synthetic_snapshot   # writes data/raw/injuries_snapshot.csv
"""
from __future__ import annotations

import csv
import random
from datetime import date, timedelta

from nba_injury.cache import raw_path

random.seed(42)

# Realistic phrasing variants per category, weighted by rough real-world freq.
REASON_BANK = {
    "lower_limb_soft_tissue": (
        38, ["sprained left ankle", "right ankle soreness", "strained calf",
             "left hamstring strain", "sore right foot", "groin strain",
             "plantar fasciitis", "left quad contusion", "hip flexor strain",
             "tight calf (DNP)"]),
    "general_soreness": (
        14, ["placed on IL", "lower body soreness", "general soreness",
             "knee surgery recovery", "out (injury)", "patellar tendinopathy"]),
    "knee_other": (
        9, ["right knee soreness", "left knee inflammation", "knee contusion"]),
    "back": (
        8, ["lower back spasms", "back tightness", "lumbar strain",
            "herniated disc"]),
    "knee_ligament": (
        6, ["torn ACL right knee", "MCL sprain", "torn meniscus",
            "partial ACL tear (out for season)"]),
    "hand_finger": (
        6, ["fractured right thumb", "dislocated finger", "sprained wrist"]),
    "upper_body_other": (
        6, ["left shoulder strain", "sore right elbow", "rib contusion",
            "oblique strain"]),
    "illness": (
        7, ["health and safety protocols", "flu-like symptoms", "illness",
            "non-COVID illness"]),
    "concussion": (
        3, ["concussion protocol", "facial fracture", "broken nose"]),
    "achilles": (
        1, ["torn left Achilles tendon (out for season)",
            "right Achilles soreness"]),
}
LOAD_MGMT = ["rest (DNP)", "load management", "coach's decision",
             "personal reasons"]
AMBIGUOUS = ["returned to lineup under unclear status xyz",
             "roster move (unspecified)"]

TEAMS = ["LAL", "BOS", "GSW", "MIA", "DEN", "PHX", "MIL", "DAL", "PHI", "NYK"]


def _weighted_categories():
    bag = []
    for cat, (w, _strings) in REASON_BANK.items():
        bag += [cat] * w
    return bag


def generate(n_players=420, seasons_start=2015, seasons_end=2025):
    """~420 players × multiple episodes over 10 seasons -> thousands of rows."""
    rows = []
    cat_bag = _weighted_categories()
    start_window = date(seasons_start, 10, 1)
    end_window = date(seasons_end, 6, 30)
    span_days = (end_window - start_window).days

    for pid in range(n_players):
        name = f"• Player{pid:03d}"
        # Each player has a Poisson-ish number of injury events over the window.
        n_events = max(0, int(random.gauss(7, 4)))
        for _ in range(n_events):
            roll = random.random()
            d_out = start_window + timedelta(days=random.randint(0, span_days))
            if roll < 0.12:  # load management
                notes = random.choice(LOAD_MGMT)
            elif roll < 0.15:  # genuinely ambiguous
                notes = random.choice(AMBIGUOUS)
            else:  # injury
                cat = random.choice(cat_bag)
                notes = random.choice(REASON_BANK[cat][1])
            team = random.choice(TEAMS)
            # OUT row
            rows.append([d_out.isoformat(), team, "", name, notes])
            # Matching IN row a plausible number of days later (80% close).
            if random.random() < 0.8:
                d_in = d_out + timedelta(days=random.randint(2, 45))
                if d_in <= end_window:
                    rows.append([d_in.isoformat(), team, name, "",
                                 "returned to lineup"])
    rows.sort(key=lambda r: r[0])
    return rows


def main():
    out = raw_path("injuries_snapshot.csv")
    rows = generate()
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "team", "acquired", "relinquished", "notes"])
        w.writerows(rows)
    print(f"[synthetic] wrote {len(rows)} rows -> {out}")
    print("[synthetic] WARNING: synthetic dev fixture, NOT real data.")


if __name__ == "__main__":
    main()
