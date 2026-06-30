"""Stage 2 — Basketball Reference biographical cross-check / gap-fill.

nba_api gives biographical data via commonplayerinfo, but the build plan calls
for Basketball Reference as an independent cross-check / gap-fill for age,
position, height, weight (mass matters for the Zion-type profile). We snapshot
the per-season totals pages (which carry Age + Pos) and the player directory
(which carries height/weight), freeze them, and read only the frozen copy.

Snapshots to data/raw/:
  bbref_season_<YYYY>.csv   per-season player totals (age, pos, team)
  bbref_bio.csv             height/weight from player pages (gap-fill)

Run:  python -m nba_injury.fetch_bbref --start 2016 --end 2025
(BBRef rate-limits hard: >20 req/min triggers a temporary block. We pace at
~1 req / 3.5s and cache every page, so a re-run resumes.)
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import time

from nba_injury.cache import raw_path, cached_text

BASE = "https://www.basketball-reference.com"
SEASON_TOTALS = BASE + "/leagues/NBA_{year}_totals.html"
PACE_SECONDS = 3.5  # BBRef blocks aggressive scrapers; be gentle


def _fetch_html(url: str, cache_name: str) -> str:
    import requests

    def build():
        time.sleep(PACE_SECONDS)
        resp = requests.get(url, timeout=30,
                            headers={"User-Agent": "research-snapshot/1.0"})
        resp.raise_for_status()
        return resp.text

    return cached_text(cache_name, build)


def _parse_totals_table(html: str, year: int) -> list[dict]:
    """Extract age/pos/team rows from a BBRef season-totals page.

    BBRef tables are standard HTML; pandas.read_html handles them. We keep only
    the identity+bio columns we need (Player, Age, Pos, Tm) and tag the season.
    """
    import pandas as pd

    # BBRef sometimes wraps the real table in HTML comments; strip comment markers
    # so pandas sees every table.
    cleaned = html.replace("<!--", "").replace("-->", "")
    try:
        tables = pd.read_html(io.StringIO(cleaned))
    except ValueError:
        return []
    # The totals table has a 'Player' and 'Age' column.
    for df in tables:
        cols = {str(c) for c in df.columns}
        if "Player" in cols and "Age" in cols:
            keep = [c for c in ("Player", "Age", "Pos", "Tm", "Team")
                    if c in df.columns]
            sub = df[keep].copy()
            sub = sub[sub["Player"] != "Player"]  # drop repeated header rows
            sub["season_start_year"] = year
            return sub.to_dict("records")
    return []


def fetch_seasons(start_year: int, end_year: int) -> None:
    """start_year/end_year are the season *ending* years (e.g. 2016 = 2015-16)."""
    all_rows: list[dict] = []
    for year in range(start_year, end_year + 1):
        url = SEASON_TOTALS.format(year=year)
        try:
            html = _fetch_html(url, f"bbref_totals_{year}.html")
        except Exception as exc:  # noqa: BLE001
            print(f"[bbref] {year} failed: {exc}", file=sys.stderr)
            continue
        rows = _parse_totals_table(html, year)
        all_rows.extend(rows)
        print(f"[bbref] {year}: {len(rows)} player rows", file=sys.stderr)

    if not all_rows:
        print("[bbref] no rows parsed — nothing written.", file=sys.stderr)
        return

    out = raw_path("bbref_season_totals.csv")
    fields = ["Player", "Age", "Pos", "Tm", "Team", "season_start_year"]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    raw_path("BBREF_PROVENANCE.md").write_text(
        f"# Basketball Reference snapshot\n\n"
        f"- Source: basketball-reference.com season totals pages\n"
        f"- Season ending-years: {start_year}..{end_year}\n"
        f"- Rows: {len(all_rows)}\n"
        f"- Purpose: independent cross-check / gap-fill for age, position, team.\n"
        f"- Read-only frozen copy; downstream reads this file.\n",
        encoding="utf-8",
    )
    print(f"[bbref] wrote {len(all_rows)} rows -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Snapshot Basketball Reference bio.")
    ap.add_argument("--start", type=int, default=2016, help="season ending year")
    ap.add_argument("--end", type=int, default=2025, help="season ending year")
    args = ap.parse_args()
    if raw_path("bbref_season_totals.csv").exists():
        print("[bbref] snapshot already exists — delete to force re-fetch.")
        return 0
    try:
        fetch_seasons(args.start, args.end)
    except Exception as exc:  # noqa: BLE001
        print(f"[bbref] FAILED: {exc}", file=sys.stderr)
        print("[bbref] If proxy-blocked, run on a machine with open egress to "
              "basketball-reference.com.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
