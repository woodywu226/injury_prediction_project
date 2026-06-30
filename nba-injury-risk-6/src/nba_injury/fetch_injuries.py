"""Stage 1 — fetch & FREEZE the public injury source.

Source: prosportstransactions.com "Injuries" transaction log. This is the
canonical public NBA injury-history source (it backs nearly every public NBA
injury dataset). It records, per date, a player moving to/from the
Inactive/Injured list with a free-text "Notes" reason string — exactly the
messy designations the mapping layer (Stage 1) normalizes.

Per BUILD_PLAN: SNAPSHOT it. Save a frozen copy to data/raw/ and never assume
the source persists. Downstream code reads ONLY the frozen snapshot, so results
are reproducible even if the site changes.

Output: data/raw/injuries_snapshot.csv with columns:
    date, team, acquired, relinquished, notes

  - `relinquished` populated  -> player went OUT (injury/IL placement begins)
  - `acquired` populated       -> player came back (activated / returned)
  - `notes`                    -> the raw reason string to be mapped

Run:  python -m nba_injury.fetch_injuries --start 2015-10-01 --end 2025-07-01
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from html.parser import HTMLParser
from urllib.parse import urlencode

from nba_injury.cache import raw_path

BASE = "https://www.prosportstransactions.com/basketball/Search/SearchResults.php"
SNAPSHOT = "injuries_snapshot.csv"
PROVENANCE = "SNAPSHOT_PROVENANCE.md"


class _ResultsTableParser(HTMLParser):
    """Parse the prosportstransactions results table into rows of 5 cells.

    The page is a single HTML <table> of class 'datatable'. We collect every
    data row (skipping the header) as [date, team, acquired, relinquished, notes].
    """

    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.current_row: list[str] = []
        self.current_cell: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        if tag == "table" and "datatable" in (attrs_d.get("class") or ""):
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.in_row = True
            self.current_row = []
        elif self.in_row and tag == "td":
            self.in_cell = True
            self.current_cell = []

    def handle_endtag(self, tag):
        if tag == "td" and self.in_cell:
            self.in_cell = False
            self.current_row.append("".join(self.current_cell).strip())
        elif tag == "tr" and self.in_row:
            self.in_row = False
            if len(self.current_row) == 5:
                self.rows.append(self.current_row)
        elif tag == "table" and self.in_table:
            self.in_table = False

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell.append(data)


def _fetch_page(start: str, end: str, offset: int, session) -> str:
    params = {
        "Player": "",
        "Team": "",
        "BeginDate": start,
        "EndDate": end,
        # InjuriesChkBx=yes restricts results to the injury log specifically.
        "InjuriesChkBx": "yes",
        "Submit": "Search",
        "start": offset,
    }
    url = f"{BASE}?{urlencode(params)}"
    resp = session.get(url, timeout=30, headers={"User-Agent": "research-snapshot/1.0"})
    resp.raise_for_status()
    return resp.text


def fetch_all(start: str, end: str, *, polite_delay: float = 2.0) -> list[list[str]]:
    """Page through the full injury log between start and end (YYYY-MM-DD).

    Polite, rate-limited paging (results paginate 25/page via the `start` offset).
    """
    import requests  # lazy import; not needed just to inspect the module

    session = requests.Session()
    all_rows: list[list[str]] = []
    offset = 0
    page = 0
    header_seen_cols: list[str] | None = None
    while True:
        html = _fetch_page(start, end, offset, session)
        parser = _ResultsTableParser()
        parser.feed(html)
        rows = parser.rows
        # First row of the table is the header; capture once, then drop it.
        if rows and header_seen_cols is None:
            header_seen_cols = rows[0]
        data_rows = [r for r in rows if r != header_seen_cols]
        if not data_rows:
            break
        all_rows.extend(data_rows)
        page += 1
        print(f"[fetch] page {page}: +{len(data_rows)} rows (total {len(all_rows)})",
              file=sys.stderr)
        offset += 25
        time.sleep(polite_delay)  # be a good citizen
        if page > 2000:  # hard safety stop
            print("[fetch] safety stop at 2000 pages", file=sys.stderr)
            break
    return all_rows


def write_snapshot(rows: list[list[str]], start: str, end: str) -> None:
    out = raw_path(SNAPSHOT)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "team", "acquired", "relinquished", "notes"])
        w.writerows(rows)
    # Freeze provenance alongside the data — when/where/how it was pulled.
    prov = raw_path(PROVENANCE)
    prov.write_text(
        f"# Injury snapshot provenance\n\n"
        f"- Source: prosportstransactions.com basketball injuries log\n"
        f"- Date range queried: {start} .. {end}\n"
        f"- Rows captured: {len(rows)}\n"
        f"- Pulled: frozen at fetch time; downstream reads this file ONLY.\n"
        f"- NOTE: this is a chosen, defensible public source. It records\n"
        f"  *availability* (IL placement/return), NOT clean medical diagnoses.\n",
        encoding="utf-8",
    )
    print(f"[fetch] wrote {len(rows)} rows -> {out}")
    print(f"[fetch] wrote provenance  -> {prov}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Snapshot the public NBA injury log.")
    ap.add_argument("--start", default="2015-10-01", help="YYYY-MM-DD")
    ap.add_argument("--end", default="2025-07-01", help="YYYY-MM-DD")
    ap.add_argument("--delay", type=float, default=2.0, help="polite delay (s)")
    args = ap.parse_args()

    if raw_path(SNAPSHOT).exists():
        print(f"[fetch] {SNAPSHOT} already exists — refusing to re-fetch "
              f"(delete it to force). This is the write-once cache rule.")
        return 0
    try:
        rows = fetch_all(args.start, args.end, polite_delay=args.delay)
    except Exception as exc:  # noqa: BLE001
        print(f"[fetch] FAILED: {exc}", file=sys.stderr)
        print("[fetch] If this is a proxy/network block, run on a machine with "
              "open egress to prosportstransactions.com.", file=sys.stderr)
        return 1
    write_snapshot(rows, args.start, args.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
