# Stage 10 — Live Monitoring Job + Phase-2 Frontend

> **Status:** the optional "make it feel alive" layer. Deferred by design until
> Stages 0–9 were solid (they are). This adds a free scheduled scoring job and a
> lightweight dashboard — neither is required for the project's credibility, both
> make it demonstrable.

## 1. Live monitoring job (`live_score.py` + GitHub Actions)

A weekly scheduled job that **scores the current week's players** with the frozen
trained model and appends a rolling record under `reports/live/`. It is
**inference + monitoring only — it never retrains.**

Each run writes:
- `reports/live/latest.json` — the most recent run (top-15 highest-hazard
  players, mean/p90 hazard).
- `reports/live/history.jsonl` — one line per run, so a drift history
  accumulates in the repo over time (free, no database).

**Graceful degradation:** if the live source (stats.nba.com) is unreachable, the
job records a `fetch_failed` entry and exits 0 — it never crashes the scheduled
workflow. (Verified: with no network it cleanly records the failure.)

**Schedule:** `.github/workflows/live-monitoring.yml` runs Mondays 13:00 UTC (and
on-demand via the Actions tab), then commits the rolling record back to the repo.
It needs no secrets — just Actions enabled and `contents: write` permission
(already set in the workflow).

```bash
# on your machine:
PYTHONPATH=src python -m nba_injury.live_score          # real (needs network)
PYTHONPATH=src python -m nba_injury.live_score --demo   # synthetic, no network
```

## 2. Streamlit dashboard (`dashboard.py`)

A thin presentation layer over the Stage 4–8 logic — **no modeling lives in the
dashboard**, so it can never disagree with the analysis. Four tabs:

1. **Player risk** — weekly modeled-hazard curve for a selected player, with
   peak hazard / at-risk weeks / events-on-record.
2. **Prescriptive levers** — SHAP attribution table (tagged actionable/fixed) and
   the player's model-implied modifiable levers, under the standing caveat.
3. **Monitoring** — runs the assurance pass: the COVID drift-vs-degradation
   verdict, calibration-drift chart, and the late-label estimate-vs-actual table.
4. **About / limits** — what the tool can and cannot say, and the honest
   data-ceiling note.

```bash
pip install streamlit
streamlit run src/nba_injury/dashboard.py
```

If no person-period table exists, the app offers to build a synthetic one so the
UI is explorable without network access (clearly flagged not-real).

## 3. Scope note (per the vision doc)

This is **Phase 2**, deliberately lightweight. The project's credibility lives in
the honest labels, the temporally-validated model, and the monitoring layer —
none of which needs a UI. The dashboard makes the existing work *demonstrable*;
it does not add modeling claims.

## 4. Gate 10

`PASSED` — dashboard, live scoring module, and the weekly workflow are present;
the live job runs end-to-end in demo mode. **Project fully complete (Stages 0–10).**

---
*This is the end of the build plan. The core portfolio artifact (Stages 0–9) plus
this optional live layer constitute the full project.*
