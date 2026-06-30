# NBA Injury-Risk & Career-Longevity Decision Support

Public-data NBA injury-risk modeling with a prescriptive lever layer and a
model-assurance/monitoring layer. See `PROJECT_VISION.md` for the *what/why*,
`BUILD_PLAN.md` for the staged *how*, and **`reports/FINAL_REPORT.md` for the
complete write-up.**

**Status:** complete (Stages 0–9). A discrete-time hazard model on a
person-period dataset, with strict temporal validation, an anti-hindsight
validation gauntlet, competing-risks + recurrent-event depth, a prescriptive
lever layer with causal humility enforced in code, and a drift/degradation
monitoring layer. 59 tests; every stage has a gate.

## Quickstart

```bash
pip install -r requirements.txt
./run_all.sh synthetic     # full pipeline, no network (synthetic, NOT real data)
./run_all.sh real          # reproduce the real analysis (needs open egress)
```

Synthetic mode lets anyone verify the pipeline runs and every gate fires without
network access. **Gates 4 & 6 are expected to not-pass in synthetic mode** — the
synthetic injuries are noise-only, so the model honestly shows no signal. The
gates are designed to fail on noise and pass on real data; that is the integrity
check working.

## Layout
```
data/raw/         cached external pulls (git-ignored; write-once)
data/processed/   canonical Parquet/CSV tables (git-ignored)
src/nba_injury/   package code
tests/            gate checks codified as tests
reports/          written deliverable (later stages)
DECISIONS.md      append-only judgment-call log
```

## Setup (Stage 0)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install nba_api          # Stage 2 backbone; needed by the Gate-0 check
```

### Gate 0 — smoke test
```bash
PYTHONPATH=src python -m nba_injury.gate0_hello
```
Pulls one player's career stats from nba_api and caches it. **Gate 0 passes
only when this succeeds** — it also confirms `stats.nba.com` is reachable from
your network.

## Stage 1 — injury labels (the critical go/no-go gate)

```bash
# 1. Snapshot + freeze the public injury source (needs open network egress)
PYTHONPATH=src python -m nba_injury.fetch_injuries --start 2015-10-01 --end 2025-07-01

# 2. Map reason strings, reconstruct episodes, run the Gate-1 tally
PYTHONPATH=src python -m nba_injury.build_labels
```

`build_labels` prints the go/no-go report and exits non-zero if Gate 1 fails.
Outputs `data/processed/injury_episodes.csv`.

### Iterating the mapping layer
The reason-string dictionary is `src/nba_injury/reason_map.yaml`. When the gate
report lists unmatched strings, add patterns there and re-run — no code changes
needed. Log each judgment call in `DECISIONS.md`.

### Tests
```bash
PYTHONPATH=src python -m pytest tests/ -q
```

## Stage 4 — discrete-time hazard model

After Stage 3 produces `data/processed/person_period.parquet`:

```bash
PYTHONPATH=src python -m nba_injury.model_hazard                        # train + temporal eval
PYTHONPATH=src python -m nba_injury.model_hazard --report-json reports/stage4_metrics.json
PYTHONPATH=src python -m nba_injury.gate4_report                        # Gate 4 verdict
```

Logistic regression (interpretable baseline) and HistGradientBoosting (native
Tier-2 missingness handling), validated **temporally** (train earlier seasons,
test later — never a random split). Reports AUC-PR / ROC-AUC / Brier /
calibration — never accuracy. Gate 4 passes when the model beats both the
constant-hazard and age+minutes baselines.

Walkthrough: `notebooks/stage4_hazard_model.ipynb`.
Write-up template: `reports/stage4_interim_report.md`.

## Stage 5 — validation gauntlet (hindsight-bias guard)

After Stage 4, with `person_period.parquet` in place:

```bash
PYTHONPATH=src python -m nba_injury.audit_leakage                       # leakage audit
PYTHONPATH=src python -m nba_injury.stylistic_comparables               # comparables check
PYTHONPATH=src python -m nba_injury.stylistic_comparables --index-player <rose_id> --index-player <zion_id>
PYTHONPATH=src python -m nba_injury.gate5_report                        # Gate 5 verdict
```

Two adversarial checks against our own model: a temporal-leakage audit
(structural causality + target-shuffle test) and the stylistic-comparables
check (does an index case outrank its *healthy* style look-alikes on
pre-injury-only data?). Gate 5 passes on **honest characterization of limits** —
leakage is the only hard fail.

Walkthrough: `notebooks/stage5_validation_gauntlet.ipynb`.
Write-up template: `reports/stage5_validation_report.md`.

**Stages 0–5 = the Minimum Viable Honest Project** (portfolio-ready on their own).

## Stage 6 — competing risks + recurrent events (depth)

```bash
PYTHONPATH=src python -m nba_injury.model_competing_risks            # sklearn anchor + CIF
PYTHONPATH=src python -m nba_injury.model_competing_risks --pydts    # also try PyDTS cross-check
PYTHONPATH=src python -m nba_injury.gate6_report                     # Gate 6 verdict
```

Cause-specific discrete-time hazards (injury vs non-injury career exit) combined
into a cumulative incidence function, plus a recurrent-event framing
(`weeks_since_last_injury`). Removes the Stage-4 bias of treating exits as
censoring. Cause models are fit **calibrated** (no balanced weighting) because
the CIF needs honest probabilities. PyDTS is wired in as an optional cross-check
that **degrades gracefully** to the sklearn anchor on any rough edge.

Walkthrough: `notebooks/stage6_competing_risks.ipynb`.
Write-up template: `reports/stage6_competing_risks_report.md`.

## Stage 7 — prescriptive / causal core (the heart)

```bash
PYTHONPATH=src python -m nba_injury.prescriptive                 # full prescriptive pass
PYTHONPATH=src python -m nba_injury.prescriptive --player <id>   # one player's levers
PYTHONPATH=src python -m nba_injury.gate7_report                 # Gate 7 verdict
```

SHAP interpretation (what the **model** attributes risk to), a modifiable/fixed
lever partition, counterfactual hazards (all-else-equal, model-implied), and
observational validation of the top lever. Every output carries a standing
caveat; an overclaim guard **refuses** counterfactuals on fixed features. The
highest overclaim-risk stage — causal humility is enforced in code, not just
prose.

Walkthrough: `notebooks/stage7_prescriptive.ipynb`.
Write-up template: `reports/stage7_prescriptive_report.md`.

## Stage 8 — monitoring / assurance

```bash
PYTHONPATH=src python -m nba_injury.monitoring              # full assurance pass
PYTHONPATH=src python -m nba_injury.monitoring --evidently  # also try Evidently cross-check
PYTHONPATH=src python -m nba_injury.gate8_report            # Gate 8 verdict
```

Covariate drift (native PSI + KS, multi-test), subgroup performance, calibration
drift (on a calibrated model), a late-label two-mode design (NannyML-style
expected-Brier estimate vs retrospective actual), and the **verifiable-event
demo** on the COVID seasons. The crux: it separates harmless input drift from
real degradation — drift is not automatically failure. Evidently/NannyML are
optional cross-checks; the native implementations are the reproducible primary
path.

Walkthrough: `notebooks/stage8_monitoring.ipynb`.
Write-up template: `reports/stage8_monitoring_report.md`.

## Stage 10 — live job + dashboard (optional)

```bash
# weekly scoring job (inference + monitoring only; never retrains)
PYTHONPATH=src python -m nba_injury.live_score          # real (needs network)
PYTHONPATH=src python -m nba_injury.live_score --demo   # synthetic, no network

# lightweight dashboard: player -> risk curve -> levers -> monitoring
pip install streamlit
streamlit run src/nba_injury/dashboard.py
```

The scheduled job (`.github/workflows/live-monitoring.yml`) runs weekly on GitHub
Actions, scores the current week, and commits a rolling drift record to
`reports/live/`. It degrades gracefully if the live source is unreachable. The
dashboard is a thin layer over the Stage 4–8 logic (no modeling lives in it).

Write-up: `reports/stage10_live_frontend_report.md`.

## ⚠️ Network note
`stats.nba.com`, `basketball-reference.com`, and `prosportstransactions.com` are
**not reachable from every environment** (some sandboxes allowlist egress).
Run the two fetch steps (`gate0_hello`, `fetch_injuries`) on a machine with open
internet. Everything downstream reads only the frozen local snapshot, so it is
fully reproducible offline.

For local development without network access, `make_synthetic_snapshot` writes a
**realistic but fake** snapshot so you can exercise the full pipeline:
```bash
PYTHONPATH=src python -m nba_injury.make_synthetic_snapshot   # DEV ONLY — not real data
```
Never make a real Gate-1 decision on synthetic data.
