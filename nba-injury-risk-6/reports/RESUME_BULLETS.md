# Resume Bullets — fill from the REAL run (no fabricated metrics)

These are drafted with bracketed placeholders. Fill each `[ ]` from the actual
output of `./run_all.sh real`. Do NOT invent numbers — the whole project's
credibility rests on honest metrics. Pick the 2–3 bullets that match the role.

## Health-AI / ML-validation framing (primary)

- Built an end-to-end, **reproducible model-assurance pipeline** on public NBA
  data — discrete-time hazard model on a `[R]`-row person-period dataset
  (`[P]` players, 10 seasons) — with strict temporal validation, achieving
  AUC-PR `[ ]` (`[ ]`× a league-average-hazard baseline) on held-out later
  seasons.

- Designed an **anti-hindsight validation gauntlet** (temporal-leakage audit +
  target-shuffle test + stylistic-comparables check) that distinguishes genuine
  pre-injury risk signal from survivorship bias, and a **monitoring layer**
  (PSI/KS drift, calibration drift, late-label performance estimation) that
  separates harmless input drift from true model degradation across the
  COVID-disrupted seasons.

- Implemented a **prescriptive lever layer** with causal humility enforced in
  code — SHAP interpretation, a modifiable/fixed feature partition, and
  counterfactual hazards framed as model-implied hypotheses, with an overclaim
  guard that refuses interventions on fixed attributes — validated observationally
  rather than asserted.

## Data-engineering / pipeline framing (secondary)

- Engineered a cached, resumable data pipeline (rate-limited API client,
  write-once snapshot caching, `[N]`-episode injury-label mapping layer from
  free-text reason strings) producing a leakage-audited person-period modeling
  table, with `[59]` automated tests and gate checks at every stage.

## Statistical-rigor framing (for quant/biostat-leaning roles)

- Modeled injury risk with **competing risks** (injury vs non-injury career exit,
  cause-specific calibrated hazards → cumulative incidence) and **recurrent
  events**, removing the longevity bias of treating career exits as ordinary
  censoring; reported rare-event metrics (AUC-PR, calibration) over accuracy
  throughout.

---
*Tip: lead with the framing bullet that matches the job's primary lane, then add
one supporting bullet. Keep total to 2–3 lines on the resume.*
