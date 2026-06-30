# NBA Injury-Risk & Career-Longevity Decision Support
## Final Report — a public-data, reproducible model-assurance project

> **What this is.** A public-data, league-wide NBA injury-risk model built as a
> *discrete-time hazard* on a person-period dataset, with a **prescriptive
> lever layer** and a **model-assurance / monitoring layer**. It is framed as
> **decision support for team performance and sports-science staff — explicitly
> not autonomous medical advice.**
>
> **What makes it credible.** Honest labels, a correctly-specified and
> temporally-validated model, an anti-hindsight validation gauntlet, causal
> humility enforced *in code*, and a monitoring layer that distinguishes harmless
> drift from real degradation. Numbers in `[brackets]` are filled from the real
> run; the pipeline and every gate are already built, tested, and reproducible.
>
> **Why NBA data.** This project reuses an "AI assurance / model monitoring"
> spine (from prior health-imaging / ML-validation work) repointed at NBA data
> *because the data is fully public and the results are verifiable by anyone* —
> exactly the property a portfolio reviewer can check.

---

## 1. Problem framing

Estimate, for each at-risk **player-week**, the probability that a time-loss
injury begins that week — a discrete-time hazard. From one such model you read
both near-term risk (forward-sum hazards) and injury-free longevity (integrate),
while it natively handles censoring of still-active careers and time-varying
style/load features.

The output surfaces *hypotheses for trained expert judgment*. The staff-facing
framing is what makes the prescriptive claim defensible at all; a consumer-facing
version would be far more fraught.

## 2. Data and its honest limitations

| Source | Role | Honest limitation |
|---|---|---|
| prosportstransactions injury log | labels (IL placement/return + reason strings) | records **availability**, not clean medical diagnoses |
| nba_api game logs + advanced | Tier-1 features (full coverage) | game-level load, not training/biomechanical load |
| nba_api player tracking | Tier-2 features (~2016+, partial) | not every player-game; absence is modeled, not imputed |
| Basketball Reference | bio cross-check (age, position, height, mass) | static attributes; context, not levers |

**The ceiling is set by label quality.** Public data captures *workload and
availability*, never the imaging, biomechanics, treatment history, or practice
exposure a team actually holds. That ceiling is stated everywhere it matters, and
§9 spells out what private data would add.

- Label tally (Gate 1): `[N]` time-loss episodes over 10 seasons; per-category
  `[…]`; ambiguous-string rate `[X%]`.
- Coverage (Gate 2): Tier-1 complete, no silent gaps; Tier-2 availability mapped.

## 3. Method overview (the pipeline)

```
injury labels ── reason-string mapping ──┐
                                          ├── person-period table (1 row = player-week)
nba_api features + Tier-2 availability ──┘        │
                                                   ├── discrete-time hazard model (Stage 4)
                                                   ├── validation gauntlet (Stage 5)
                                                   ├── competing risks + recurrent (Stage 6)
                                                   ├── prescriptive lever layer (Stage 7)
                                                   └── monitoring / assurance (Stage 8)
```

Each stage has a **gate** — a concrete pass/fail check codified as tests. The
build fails fast on its riskiest dependency (label quality) before anything is
built on top.

## 4. Results — the hazard model (Stage 4)

Strictly temporal validation (train earlier seasons `[2015–22]`, test later
`[2022–25]`; never a random split). Metrics suited to rare events
(prevalence ≈ `[rate]`), **never accuracy**:

| Model | AUC-PR | ROC-AUC | Brier | lift vs constant |
|---|---|---|---|---|
| age + minutes baseline | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| logistic (Tier-1) | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| HistGradientBoosting (Tier-1+2) | `[ ]` | `[ ]` | `[ ]` | `[ ]` |

Calibration: `[describe the curve — where the model is over/under-confident]`.

> Gate 4 passes only if the model beats both a constant league-average hazard and
> an age+minutes baseline. It establishes that the features carry *some* signal —
> not that the model is strong. No claims rest on it until Stage 5.

## 5. Earning the right to make claims (Stage 5)

**Leakage audit:** structural causality checks (cumulative + prior-injury
monotonicity) plus a **target-shuffle test** — permuting training labels collapses
performance to baseline on a clean pipeline. Result: `[clean / inconclusive]`.

**Stylistic-comparables check (the anti-hindsight test):** for each index case
(Rose c. 2011, Zion current), build the cohort of *healthy* players nearest in
pre-injury style space, then ask whether the model ranks the index case above
them using pre-injury-only data.

| Index case | n healthy comparables | index hazard percentile | verdict |
|---|---|---|---|
| Rose | `[ ]` | `[ ]` | `[ ]` |
| Zion | `[ ]` | `[ ]` | `[ ]` |

> Gate 5 passes on **honest characterization**, not a performance bar. If the
> model can't separate an index case from its healthy look-alikes, that is
> reported plainly — the risk signal is style-driven and no "we'd have caught X"
> claim is made.

## 6. Depth: competing risks + recurrent events (Stage 6)

Non-injury career exits are modeled as their own competing cause rather than
ordinary censoring (which had inflated the injury incidence). Cause-specific
**calibrated** hazards combine into a cumulative incidence function over a
one-season horizon:

| Quantity | Value |
|---|---|
| CIF(injury) | `[ ]` |
| CIF(non-injury exit) | `[ ]` |
| Survival | `[ ]` |

A recurrent-event framing (every post-recovery at-risk week, plus
`weeks_since_last_injury`) replaces time-to-first-injury. PyDTS was wired in as an
academic cross-check; per the build plan's warning it hit version-rough-edges and
the pipeline fell back to the scikit-learn anchor — `[status]`.

## 7. The prescriptive layer, with causal humility (Stage 7)

Three jobs, each with a hard boundary:

1. **SHAP interpretation** — what the *model* attributes risk to (not what causes
   injury). `[top features, tagged actionable/fixed]`.
2. **Modifiable/fixed partition** — levers surfaced only on actionable in-game
   load (minutes, games-in-7, back-to-backs, usage, drive load). Fixed attributes
   (age, mass, prior injuries) are context. This honors the public-data ceiling:
   training/recovery are *not* in the data, so we don't advise on them.
3. **Counterfactual hazards** — "predicted risk if this lever differed, all else
   equal, under learned associations." An **overclaim guard refuses by
   construction** to surface a lever on any fixed feature.

**Observational validation:** do comparable players with lower lever values show
lower *realized* injury? Result: `[support in N/M strata / honest null]`. Every
output carries a standing caveat: *model-implied hypothesis, not advice.*

## 8. The assurance layer (Stage 8)

- **Covariate drift:** native PSI + KS (multi-test), multivariate shift.
- **Subgroup performance:** by era / position / age band / injury type.
- **Calibration drift:** Brier vs trivial bound per window (on a *calibrated*
  model).
- **Late-label two-mode:** estimate performance pre-labels (expected Brier) and
  confirm post-labels; monitor the gap.
- **Verifiable-event demo (COVID seasons):** `[drifted features]`; AUC-PR
  pre=`[ ]` → during=`[ ]`; verdict `[DRIFT WITHOUT/WITH DEGRADATION]`.

> The crux: the monitor separates *"inputs looked different"* from *"the model got
> worse."* Drift is not automatically failure — that distinction is the deliverable.

## 9. Limitations & what private data would add (the honesty that signals maturity)

- Public labels conflate load-management with true time-loss and lack diagnosis
  granularity.
- Only in-game style/load is observable — **not** training, sleep, nutrition,
  treatment, or biomechanical load, which is where injuries are most influenced.
- Rare severe injuries (Achilles) are too sparse for a standalone target; they
  are a spotlighted tail, not a modeled class.
- **With private data**, the same pipeline would gain true severity/mechanism
  labels, imaging and treatment history, GPS/biomechanical load, and practice
  exposure — the realistic source of the largest accuracy gains. The architecture
  is built to absorb these without redesign.

## 10. Reproducibility

- `./run_all.sh synthetic` runs the entire pipeline with no network (synthetic
  world, clearly flagged not-real); `./run_all.sh real` reproduces the actual
  analysis on a machine with open egress.
- Every external pull is cached and frozen (write-once); the modeling table is a
  single Parquet file; `pytest` (59 tests) covers the mapping layer, episode
  reconstruction, person-period integrity, temporal-split guarantees, the
  prescriptive overclaim guard, and the drift/degradation logic.
- Every stage has a gate; gates are designed to **fail on noise and pass on
  signal** (verified: on the synthetic noise fixture, Gates 4 & 6 correctly do
  not pass).

## 11. What this project demonstrates

A complete, honest, reproducible model-assurance pipeline on fully public data:
correct rare-event methodology, anti-hindsight validation, causal humility
enforced in code, and a monitoring layer that tells drift from degradation — the
same assurance spine that transfers directly to clinical/health-AI model
validation.

---
*See `reports/stage4`…`stage8` for the per-stage write-ups, `notebooks/` for the
runnable walkthroughs, and `DECISIONS.md` for the judgment-call log.*
