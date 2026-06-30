# NBA Injury-Risk Decision Support — Interim Report (through Stage 4)

> **Status:** modeling pipeline complete through the discrete-time hazard baseline.
> Numbers in `[brackets]` are placeholders to be filled from the real run — this
> template is committed before real results exist by design (results-first resume
> bullets, no fabricated metrics).
>
> **Audience:** team performance / sports-science staff, as *decision support* —
> explicitly **not** autonomous medical advice.

## 1. Problem framing

We model **weekly injury onset** as a discrete-time hazard: for each at-risk
player-week, estimate the probability that a time-loss injury begins that week.
This framing turns a survival problem into a binary-classification problem on a
person-period table (one row per player-week), which keeps the modeling honest
and the temporal structure explicit.

The intended use is conversation-starting risk surfacing for staff who already
hold the clinical context — never a standalone verdict on a player.

## 2. Data and its honest limitations

- **Injury labels:** public injury-transaction log (prosportstransactions),
  snapshotted and frozen. This records **availability** (placement on / return
  from the inactive list), *not* clean medical diagnoses. Reason strings are
  normalized by a transparent mapping layer into categories (lower-limb
  soft-tissue, knee-ligament, back, hand/finger, concussion, illness, …) with an
  explicit `ambiguous` flag for the unclassifiable remainder.
- **Features:** nba_api game logs + advanced stats (Tier-1, full coverage),
  plus player-tracking endpoints (Tier-2, available ~2015-16 onward and **not**
  for every player-game). Tier-2 absence is modeled as an explicit missingness
  indicator, never silently imputed.
- **Key limitation:** public data captures *workload and availability*, not the
  internal medical signals (imaging, biomechanics, treatment history) a team
  actually holds. The honest ceiling of this project is set by label quality;
  the limitations section (§7) states what private data would add.

**Label tally (Gate 1):** `[N]` time-loss episodes across `[10]` seasons;
per-category counts `[…]`; ambiguous-string rate `[X%]`.

**Coverage (Gate 2):** Tier-1 complete with no silent gaps; Tier-2 availability
mapped per player-season (`[summary]`).

## 3. The person-period table (Stage 3)

`[R]` player-weeks across `[P]` players. Weekly grid is Monday-anchored,
active-weeks-only. Every feature for week *W* uses only information available
before *W* (cumulative load, prior-injury count, rolling games). Event = an
injury episode *began* that week; recovery weeks leave the risk set
(time-to-first-injury for v1). Each player's terminal week carries a
competing-risks exit type (injury / active_end / non-injury exit).

Integrity checks (Gate 3): unique player-week, event↔episode alignment,
censoring correctness, and a causal-feature audit — all `[PASS]`.

## 4. Models and validation (Stage 4)

Two deliberately simple models on the person-period table:

1. **Logistic regression** (Tier-1 features) — interpretable baseline with
   median imputation, scaling, and balanced class weights.
2. **HistGradientBoosting** (Tier-1 + Tier-2) — handles Tier-2 NaN missingness
   natively, so the missingness pattern itself can carry signal.

**Validation is strictly temporal:** train on the earlier seasons
(`[2015-16 … 2021-22]`), test on the later ones (`[2022-23 … 2024-25]`). No
random splits — a random split would leak the future into the past.

**Metrics suited to rare events** (prevalence ≈ `[rate]`):

| Model | AUC-PR | ROC-AUC | Brier | Lift vs constant hazard |
|---|---|---|---|---|
| age + minutes baseline | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| logistic (Tier-1) | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| HistGradientBoosting | `[ ]` | `[ ]` | `[ ]` | `[ ]` |

Accuracy is intentionally **not** reported — at this prevalence a model that
predicts "never injured" scores ~`[1-rate]` accuracy while being useless.

**Calibration:** `[describe the calibration curve — over/under-confident where?]`

## 5. Gate 4 verdict

The best model beats both the constant league-average-hazard predictor and the
age+minutes-only baseline on AUC-PR: `[PASS / NOT PASS, with the lift figure]`.
This establishes only that the features carry *some* signal beyond the trivial —
the point of Stage 4. It is not yet a strong model, and no claims rest on it
until the Stage-5 validation gauntlet.

## 6. What this can and cannot say (so far)

- **Can:** rank at-risk player-weeks by modeled hazard better than workload-only
  rules of thumb, with calibrated probabilities `[to the extent §4 shows]`.
- **Cannot (yet):** attribute risk to specific causes, distinguish a genuinely
  high-risk player from a stylistically-similar healthy one (that is exactly the
  Stage-5 test), or support any causal/prescriptive claim.

## 7. Limitations & what private data would add

Public availability data conflates load-management with true time-loss; lacks
diagnosis granularity; and omits the medical/biomechanical signals teams hold.
With private data, the same pipeline would gain: true injury severity and
mechanism labels, imaging/treatment history, GPS/biomechanical load, and
practice (not just game) exposure — which is where the largest accuracy gains
would realistically come from.

## 8. Reproducibility

Every stage is a CLI module with a gate check; all external pulls are cached and
frozen; the modeling table is a single Parquet file; `pytest` covers the mapping
layer, episode reconstruction, person-period integrity, and the temporal-split
guarantees. See `README.md` to run end to end.

---
*Next: Stage 5 — leakage audit and the stylistic-comparables check on the
Rose/Zion cases against cohorts of healthy look-alikes.*
