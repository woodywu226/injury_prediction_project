# Stage 6 — Competing Risks & Recurrent Events

> **Status:** the graduate-level depth layer. Numbers in `[brackets]` are filled
> from the real run. This stage removes two biases baked into the Stage-4 model.

## 1. Why Stage 4 wasn't enough

The Stage-4 discrete-time hazard model made two simplifying assumptions that are
fine for a first pass but bias the conclusions:

1. **It treated every non-injury exit as ordinary censoring.** When a player is
   cut, retires, or ages out, Stage 4 records them as "still at risk, just not
   yet injured." That inflates the injury cumulative incidence — a retired player
   is not at risk of an NBA injury forever.
2. **It modeled time-to-FIRST-injury only.** Real careers have many injuries, and
   a prior injury changes future risk. Stage 4 left that recurrent structure on
   the table.

## 2. Competing risks

We estimate **cause-specific discrete-time hazards** — one model per cause,
sharing the temporal-validation split:

- `h_injury(t)` = P(a time-loss injury begins at week *t* | at risk)
- `h_exit(t)` = P(a non-injury career exit at week *t* | at risk)

and combine them into a **cumulative incidence function (CIF)** over a one-season
horizon (≈26 active weeks):

| Quantity | Value |
|---|---|
| CIF(injury) | `[ ]` |
| CIF(non-injury exit) | `[ ]` |
| Survival (neither) | `[ ]` |
| Sum (coherence check ≈ 1) | `[ ]` |

> **Calibration matters here, not ranking.** We deliberately fit these
> cause-specific models *without* balanced class weights: balanced weighting
> improves AUC-PR but inflates the predicted hazards to ~0.5, which destroys
> calibration and makes the CIF accumulation explode. The CIF needs *honest
> probabilities*, so calibration (Brier below the trivial `p(1−p)` bound) is the
> governing metric.

| Cause | AUC-PR | ROC-AUC | Brier | trivial bound | calibrated? |
|---|---|---|---|---|---|
| injury | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| non-injury exit | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |

## 3. Recurrent events

The recurrent-event framing keeps **every at-risk week** (post-recovery weeks
re-enter the risk set), not just the weeks up to a player's first injury. A new
causal covariate, `weeks_since_last_injury`, lets the model express elevated
(or recovered) risk following a prior injury. `prior_injury_count` (already
causal from Stage 3) carries the cumulative burden.

## 4. PyDTS cross-check (academic competing-risks package)

The build plan flagged PyDTS for "academic-package rough edges." That played out
exactly: `[describe — e.g. PyDTS's data-expansion step raised a pandas
compatibility error on this version]`. The implementation wires PyDTS in
**defensively** — it attempts a `TwoStagesFitter` competing-risks fit and, on any
failure, logs the reason and falls back to the scikit-learn anchor. The anchor is
the reliable primary path; PyDTS is a cross-check, not a dependency.

PyDTS status on this run: `[fit_ok / failed: <reason> / unavailable]`.

## 5. What the depth buys (the Gate-6 articulation)

- **Unbiased longevity estimates.** Once non-injury exits are modeled as their
  own cause rather than censoring, the injury CIF is no longer inflated by
  players who simply left the league. `[quantify the difference vs the Stage-4
  naive estimate if computed]`.
- **Recurrent risk.** The model now speaks to career-long injury burden and
  post-injury risk elevation, not just "will this player ever get hurt."

## 6. Gate 6 verdict

`[PASSED — cause-specific models calibrated, CIF coherent, depth articulated /
NOT PASSED — calibration or CIF coherence needs work]`.

---
*Next: Stage 7 — the prescriptive / causal core (SHAP, modifiable-lever
partition, counterfactual hazards under stated assumptions). The highest
overclaim-risk stage; humility is designed in.*
