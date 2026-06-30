# Stage 5 — Validation Gauntlet & Honest Limits

> **Status:** the hindsight-bias / leakage gauntlet. Numbers in `[brackets]` are
> filled from the real run. This is the stage that earns (or withholds) the right
> to make claims — its job is to be skeptical of our own model.
>
> **Completing this stage marks the Minimum Viable Honest Project** (Stages 0–5):
> a credible portfolio artifact even if no further depth is added.

## 1. Why this stage exists

It is trivially easy to build a model that "would have predicted Derrick Rose,"
because the outcome is already known. That is hindsight bias, not prediction.
The real test has two parts, and we run both adversarially against ourselves:

1. **Temporal honesty:** prove no feature for player-week *W* secretly encodes
   information from week ≥ *W*.
2. **Stylistic comparables:** prove the model flags an index case (Rose c. 2011,
   Zion current) using **pre-injury-only** data **without** also flagging the
   many stylistically-similar players who stayed healthy.

## 2. Temporal honesty (leakage) audit

Three checks (`audit_leakage.py`):

- **Cumulative causality** — `cum_season_minutes` / `cum_season_games` must be
  non-decreasing within a (player, season) and week-1 must equal that week's own
  value. Result: `[PASS]`.
- **Prior-injury monotonicity** — `prior_injury_count` never decreases over a
  player's weeks. Result: `[PASS]`.
- **Target-shuffle test** — permute the training labels (destroying real signal)
  and re-evaluate on the real-labelled test set. A clean pipeline collapses to
  ~baseline; a model that still scores well is leaking. Result:
  - real AUC-PR `[ ]` vs shuffled `[ ]` ± `[ ]` → gap `[ ]`
  - verdict: `[real clearly beats shuffled — clean / inconclusive on this split]`

> The detector is itself unit-tested: planting a corrupt cumulative or a
> decreasing prior-injury count makes the audit fail, confirming it can see leaks.

## 3. Stylistic-comparables check

For each index case (`stylistic_comparables.py`):

1. Compute the index player's **pre-injury style profile** (usage, pace, minutes
   density, back-to-backs, cumulative load — the slowly-varying "kind of player"
   signal).
2. Build the **k nearest healthy look-alikes**: players closest in standardized
   style space who had **no** time-loss injury in the horizon. The cohort
   explicitly excludes injured players (unit-tested).
3. Score everyone with the temporally-trained model using **pre-injury weeks
   only**, and ask whether the index case's modeled hazard sits **above** its
   healthy comparables.

| Index case | n comparables | index max hazard | comparable median | comparable p90 | index percentile | verdict |
|---|---|---|---|---|---|---|
| Rose (c. 2011) `[id]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| Zion (current) `[id]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |

**Reading the verdict honestly:**
- *Index above its healthy look-alikes* → the case is a validation anecdote, not
  a cherry-pick. The model sees something beyond raw style.
- *Index NOT clearly above* → the risk signal is substantially **style-driven**;
  the model cannot distinguish this player from similar players who stayed
  healthy. We **say so** — no "we would have caught X" claim is made.

## 4. What the model can and cannot distinguish (the Gate-5 statement)

> Fill from the real run. State plainly, e.g.:
>
> - **Can:** rank at-risk player-weeks by modeled hazard above workload-only
>   heuristics, with `[calibration quality]` probabilities, and `[does/does not]`
>   place the index cases above stylistically-matched healthy players.
> - **Cannot:** `[e.g. separate high-mass explosive players who get hurt from
>   those who don't, on public data alone]` — this is a label/feature-resolution
>   limit, not a tuning problem.

## 5. Failure modes, documented plainly

- `[e.g. low-minute players have unstable hazard estimates]`
- `[e.g. Tier-2 absence pre-2016 weakens early-season profiles]`
- `[e.g. load-management vs true-injury ambiguity inflates/deflates certain
  categories]`

## 6. Gate 5 verdict

`[PASSED — claims are bounded by evidence; structural leakage clean; index-case
separation characterized honestly / FAILED — leakage present, fix first]`

Passing Gate 5 means **honesty about limits**, not hitting a performance bar. A
model that demonstrably can't beat stylistic comparables still passes — provided
this report says so. What does **not** pass is leakage, or any claim the evidence
doesn't support.

---
*This completes the Minimum Viable Honest Project (Stages 0–5). Stages 6–9 add
competing risks, the prescriptive layer, monitoring, and the final report.*
