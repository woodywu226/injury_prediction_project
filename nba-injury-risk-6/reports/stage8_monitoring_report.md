# Stage 8 — Monitoring / Assurance Layer

> **Status:** the clinical-governance signal — the component this whole project's
> "AI assurance" spine was built to demonstrate. Numbers in `[brackets]` are
> filled from the real run.
>
> **The core idea:** a risk model that silently degrades is worse than no model.
> This layer keeps it honest over time and — crucially — **distinguishes harmless
> input drift from real performance degradation**. Drift is not automatically
> failure.

## 1. Why this matters most for a rare-event model

Severe injuries (especially Achilles) are sparse. Sparse signal is fragile: a
model can look fine on aggregate while quietly failing on a subgroup or drifting
out of calibration. The monitoring layer's job is to **refuse to over-trust
sparse signal** and to surface degradation early — the fragility of rare-event
risk models is itself the motivation for the assurance layer.

## 2. Covariate drift (PSI + KS, multivariate)

Reference = the training-era feature distribution; analysis = a later window.
Per feature we compute:

- **PSI** (population stability index): <0.1 negligible, 0.1–0.25 moderate,
  >0.25 significant. Made robust to disjoint/shifted ranges (edges extended to
  cover the analysis range so a shifted feature never returns NaN).
- **KS** (Kolmogorov–Smirnov): max CDF gap, dependency-free.
- **Multivariate** mean standardized shift across features.

A feature is flagged drifted if **PSI > 0.1 OR KS > 0.2** — a multi-test design
(like Evidently) so no single metric's edge case silently hides real drift.

| Feature | PSI | KS | drifted? |
|---|---|---|---|
| `[ ]` | `[ ]` | `[ ]` | `[ ]` |

## 3. Subgroup performance

Global metrics hide localized failure. We break AUC-PR and event rate down by
era/season (and, where bio is joined, position band and age band; by injury type
where labels allow). `[summary — which subgroups, if any, underperform the
global model]`.

## 4. Calibration drift over time

Per time window: Brier vs the trivial `p(1−p)` bound, and the reliability gap
(mean predicted − observed). **These panels use a CALIBRATED model** (unweighted
logistic), not the balanced-weight ranking model — balanced weighting inflates
probabilities and would make calibration look far worse than it is (the same
tradeoff identified in Stage 6). `[which windows, if any, show calibration
slip]`.

## 5. Late-label two-mode design

Injuries are confirmed only after games are missed, so labels lag in production.

- **Mode A (pre-labels, NannyML-style):** estimate performance from the model's
  own probabilities — expected Brier `= mean p(1−p)` — *before* any labels
  arrive.
- **Mode B (post-labels):** the realized Brier once labels land.
- The **gap** between A and B is itself monitored: a persistent gap signals the
  model is mis-estimating its own confidence, which pure label-lag would hide.

| Window | est. Brier (pre) | actual Brier (post) | gap |
|---|---|---|---|
| `[ ]` | `[ ]` | `[ ]` | `[ ]` |

## 6. Verifiable-event demo — the COVID seasons

The 2019-20 bubble and 2020-21 compressed seasons sit inside the window and are
the perfect natural drift event (schedule density, no-travel bubble, condensed
back-to-backs). Reference = pre-COVID seasons; analysis = COVID seasons.

- **Drifted inputs:** `[which features moved]`
- **Performance:** AUC-PR pre = `[ ]` → during = `[ ]`
- **Verdict:** `[DRIFT WITHOUT DEGRADATION / DRIFT WITH DEGRADATION]`

> This is the crux and the Gate-8 deliverable: the monitor separates *"inputs
> looked different"* from *"the model got worse."* If inputs drifted but
> performance held, that's **harmless drift** — no retrain. If both, that's the
> **retrain-with-care** signal. A monitor that cried wolf at every drift would be
> useless; the value is in telling the two apart.

## 7. Optional Evidently cross-check

The native PSI/KS implementation is the **primary, reproducible** path (the build
plan warns against letting Evidently dashboards become the project). An Evidently
`DataDriftPreset` is wired in as an optional cross-check that degrades gracefully
if the package or its version pins aren't available. Status on this run: `[ok /
failed / unavailable]`.

## 8. Gate 8 verdict

`[PASSED — the monitor distinguishes harmless drift from degradation on the COVID
event with a clear narrative / NOT PASSED — a capability is missing]`.

---
*Next: Stage 9 — the written report that turns the whole pipeline (labels →
person-period → hazard model → validation → competing risks → prescriptive →
monitoring) into the portfolio artifact.*
