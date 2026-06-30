# Stage 7 — Prescriptive / Causal Core

> **Status:** the heart of the project — and the highest overclaim-risk stage.
> Numbers in `[brackets]` are filled from the real run. **Causal humility is
> non-negotiable here, and it is a credibility *strength*, not a hedge.**
>
> **Standing caveat (wired into every prescriptive output):**
> *Model-implied hypothesis, not advice. Surfaces associations the model
> considers actionable, for trained performance/sports-science staff to weigh
> against everything the model cannot see (private biomechanical, training, and
> medical data). SHAP explains the MODEL, not real-world causation.
> Counterfactuals are "all-else-equal under learned associations," not promises
> that an intervention reduces injury.*

## 1. The three jobs and their hard boundaries

| Job | What it does | The boundary it must not cross |
|---|---|---|
| **1 — Interpretation** | SHAP shows what the **model** attributes a player's risk to | NOT "what causes injury in the world" |
| **3 — Lever partition** | split features into actionable vs fixed; surface levers only on actionable | fixed features (age, mass, prior injuries) are context, never levers |
| **2 — Counterfactual** | predicted hazard if one modifiable feature differed, all else equal | NOT a promise the intervention works |

## 2. Job 1 — what the model attributes risk to (SHAP)

Mean |SHAP| per feature on the hazard model (`shap.LinearExplainer`, exact for
the logistic pipeline; transparent fallback to standardized coefficients if SHAP
is unavailable). Each feature is tagged actionable or fixed.

| Feature | mean |SHAP| | partition |
|---|---|---|
| `[ ]` | `[ ]` | `[actionable/fixed]` |
| … | … | … |

> Read as: *"the model leans on these features when assigning risk."* It is
> **not** a claim that these features cause injuries. A marker (e.g. high drive
> frequency) can be a *signature* of an injury-prone profile rather than an
> intervenable cause — Job 2 handles that trap explicitly.

## 3. Job 3 — modifiable / non-modifiable partition

- **Modifiable (lever-able):** in-game style/load — minutes, games this week,
  games-in-7-days, back-to-backs, usage, cumulative season minutes, drive load,
  speed/distance.
- **Fixed (context only):** prior-injury count, team pace, plus bio attributes
  (age, height, mass, position) where joined.

This partition also honors the **public-data ceiling**: only in-game style/load
levers are prescribable. Training and recovery — the levers a team most wants —
are **not in public data**, so we do not pretend to advise on them.

## 4. Job 2 — counterfactual hazards (model-implied, all-else-equal)

For a focus player-week, each modifiable lever is reduced (e.g. −20%) with all
else held fixed, and the model's hazard delta is read off. The overclaim guard
**refuses** counterfactuals on fixed features by construction.

Example (player `[id]`, week `[date]`, modeled hazard `[ ]`):

| Lever | from → to | Δ modeled hazard |
|---|---|---|
| `[ ]` | `[ ] → [ ]` | `[ ]` |

> Each row reads: *"if this in-game load feature had been lower, all else equal,
> the model would have assigned this much less risk."* That is a hypothesis about
> the **model**, offered for expert review — not evidence that changing it
> reduces real injuries.

## 5. Observational validation of the prescriptive layer

The graduate-caliber bar: do comparable-profile players who actually had **lower**
lever values show **lower realized** injury? We stratify by a coarse style
profile, split each stratum at the lever median, and compare realized event
rates.

- Top modifiable lever: `[ ]`
- Result: `[observational support in N/M comparable strata / NO clear support]`

> This is observational and confounded, and we **say so**. Support upgrades a
> lever from "SHAP plot" to "hypothesis with observational backing and stated
> limits." A null result is reported as a **limitation**, not buried — and it is
> the honest outcome the model owes the staff who'd act on it.

## 6. The Rose / Zion narrative (real run)

For each case study: `[here are the model-implied modifiable levers at the
pre-injury focus week, with the standing caveat and the observational check —
stated as hypotheses for staff, not as "we would have prevented it"]`.

## 7. Gate 7 verdict

`[PASSED — outputs produced, partitioned, caveated, observationally checked;
overclaim guard holds / FAILED — a discipline check failed]`.

Passing Gate 7 is about **discipline**: caveats present, levers only on
actionable features, the guard refusing fixed-feature counterfactuals, and honest
reporting of observational support or its absence. Nothing crosses into causal
overclaim.

---
*Next: Stage 8 — monitoring / assurance (drift, subgroup performance, calibration
over time, late-label estimation, a verifiable-event demo).*
