# Decisions log (append-only)

Judgment calls made during the build. Future-you will thank present-you.

---

**2025 — Stage 0/1 scaffold**

- **Injury source = prosportstransactions.com injury log.** The canonical public
  NBA injury-history source (records IL movements + free-text reason strings).
  Chosen over ad-hoc community CSVs because it's the upstream of most of them and
  can be re-snapshotted. Records *availability*, not diagnoses — accepted per
  vision §4.

- **Snapshot-and-freeze.** `fetch_injuries` writes one frozen CSV to
  `data/raw/injuries_snapshot.csv`; all downstream code reads only that file.
  Reproducible even if the site changes/disappears.

- **Mapping layer in YAML, not code.** `reason_map.yaml` holds the dictionary so
  it can grow without code edits (the build plan expects heavy iteration here).

- **Category priority order matters.** Load-management stripped first; then
  `achilles` is checked before generic `lower_limb_soft_tissue` (else "achilles"
  gets swallowed); `knee_ligament` (ACL/MCL/meniscus) before generic `knee`.
  First match wins.

- **Gate-1 thresholds:** total ≥ 2000 episodes; each modeled category ≥ 200
  (Achilles excepted as the spotlighted rare tail); ambiguous rate ≤ 15%.
  Tunable in `build_labels.py`. These are the build-plan defaults.

- **Episode reconstruction:** pair each OUT row (relinquished + reason) with the
  next IN row (acquired) per player. Unclosed OUT = right-open episode (still out
  at snapshot end). `days_out` from the date gap is a *placeholder* — Stage 3
  replaces it with real games-missed from the schedule.

- **Network constraint observed.** In the build sandbox, stats.nba.com /
  basketball-reference / prosportstransactions / kaggle all returned 403 (egress
  allowlist). The two fetch steps must run on an open network; everything else is
  offline-reproducible. A `make_synthetic_snapshot` dev fixture lets the pipeline
  be exercised without network — flagged loudly as NOT real data.

<!-- Add new entries below, newest last. -->

---

**Stage 2 — feature pull**

- **Player universe via `commonallplayers` per season**, not all-time. Manifest
  = every (player_id, season) active in the 10-season window; it's the resumable
  work list. Re-running skips cached keys.

- **Two-tier feature design enforced at pull time.** Tier-1 (gamelogs + advanced)
  pulled for every (player, season). Tier-2 tracking pulled per season with an
  EXPLICIT per-(player,season) availability boolean per measure
  (speed_distance, drives, defense, rebounding, possessions) → never silently
  dropped. Written to `data/raw/tracking_availability.json`.

- **All pulls go through `nba_client`** (0.7s pacing, exp backoff + jitter, 5
  retries). Treated as an overnight job; safe to Ctrl-C and resume from cache.

- **Basketball Reference = independent cross-check/gap-fill** for age/pos/
  height/weight, snapshotted and frozen (paced ~3.5s; BBRef blocks fast
  scrapers). nba_api `commonplayerinfo` remains the primary bio source.

- **Gate-2 audits the cache on disk, no network.** Passes when Tier-1 has zero
  silent gaps and the Tier-2 availability map exists. Verified end-to-end on a
  synthetic cache (network blocked in sandbox); real run pending on open network.

---

**Stage 3 — person-period dataset (the expensive-to-undo shape)**

- **One row = one player-week, Monday-anchored ISO weeks, active weeks only.**
  Grid runs from the week of a player's first game to the week of their last
  game per season. No off-season or pre-debut rows.

- **Strictly causal features.** cum_season_minutes / cum_season_games accumulate
  through the current week only; games-in-7-days and back-to-backs are computed
  within the week bucket. Gate-3 audits monotonicity + week-1 == own-week to
  catch any future-info leak.

- **Event = injury episode BEGAN that week** (matched player+week). v1 is
  time-to-FIRST-injury: weeks inside a recovery interval are removed from the
  risk set (at_risk=0). Recurrence handling deferred to Stage 6.

- **An event week is always at_risk=1** — overrides any overlapping earlier
  recovery interval. Re-injury-during-recovery is a Stage-6 recurrence concern;
  for v1 the event week must be a valid at-risk row. (Caught by Gate-3's
  at-risk/event consistency check during the build.)

- **exit_type on each player's terminal week** for competing-risks groundwork:
  injury / active_end (censored, last season == final window season) /
  exit (non-injury career exit, last active season precedes the window end).

- **Tier-2 attached with explicit *_missing indicators**, NaN where absent —
  never silently imputed. Honors the modeled-missingness design from vision §13.

- **Bugfix:** _parse_date now tolerates pandas NaN/float blanks (open-ended
  episodes have empty end_date). Added regression test.

- Output: data/processed/person_period.parquet (canonical modeling table).
  Verified end-to-end on a coherent synthetic world (network blocked in
  sandbox); real build pending Stages 1-2 live pulls.

---

**Stage 4 — discrete-time hazard model**

- **Discrete-time hazard = binary classifier on the person-period table.** Target
  = event (injury began this week). Only at_risk==1 weeks enter the model (the
  time-to-first-injury risk set).

- **Two models by design:** logistic regression (Tier-1 only, needs impute+scale,
  interpretable baseline) and HistGradientBoosting (Tier-1 + Tier-2; consumes NaN
  natively so Tier-2 missingness can itself carry signal). Both use balanced
  class weights for the rare positive.

- **Strict temporal validation:** train earlier seasons, test the last 3. Never a
  random split (would leak future into past). Implemented in temporal_split().

- **Honest rare-event metrics only:** AUC-PR (headline), ROC-AUC, Brier,
  calibration curve, lift vs constant league-average-hazard. Accuracy is
  deliberately NOT reported (meaningless at ~5% prevalence).

- **Two baselines to beat:** constant league-average hazard, and age+minutes-only
  logistic. Gate 4 requires the best model to beat both (≥10% AUC-PR lift over
  constant, ≥5% relative AUC-PR gain over minimal).

- **Deliverable shape:** model_hazard.py (+ --report-json), gate4_report.py,
  notebooks/stage4_hazard_model.ipynb, reports/stage4_interim_report.md
  (bracketed placeholders, filled from the real run — no fabricated metrics).
  Data folders ship empty (.gitkeep); real pulls run on an open network.

- Verified end-to-end in-sandbox on a temporarily load-biased synthetic world
  (Gate 4 PASS, AUC-PR 2.2x baseline); the bias was reverted before shipping so
  the committed fixture is honest noise, not a signal-loaded demo.

---

**Stage 5 — validation gauntlet (hindsight-bias guard)**

- **Leakage audit (audit_leakage.py):** structural causality checks (cumulative
  monotonicity, prior-injury monotonicity) PLUS a target-shuffle test — permute
  training labels and confirm test performance collapses toward baseline. Leak
  suspicion is judged on the real-vs-shuffled GAP (real must clear shuffled +2sd),
  not shuffled-vs-prevalence, so it's robust on small samples. Detector is
  unit-tested by planting corrupt cumulatives / decreasing prior-injury counts.

- **Stylistic-comparables check (stylistic_comparables.py):** the core
  anti-hindsight test. Build each index case's PRE-injury style profile, find the
  k nearest HEALTHY look-alikes in standardized style space (cohort explicitly
  excludes injured players — unit-tested), score all with the temporally-trained
  model on pre-injury weeks only, and report the index case's hazard percentile
  vs its healthy comparables. Reusable for any (player_id); pass real Rose/Zion
  ids via --index-player on the real table.

- **Gate 5 = honesty, not a perf threshold.** A model that can't separate the
  index case from healthy comparables still PASSES — provided the report says so
  plainly. The only hard fail is structural leakage. "We would have caught X"
  claims are made ONLY when the index case clearly outranks its healthy
  look-alikes AND the shuffle test is clean.

- On the tiny synthetic split the shuffle test correctly returns inconclusive
  (~15 events can't distinguish signal from noise) — the audit reports this
  honestly rather than rubber-stamping. Real 10-season data has the volume to
  resolve it.

- Deliverable: audit_leakage.py, stylistic_comparables.py, gate5_report.py,
  notebooks/stage5_validation_gauntlet.ipynb, reports/stage5_validation_report.md.
  STAGES 0–5 = Minimum Viable Honest Project.

---

**Stage 6 — competing risks + recurrent events (depth)**

- **scikit-learn is the PRIMARY path, PyDTS the optional cross-check.** As the
  build plan predicted, PyDTS (0.1.0) has rough edges — its data-expansion step
  raises a pandas-compat ValueError on this version. The _try_pydts wrapper never
  raises: it attempts a TwoStagesFitter fit and falls back to the sklearn anchor
  with a logged reason. Verified the fallback fires cleanly.

- **Competing risks = cause-specific discrete-time hazards** (injury vs
  non-injury career exit), two classifiers sharing the temporal split, combined
  into a cumulative incidence function (CIF) over a ~26-week season horizon.

- **CALIBRATION, not ranking, governs the CIF.** Critical fix: cause-specific
  models are fit WITHOUT class_weight='balanced'. Balanced weighting inflates
  predicted hazards to ~0.5, destroying calibration and making the per-step CIF
  accumulation explode to survival=0. Without it, hazards are honest low
  probabilities and CIF_injury + CIF_exit + survival ≈ 1 (coherence verified).

- **CIF fix:** accumulate the population MEAN per-week cause-specific hazard over
  a fixed horizon, not across all heterogeneous test rows (which isn't a survival
  curve). Coherent, monotone, sums to 1.

- **Recurrent events:** keep all at-risk weeks (post-recovery re-entry), add
  causal weeks_since_last_injury; prior_injury_count carries cumulative burden.

- On the noise-only synthetic fixture, injury calibration sits ~at the trivial
  bound (no signal to beat base rate — correct behavior on noise); exit
  calibration passes clearly. Real data with signal will clear the bound.

- Deliverable: model_competing_risks.py, gate6_report.py,
  notebooks/stage6_competing_risks.ipynb, reports/stage6_competing_risks_report.md.

---

**Stage 7 — prescriptive / causal core (the heart; highest overclaim risk)**

- **Three jobs with hard boundaries, enforced in code:**
  - Job 1 (SHAP): explains what the MODEL attributes risk to, tagged
    actionable/fixed. shap.LinearExplainer (exact for the logistic pipeline) with
    a transparent fallback to standardized coefficients if SHAP unavailable.
  - Job 3 (partition): MODIFIABLE (in-game style/load levers) vs NON_MODIFIABLE
    (prior-injury count, pace, + bio context). Sets are disjoint (unit-tested).
    Honors the public-data ceiling: only in-game load is prescribable, NOT
    training/recovery (not in the data).
  - Job 2 (counterfactual): alter one modifiable feature all-else-equal, read the
    model's hazard delta. Framed permanently as "predicted risk IF this differed
    under learned associations" — never a promise.

- **Overclaim guard:** counterfactual_hazard REFUSES any non-modifiable feature
  by construction (returns an error, not a lever). Unit-tested as the single most
  important discipline check — surfacing a lever on age/prior-injury would be an
  overclaim by definition.

- **Observational validation:** stratify by coarse style profile, split each
  stratum at the lever median, compare realized event rates. Reports support OR
  an honest null — a null is a limitation, never buried. Explicitly labeled
  observational + confounded.

- **STANDING_CAVEAT wired into every prescriptive output.** Audience = team
  performance staff; outputs surface hypotheses for expert judgment, not advice.

- **Fixed a partition inconsistency:** games_this_week was mis-tagged fixed while
  games_in_7days was actionable (same kind of rotation lever) — now both
  modifiable.

- Deliverable: prescriptive.py, gate7_report.py,
  notebooks/stage7_prescriptive.ipynb, reports/stage7_prescriptive_report.md.

---

**Stage 8 — monitoring / assurance**

- **Native PSI/KS is the PRIMARY path; Evidently/NannyML optional.** The build
  plan warns "don't let Evidently dashboards become the project." PSI, KS, and
  NannyML-style expected-Brier are ~100 lines of well-defined statistics —
  implemented directly for full reproducibility. Evidently wired in as an
  optional cross-check (succeeds here, degrades gracefully if version-fragile).

- **Multi-test drift detection:** a feature is flagged drifted if PSI>0.1 OR
  KS>0.2. PSI alone returns NaN on near-disjoint ranges (fixed by extending bin
  edges to cover the analysis range), so KS provides a robust second signal — no
  single metric's edge case can hide real drift.

- **Calibration panels use a CALIBRATED (unweighted) model, drift/subgroup use
  the balanced (ranking) model.** Same balanced-vs-calibrated tradeoff from
  Stage 6: balanced weighting inflates probabilities to ~0.5, which would make
  the calibration panel look far worse than reality. Two scores (_pred, _pred_cal)
  computed; each panel uses the right one.

- **Late-label two-mode:** Mode A estimates Brier pre-labels from the model's own
  probabilities (expected Brier = mean p(1-p)); Mode B is the realized Brier
  post-labels; the gap is monitored.

- **Verifiable-event demo = COVID seasons (2019-20, 2020-21).** Separates DRIFT
  WITHOUT DEGRADATION (inputs shifted, model held — harmless) from DRIFT WITH
  DEGRADATION (inputs shifted AND performance dropped — retrain signal). This
  drift!=failure distinction is the Gate-8 deliverable.

- **Bugfix (found via tests):** HistGradientBoosting crashes on a 100%-NaN column
  (sklearn 1.9 binning). fit_hgb now drops all-NaN columns (they carry no signal;
  real Tier-2 always has partial coverage). Also hardened _xy to return writable
  contiguous arrays. Test updated to use partial-NaN (realistic) not 100%-NaN.

- Deliverable: monitoring.py, gate8_report.py,
  notebooks/stage8_monitoring.ipynb, reports/stage8_monitoring_report.md.

---

**Stage 9 — written report + reproducibility (the deliverable)**

- **FINAL_REPORT.md** ties all stages into one self-contained portfolio artifact:
  framing, data limitations, method pipeline, temporal results, the validation
  gauntlet, competing risks, the prescriptive layer with caveats, the monitoring
  demo, and an explicit "limitations & what private data would add" section.

- **run_all.sh** = one-command reproducibility, synthetic (no network) or real.
  Verified end-to-end from a clean state: all gates fire, 59 tests pass.

- **Gates fail on noise, pass on signal — by design.** On the shipped synthetic
  (noise-only) fixture, Gates 4 & 6 correctly DO NOT pass; this is documented in
  run_all.sh output and the README so a reviewer isn't confused. It's the
  integrity check working, not a defect.

- **RESUME_BULLETS.md** drafted with placeholders across three role framings
  (health-AI/ML-validation, data-engineering, statistical-rigor). No fabricated
  metrics — filled from the real run only. Gate 9 verifies placeholders remain.

- **Gate 9 = deliverable completeness** (not a model check): final report covers
  all sections, every stage has report+notebook+gate, reproducibility entrypoint
  + README + DECISIONS + pinned requirements present, resume bullets intact, test
  suite present. PASSED.

- PROJECT COMPLETE (Stages 0–9). Optional Stage 10 = live GitHub Actions
  monitoring job + Streamlit frontend (deferred by design).

---

**Stage 10 — live job + frontend (optional, deferred by design)**

- **live_score.py = inference + monitoring ONLY, never retrains.** Scores the
  current week's players with the frozen model; writes reports/live/latest.json +
  appends history.jsonl (rolling drift record accumulates in the repo, no DB).
  Demo mode (--demo) samples recent at-risk rows so it runs without network.

- **Graceful degradation is the key design choice for a scheduled job:** if
  stats.nba.com is unreachable, it records fetch_failed and exits 0 — never
  crashes the workflow. Verified with no network.

- **GitHub Actions workflow** runs weekly (Mon 13:00 UTC) + on-demand, commits
  the rolling record back with contents:write. No secrets needed.

- **dashboard.py = thin presentation layer over Stage 4-8 logic.** No modeling in
  the UI, so it can never disagree with the analysis. Tabs: player risk curve,
  prescriptive levers (with standing caveat), monitoring, about/limits. Offers to
  build a synthetic world if no parquet exists.

- **Bugfix:** itertuples renames columns starting with '_' (e.g. _pred), breaking
  attribute access. Renamed to pred_hazard.

- Phase-2 / lightweight by design (vision doc): the credibility lives in
  Stages 0-9; this only makes the work demonstrable. Gate 10 PASSED.

- PROJECT FULLY COMPLETE (Stages 0-10). 63 tests passing.
