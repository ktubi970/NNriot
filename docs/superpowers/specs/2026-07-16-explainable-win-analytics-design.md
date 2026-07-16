# Explainable Win Analytics and French Chatbot Design

**Date:** 2026-07-16

**Status:** Approved design, pending written-spec review

**Scope:** Phase 1 retrospective analytics over final Match-V5 data
**Deferred:** Timeline-derived 5/10/15-minute analysis

## 1. Purpose

NNriot will gain an explainable analytics subsystem that answers retrospective
questions about how matches are won. Statistical analysis is the source of truth.
An interpretable model and the existing neural network may discover interactions,
segments, and anomalies, but neither may create an evidence-backed claim without
independent statistical validation.

The user-facing result is an interactive French chatbot. It translates natural
language into a validated analysis request, executes allowlisted computations, and
renders a readable recommendation with traceable evidence. It must distinguish:

- an observed association;
- a model prediction;
- a causal claim, which phase 1 does not make.

The intended wording is prudent: for example, "prioriser X semble pertinent dans
cette cohorte," never "X garantit la victoire."

## 2. Audited Starting Point

The design is based on a read-only audit of the current repository and database:

- `training_data.db` contains 77,136 matches and 69,906 labeled training rows;
- 68,873 training rows are `CLASSIC` matches;
- the corpus contains 6,924 registered players and 173 champions;
- approximate training-match distribution is NA 21,980, EUW 20,324, EUN 17,341,
  KR 10,211, and 50 other/unknown;
- every one of 10,000 audited CLASSIC matches contained kills, deaths, assists,
  gold, champion damage, vision score, lane and jungle farm, champion, role, PUUID,
  challenges, and team objectives;
- `match_timelines` contains zero rows;
- all 69,906 training rows are marked trained, so the current evaluator has no
  untouched holdout rows;
- the current NN feature extractor retains only champion, KDA, final gold, role,
  version, mode, season, and optional region;
- the current 22-head model has useful signal on some outputs, but several rare
  heads remain at the majority baseline and all first-to-N timeline labels are
  absent;
- a read-only scan and decompression of 10,000 raw matches took about 82 seconds.

The corpus is rich enough for retrospective analytics but is not queryable enough
for interactive chat. Current winner accuracy is not evidence of a pre-match
prediction capability because the inputs contain final-game outcome proxies and
some reported evaluations reuse trained rows.

## 3. Goals and Non-goals

### 3.1 Phase 1 goals

1. Materialize a rebuildable analytical database separate from the operational DB.
2. Support named final-game features: champion, role, KDA, kills, deaths, assists,
   gold, lane and jungle farm, champion damage, vision, and team objectives.
3. Define comparable cohorts by mode, queue, patch, period, and region.
4. Produce deterministic evidence packets with statistical uncertainty and
   provenance.
5. Detect robust inter-region differences automatically.
6. Add an interpretable model for conditional effects and selected interactions.
7. Use the existing NN only as an auxiliary candidate generator.
8. Add a constrained French-language chat API and web UI.
9. Abstain instead of fabricating unsupported, underpowered, or causal claims.
10. Preserve all existing collection, training, prediction, and API behavior.

### 3.2 Non-goals

- No 5/10/15-minute state analysis until timelines are populated.
- No claim that a post-match association is a causal intervention.
- No pre-match performance claim from final-game features.
- No esports-organization or stable-team claim without a reliable team identity.
- No arbitrary model-generated SQL.
- No free-form LLM arithmetic or uncited numeric statement.
- No replacement of `training_data.db` or the existing prediction model.
- No unrelated refactoring of collectors, trainers, or existing pages.

## 4. Chosen Architecture

The chosen variant is a separate SQLite analytical mart named `analytics.db`.
It is a derived artifact, ignored by Git, and can be rebuilt from
`training_data.db` at any time.

The top-level flow is:

```text
training_data.db (read-only source)
  -> analytical extractor
  -> analytics.db
  -> cohort resolver
  -> statistical engine
  -> evidence registry
  -> constrained chat planner
  -> deterministic evidence renderer + qualitative LLM narrative
```

The existing Flask application exposes the new API and page, but the analytics
package does not import Flask. The existing NN path does not import analytics.

## 5. Component Boundaries

The implementation will use a focused `analytics/` package. Exact filenames may
change in the implementation plan, but the boundaries are fixed.

### 5.1 Analytical extractor

Responsibilities:

- open `training_data.db` in read-only/query-only mode;
- decode each raw match once;
- validate Match-V5 structure and include only `CLASSIC` matches;
- normalize match, team, and participant facts;
- record field-level extraction errors without aborting the entire backfill;
- support idempotent full rebuild and incremental update by `match_id`;
- calculate a deterministic dataset version;
- never mutate the operational database.

The extractor version is part of the dataset version. A changed extractor version
requires a rebuild so evidence never silently mixes incompatible feature logic.

### 5.2 Cohort resolver

Responsibilities:

- validate filters against an allowlist;
- resolve default mode, queue, patch window, period, and regions;
- return sample counts before expensive analysis;
- widen only the patch window when necessary and disclose that widening;
- refuse unsupported team-organization, timeline, or causal questions.

### 5.3 Statistical engine

Responsibilities:

- execute registered analysis types over named fields;
- avoid counting the same match as two independent observations;
- compute baseline, estimate, effect size, confidence interval, and significance;
- correct multiple comparisons within each candidate-analysis family;
- run temporal and regional robustness checks;
- return structured results, never prose.

### 5.4 Interpretable model

The primary interpretable model is a regularized logistic regression over named,
documented feature deltas. Pre-registered interactions may cover region, patch,
role, and a small set of gameplay interactions. It supplies conditional estimates,
calibration, and sensitivity analysis. It does not replace paired descriptive
statistics.

A nonlinear exploratory model may be added only if its explanations can be mapped
to named features and validated by the statistical engine. It cannot directly
produce user-facing evidence.

### 5.5 Neural-network auxiliary adapter

The existing NN may provide:

- embedding-based match segments;
- anomalous matches or cohorts;
- candidate nonlinear interactions;
- agreement/disagreement signals against the interpretable model.

An NN-derived candidate becomes user-visible only after the statistical engine
validates it on a cohort that was not used to discover the candidate. The current
checkpoint must be labeled as discovery-only because its inputs are hashed,
post-match, and trained on all currently marked rows.

### 5.6 Evidence registry

The evidence registry stores immutable evidence JSON keyed by dataset version,
analysis request hash, method version, and random seed. Repeating the same request
against the same dataset must reproduce the same evidence packet.

### 5.7 Chat planner and renderer

The planner converts French natural language into a typed `AnalysisRequest`. It
cannot emit SQL, Python, field names outside the registry, or unbounded filters.
The executor validates the request before accessing analytics.

The numeric evidence section is rendered deterministically from evidence packets.
An optional LLM adapter may add a qualitative French narrative, but that narrative:

- receives evidence packets rather than raw database access;
- may not introduce digits, percentages, counts, or new factual comparisons;
- must cite evidence IDs when referring to a finding;
- is rejected and replaced by the deterministic fallback if validation fails.

The runtime adapter is provider-neutral. Tests use a fake adapter. A deterministic
French intent parser and evidence renderer remain available when no LLM credential
is configured, so analytics never depends on an external service.

## 6. Analytical Database Schema

### 6.1 `analytics_meta`

```text
key TEXT PRIMARY KEY
value TEXT NOT NULL
```

Required keys include schema version, extractor version, source schema version,
source match count, maximum source creation time, build time, and dataset version.

### 6.2 `match_facts`

One row per CLASSIC match:

```text
match_id TEXT PRIMARY KEY
region TEXT NOT NULL
platform_prefix TEXT NOT NULL
game_version TEXT NOT NULL
patch TEXT NOT NULL
queue_id INTEGER
game_creation_ms INTEGER NOT NULL
game_duration_s INTEGER NOT NULL
winner_team_id INTEGER NOT NULL CHECK (winner_team_id IN (100, 200))
participant_count INTEGER NOT NULL
source_season_id INTEGER
```

Indexes cover `(patch, region, queue_id)`, creation time, and region.

### 6.3 `team_facts`

Exactly two rows per valid match:

```text
match_id TEXT NOT NULL
team_id INTEGER NOT NULL CHECK (team_id IN (100, 200))
side TEXT NOT NULL CHECK (side IN ('BLUE', 'RED'))
win INTEGER NOT NULL CHECK (win IN (0, 1))
kills INTEGER NOT NULL
deaths INTEGER NOT NULL
assists INTEGER NOT NULL
gold INTEGER NOT NULL
champion_damage INTEGER NOT NULL
vision_score REAL NOT NULL
lane_cs INTEGER NOT NULL
jungle_cs INTEGER NOT NULL
barons INTEGER NOT NULL
dragons INTEGER NOT NULL
towers INTEGER NOT NULL
inhibitors INTEGER NOT NULL
rift_heralds INTEGER NOT NULL
atakhan INTEGER NOT NULL
first_blood INTEGER NOT NULL
first_baron INTEGER NOT NULL
first_dragon INTEGER NOT NULL
first_tower INTEGER NOT NULL
first_inhibitor INTEGER NOT NULL
PRIMARY KEY (match_id, team_id)
```

The schema may add other consistently available final-game metrics, but every added
field must have an extraction definition and test.

### 6.4 `player_facts`

One row per participant:

```text
match_id TEXT NOT NULL
puuid TEXT NOT NULL
team_id INTEGER NOT NULL
side TEXT NOT NULL
win INTEGER NOT NULL
champion TEXT NOT NULL
role TEXT NOT NULL
kills INTEGER NOT NULL
deaths INTEGER NOT NULL
assists INTEGER NOT NULL
gold INTEGER NOT NULL
champion_damage INTEGER NOT NULL
vision_score REAL NOT NULL
lane_cs INTEGER NOT NULL
jungle_cs INTEGER NOT NULL
PRIMARY KEY (match_id, puuid)
```

Indexes cover PUUID, champion, role, and `(champion, role, patch)` through joins.
No player name is required for statistical identity; display-name lookup remains in
the operational DB.

### 6.5 `extraction_issues`

Stores match ID, stable issue code, field, and extractor version. It must not store
secrets or arbitrary exception payloads.

### 6.6 `evidence_cache`

Stores evidence ID, dataset version, request hash, method version, evidence JSON,
and creation time. A changed dataset or method version invalidates reuse.

## 7. Cohort Semantics

### 7.1 Default cohort

If the user supplies no filters, the resolver uses:

- `game_mode = CLASSIC`;
- ranked solo queue (`queue_id = 420`), because it is the dominant comparable
  queue in the audited sample;
- the most recent patch with at least 500 eligible matches;
- all supported regions, with region included as a control variable;
- no esports-team interpretation.

If the latest patch has fewer than 500 matches, the resolver adds immediately
preceding patches one at a time, up to three patches total. It records the widening
in the evidence packet. If the resulting cohort remains underpowered, it abstains.

### 7.2 User filters

Supported filters are region, patch or patch range, date range, queue, champion,
role, and a registered gameplay metric. Unsupported fields fail validation.

### 7.3 Independence and pairing

Team comparisons use one analytical observation per match. Continuous metrics use
winner-minus-loser paired differences. Side-based regression uses one blue-team row
per match with blue-minus-red feature deltas and a blue-win target. Player analyses
use player-match observations and cluster or split by PUUID.

## 8. Statistical Methods and Thresholds

### 8.1 Continuous gameplay metrics

For gold, kills, farm, damage, vision, and objective counts:

- estimate the paired winner-minus-loser difference;
- report the raw difference and paired standardized effect (`d_z`);
- compute a seeded 2,000-resample match bootstrap 95% confidence interval;
- compute a seeded two-sided paired sign-permutation p-value;
- compare against a zero-difference baseline.

### 8.2 Binary and categorical exposures

For first objectives, champion presence, and registered categorical exposures:

- report win-rate or risk difference against the eligible complement;
- report an odds ratio where meaningful;
- use a Wilson/Newcombe confidence interval and an appropriate exact or
  chi-square test;
- refuse comparisons with sparse cells.

### 8.3 Conditional effects

The interpretable logistic model uses named feature deltas with region and patch
controls. It reports calibrated probabilities, coefficients/odds ratios, and
confidence or bootstrap intervals. Its claims must agree in direction with the
paired analysis or be labeled conditional and reviewed separately.

### 8.4 Multiple comparisons

Every analysis run defines its candidate family before testing. Benjamini-Hochberg
false-discovery-rate correction is applied within that family. A user-facing
recommendation requires `q <= 0.05`; `p` without corrected `q` is insufficient.

### 8.5 Minimum evidence

A recommendation requires all of the following:

- at least 500 matches;
- at least 100 distinct players;
- at least 100 observations or events in each compared group;
- a 95% confidence interval that excludes the null;
- `q <= 0.05`;
- practical effect of at least 3 percentage points in win rate or `|d_z| >= 0.20`;
- consistent direction in at least two temporal folds.

"Strong" evidence additionally requires at least 2,000 matches, `q <= 0.01`, a
5-point win-rate difference or `|d_z| >= 0.50`, and consistent direction in all
available temporal folds. Results below the recommendation threshold may be shown
as descriptive facts only when clearly labeled weak/uncertain.

### 8.6 Automatic inter-region comparison

An inter-region difference is shown automatically only when:

- every displayed region contributes at least 500 matches and 100 wins/losses;
- the patch-adjusted interaction has `q <= 0.05`;
- the between-region contrast is at least 3 win-rate points or 0.20 standardized
  effect units;
- the direction survives a recent-period sensitivity check.

Otherwise the response may state that no robust regional difference was detected,
but it may not rank regions.

## 9. Evidence Contract

Each evidence packet contains at least:

```json
{
  "evidence_id": "stable-id",
  "question": "French user question",
  "analysis_request": {},
  "cohort": {
    "mode": "CLASSIC",
    "queue_ids": [420],
    "patches": [],
    "regions": [],
    "date_range": {},
    "widened": false
  },
  "sample": {
    "matches": 0,
    "players": 0,
    "events_by_group": {}
  },
  "estimate": {
    "metric": "registered_metric",
    "value": 0.0,
    "unit": "registered_unit",
    "baseline": 0.0,
    "effect_size": 0.0,
    "confidence_interval_95": [0.0, 0.0],
    "p_value": 0.0,
    "q_value": 0.0
  },
  "method": {
    "name": "registered_method",
    "version": "semver",
    "seed": 0
  },
  "robustness": {
    "temporal_folds": [],
    "regional_effects": [],
    "player_disjoint_check": {},
    "passed": false
  },
  "recommendation": {
    "allowed": false,
    "strength": "none|moderate|strong",
    "wording_key": "registered-template"
  },
  "limitations": [],
  "provenance": {
    "dataset_version": "sha256",
    "source_match_ids_hash": "sha256",
    "generated_at": "ISO-8601"
  }
}
```

Zero values in this example describe types, not defaults. Missing statistical
quantities must be `null` with an explanatory limitation, never silently zero.

## 10. Chat Contract

### 10.1 Request

`POST /api/analytics/chat`

```json
{
  "message": "Comment gagne-t-on le plus souvent en EUW ?",
  "context": null,
  "filters": {}
}
```

`context` may contain only the previously validated `AnalysisRequest`, allowing
follow-ups such as "et en KR ?". It is not arbitrary chat history or SQL.

### 10.2 Response

```json
{
  "success": true,
  "answer": "French deterministic and optionally qualitative response",
  "analysis_request": {},
  "evidence": [],
  "warnings": [],
  "abstained": false,
  "dataset_version": "sha256"
}
```

Each numeric token in `answer` must originate from a deterministic evidence
template and map to an evidence ID. The optional LLM narrative is validated to
contain no new numeric token.

### 10.3 Supported phase 1 intents

- rank registered conditions associated with winning;
- explain one registered gameplay metric;
- compare a metric across supported regions;
- narrow by champion and/or role when adequately sampled;
- compare recent patches;
- request the evidence or limitations behind a prior answer.

Unsupported timeline, organization-team, causal, betting-certainty, or arbitrary
data-mining requests return a structured abstention.

## 11. Error Handling and Abstention

Stable abstention/error codes include:

- `UNSUPPORTED_INTENT`;
- `UNSUPPORTED_FILTER`;
- `TIMELINE_DATA_UNAVAILABLE`;
- `ESPORT_TEAM_IDENTITY_UNAVAILABLE`;
- `INSUFFICIENT_MATCHES`;
- `INSUFFICIENT_PLAYERS`;
- `SPARSE_COMPARISON_GROUP`;
- `EFFECT_NOT_ROBUST`;
- `ANALYTICS_DB_STALE`;
- `SOURCE_DATA_INVALID`;
- `LLM_UNAVAILABLE`;
- `LLM_OUTPUT_REJECTED`.

If the LLM is unavailable or rejected, the deterministic planner/renderer handles
supported intents. If `analytics.db` is stale, the API reports the source and mart
versions and refuses evidence generation unless explicitly configured for a
documented stale-read tolerance. Phase 1 defaults to refusal.

An extraction run fails its quality gate if more than 0.5% of eligible matches are
unexplained failures. Known invalid records are counted separately by stable issue
code and disclosed in build statistics.

## 12. Evaluation and Anti-leakage

### 12.1 Descriptive analytics

Synthetic fixtures with known effects and null effects verify estimates, intervals,
FDR correction, and abstention. Swapping blue/red sides must not change a paired
claim. Duplicating a match must be detected rather than double its weight.

### 12.2 Interpretable model

Primary evaluation is chronological:

- earliest 70% train;
- next 15% validation;
- latest 15% final test.

The final test remains untouched until model and thresholds are fixed. Metrics
include majority and mean baselines, log loss, Brier score, calibration error,
ROC-AUC where valid, and confidence intervals. Accuracy alone is insufficient.

A player-disjoint sensitivity split greedily constructs train/test match sets with
no shared PUUID and discards bridge matches. If fewer than 500 test matches survive,
the grouped result is reported unavailable rather than replaced by a weaker split.
Player-profile analyses use standard PUUID group splits.

### 12.3 Leakage classification

Every feature is registered as one of:

- pre-match;
- in-game temporal;
- final-game retrospective;
- target or forbidden.

Phase 1 evidence may use final-game retrospective features but must label the
result retrospective. Target/forbidden fields such as raw `win` indicators never
enter explanatory features. A test fails if a target/forbidden field reaches a
model matrix or statistical exposure.

The existing NN checkpoint is never described as independently validated. A clean
retraining/evaluation is required before any auxiliary performance statement.

## 13. Testing Strategy

### 13.1 Unit tests

- extraction of valid/malformed Match-V5 payloads;
- exactly one match row, two team rows, and ten participant rows;
- queue, patch, region, role, objective, and feature normalization;
- cohort defaults, widening, filters, and sample thresholds;
- paired effects, bootstrap intervals, permutation tests, and FDR;
- practical-effect and evidence-strength classification;
- regional comparison gates;
- evidence serialization, stable IDs, and cache invalidation;
- planner schema validation and all abstention codes;
- numeric trace validator and LLM fallback;
- leakage registry enforcement.

### 13.2 Integration tests

- temporary operational DB -> temporary `analytics.db` backfill;
- idempotent rerun and incremental addition;
- source DB remains byte/logically unchanged by extraction;
- chat question -> validated request -> evidence -> French answer;
- follow-up context such as EUW -> KR;
- unsupported question -> explicit abstention;
- existing Flask endpoints retain their response contracts.

### 13.3 Performance targets

On the audited corpus size after materialization:

- cached evidence lookup p95 below 100 ms;
- ordinary single-metric cohort analysis p95 below 2 seconds;
- chat API excluding external LLM latency p95 below 3 seconds;
- no request-time raw JSON full scan.

Performance tests must report hardware/context and are not substitutes for
correctness tests.

## 14. Web Integration

The Flask app gains an analytics blueprint rather than expanding the existing
prediction route module. It exposes the chat endpoint, evidence lookup, analytics
health/version, and the `/analytics` page.

The page contains:

- a French chat transcript;
- suggested supported questions;
- visible active cohort filters;
- collapsible evidence cards;
- explicit association/causality labels;
- warnings and abstention explanations;
- dataset version and freshness.

No chat request triggers model training, source backfill, or external collection.
Those remain explicit maintenance commands.

## 15. Rollout Phases

### Phase 0: specification and plan

- approve this specification;
- write an implementation plan with file-level tasks and verification commands.

### Phase 1: analytical mart

- implement schema, extractor, rebuild/incremental commands, metadata, and tests;
- backfill into a non-versioned `analytics.db`;
- publish build quality and dataset-version reports.

### Phase 2: statistics and evidence

- implement registry, cohort resolver, paired analyses, FDR, regional gates,
  evidence packets, cache, and abstention;
- validate on synthetic and real read-only cohorts without making performance
  claims from trained rows.

### Phase 3: interpretable and NN auxiliary models

- implement chronological and player-disjoint evaluation;
- train/evaluate the interpretable model;
- add the NN discovery adapter and independent candidate-validation gate;
- publish calibration, baseline, uncertainty, and leakage reports.

### Phase 4: chatbot and web UI

- implement typed planner, deterministic renderer, provider-neutral LLM adapter,
  numeric trace validation, API, page, and end-to-end tests.

### Phase 5: hardening and completion audit

- run existing and new tests;
- measure performance targets;
- verify every Goal requirement against authoritative evidence;
- document commands, limitations, and measured results.

### Future timeline phase

Populate timelines and design a separate specification for 5/10/15-minute state
features. Timeline work is not silently folded into phase 1 and is not required for
the retrospective final-state subsystem to be complete.

## 16. Compatibility and Change Safety

- Existing dirty-worktree changes are preserved and never staged accidentally.
- Only feature-related files are staged or committed.
- `training_data.db` remains the operational source and is opened read-only by the
  extractor.
- `analytics.db` is derived, separate, and ignored by Git.
- Existing prediction endpoints and response shapes remain unchanged.
- The new analytics blueprint and package are additive.
- Existing timeline code remains disabled until data exists.
- No schema migration is applied to `training_data.db` for phase 1 analytics.

## 17. Acceptance Criteria

The retrospective subsystem is complete only when all of the following are proven:

1. The mart extracts the required named features from the real source without
   mutating it and passes its extraction-quality gate.
2. Cohort defaults and filters are deterministic, indexed, and disclosed.
3. Every evidence packet contains the required sample, estimate, baseline, effect,
   interval, corrected significance, method, robustness, limits, and provenance.
4. Regional comparisons appear automatically only through their stated gate.
5. Recommendations obey minimum-sample, statistical, practical-effect, and temporal
   stability thresholds.
6. The interpretable model is evaluated chronologically and against simple
   baselines with calibration and uncertainty.
7. Grouped/player-disjoint evaluation is run or explicitly reported underpowered.
8. The NN is auxiliary and cannot bypass evidence validation.
9. A supported French question produces a readable answer whose every number is
   traceable to evidence.
10. Unsupported or underpowered questions abstain clearly.
11. No phase 1 answer claims timeline, causal, pre-match, or esports-team evidence.
12. Existing relevant tests and all new tests pass.
13. Documentation records commands, dataset version, limitations, and actual
    measured evaluation results.

## 18. Explicit Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Seed-and-network sampling bias | Describe the population as "NNriot-collected matches"; never generalize to all players without rank/source evidence. |
| Rank absent from Match-V5 | Do not claim rank-adjusted results; add rank only through a separately audited data source. |
| Post-match leakage | Label phase 1 retrospective; classify features and ban target fields. |
| Patch drift and stale data | Default to recent adequately powered patches; expose freshness and dataset version. |
| Regional imbalance | Use adjusted effects, per-region minimums, and abstention. |
| Repeated players | Pair by match, use temporal folds, cap/sensitivity-check frequent players, and run player-disjoint evaluation. |
| Rare events | Enforce event-count and practical-effect thresholds. |
| Multiple testing | Predefine candidate families and apply Benjamini-Hochberg correction. |
| Hashed NN explanations | Treat NN as discovery-only; validate on named analytical fields. |
| LLM hallucination | Typed plans, allowlisted executor, deterministic numeric rendering, numeric validation, and fallback. |
| Operational lock/contention | Separate derived DB; source opened read-only; no request-time backfill. |
| Dirty worktree | Stage only files explicitly created or changed for this feature. |
