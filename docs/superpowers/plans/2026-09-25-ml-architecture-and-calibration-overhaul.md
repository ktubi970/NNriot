# ML Architecture, Calibration & Baseline Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a reproducible comparison of constant, tree and neural winner predictors, then fit and serve a traceable winner calibration without weakening the existing data and API contracts.

**Architecture:** Share feature normalization, frozen split membership and dataset fingerprints between trainers. Keep the existing metrics API, fit an optional temperature on the selected neural checkpoint, and bind its calibration artifact to the exact checkpoint and split manifest. Apply calibration once in `ContinuousTrainer.predict`; invalidate it before any weight mutation. Add descriptive diagnostics after the benchmark and serving contracts are covered.

**Tech Stack:** Existing Python environment `.venv312`, NumPy, SciPy, scikit-learn, joblib, TensorFlow/Keras, Flask and pytest. Keep Python 3.10-compatible source syntax; execute checks with the repository's Python 3.12 environment. Do not change dependency floors or install another ML framework.

**Spec:** The [implementation contract](#implementation-contract) in this document consolidates the original scope and the review of 2026-09-25. This revision supersedes the original snippets, including their incompatible metrics signature and approximate Brier identity.

**Executor:** Gemini 3.8. This is a self-contained handoff, not a claim that this model was run or benchmarked. Use repository tools and skills available in the execution environment; if a named skill is absent there, follow this document's sequential test/review steps and disclose that limitation.

## Global Constraints

- Work from `D:\projet\NNriot`; run PowerShell commands from that root.
- Preserve unrelated working-tree changes. Several affected files were already modified or untracked when this plan was written. Record `git status --short` and inspect each file before editing; never reset, clean, or stage the whole repository.
- Read the applicable `AGENTS.md` and `C:/Users/ktubi/.agents/skills/honesty/SKILL.md` when available.
- Preserve `evaluate_winner(y_true, probabilities, train_positive_rate)` and its existing keys and input validation. Add diagnostics through an optional keyword only.
- Preserve the default chronological 70/15/15 protocol, equal-timestamp groups, unique match IDs and strictly increasing split time ranges.
- Fit encoders and predictive weights on train only. Select epochs/tree settings on validation only. Fit temperature on validation only after model selection. Never use test scores to choose settings, temperature, features or promotion.
- Validation used for epoch selection and temperature fitting is reused data: label its before/after calibration measurements as in-sample fitting diagnostics. The untouched test is the final assessment; do not call validation evidence an independent calibration assessment.
- SQLite sources remain read-only. Candidate runs use fresh output paths and never automatically replace the active model. Production model activation is outside this implementation.
- Keep successful prediction response shapes and existing HTTP error conventions. Only winner is calibrated; other heads remain unvalidated and must not inherit a calibration claim.
- `is_calibrated=True` means a verified winner temperature transform belongs to these exact weights; it does not establish predictive usefulness, calibration on every population, or calibration of other heads.
- Do not run full-data training, collection, or active-model promotion as part of implementation validation. Use isolated synthetic fixtures and tiny Keras models.
- Do not assert that tests passed based on this document's expected results. Execute them and record actual outcomes.

## Review Focus

1. **Stale or malformed calibration:** missing file, mismatched checkpoint/manifest, nonfinite or nonpositive temperature must keep valid weights loaded but uncalibrated. Covered in Tasks 4 and 6.
2. **Changed data behind a frozen run:** a missing ID, duplicate ID, edited label/feature, changed timestamp or altered split membership must prevent a purported same-cohort comparison. Covered in Tasks 1 and 3.
3. **Patch boundaries and unknown versions:** overlapping time ranges must fail; fallback must report the actual chronological strategy. Covered in Task 2.
4. **Softmax probabilities and concurrent training:** preserve [A, B], normalize the calibrated pair, apply once, and reject inference after mutation begins, including failed training. Covered in Tasks 4 and 6.
5. **Degenerate evaluation:** tied probabilities, absent classes, failed slope fitting and small calibration samples must produce explicit unavailable states or errors, never ideal-looking invented diagnostics. Covered in Tasks 4 and 7.

## Implementation contract

### What the existing code already does

Read these symbols before implementing their tasks; line numbers are intentionally not treated as stable in the dirty working tree:

| File / symbol | Current contract to preserve |
| --- | --- |
| `core/pre_match_dataset.py:load_records` | Latest bounded candidates, CLASSIC only, decoded/validated rows, read-only SQLite |
| `core/pre_match_dataset.py:chronological_split` | Train/validation/test, whole timestamps, duplicate rejection, minimum 20 records |
| `scripts/train_pre_match_model.py:to_pre_match_features` | Draft/context allowlist, champion aliases, role aliases, exclusion of undated career stats |
| `scripts/train_pre_match_model.py:train_model` | Select epoch by validation winner log-loss, candidate output, frozen manifest, train-derived constant baseline |
| `core/training_metrics.py:evaluate_winner` | Three positional arguments; `model`, `baseline`, `accuracy_lift`, `log_loss_improvement` |
| `core/generate_graph.py:build_multi_output_model` | `winner` is a two-column softmax: column 0 = A, column 1 = B |
| `core/model_factory.py:ModelFactory.get_model` | Architecture validation; valid weights must not be rebuilt because only their calibration is missing |
| `scripts/continuous_trainer.py:ContinuousTrainer.predict` | Shared prediction path for both web prediction endpoints |
| `app/core.py:final_predict_match_outcome` and `app/routes/api.py:api_predict_custom` | Two inference entry points; both must use the same feature allowlist and calibrated predictor |

The existing `runs/pre_match_10k_15epochs_20260923_211035/model.metrics.json` reports test accuracy 0.526, equal to the constant baseline, AUC about 0.514 and log-loss improvement about 0.0000203. Preserve this as historical evidence; do not overwrite or retrofit its reports. It does not demonstrate useful predictive lift. Calibration cannot create discrimination.

### Decisions fixed for this revision

- One plan is retained because the tasks share the same data, artifact and serving contracts. Delivery gates are: **Tasks 1-3 benchmark**, **Tasks 4-6 calibration**, **Task 7 diagnostics**.
- Keep the current three splits. An optional `--calibrate` flag enables temperature fitting; the default remains an uncalibrated candidate, preserving current CLI behavior. A future independent calibration window is outside this revision.
- `--split-strategy patch` means newest major.minor patch for test, preceding patch for validation, all earlier patches for train, subject to strict timestamp separation. It is a patch holdout, not random stratification.
- Fewer than three valid patches falls back to chronological splitting with a recorded reason. Unknown/malformed patches or overlapping patch time ranges fail explicitly; they do not silently change the requested experiment.
- Frozen replay loads exact match IDs from the manifest. It never obtains a supposedly equivalent cohort by selecting the latest N matches again.
- Tree inputs contain **24 categorical columns**: champion and role for each of ten participants, plus game mode, game version, season and region. Preserve every participant, including duplicate/unknown roles. Sort within each team by role rank, then role string and champion string; this avoids overwriting duplicate role entries. This intentionally replaces the original inconsistent 10-versus-20-column proposal.
- Fix tree candidate settings before test access: `max_iter` in `(50, 100, 150)`, `learning_rate=0.1`, `max_leaf_nodes=15`, `min_samples_leaf=20`, `early_stopping=False`, seed 42 by default. Select lowest validation log-loss, breaking ties toward fewer iterations. Do not use an internal random validation fraction.
- Temperature domain is `[0.05, 10.0]`; evaluate identity T=1 explicitly and never accept worse validation NLL. Record boundary fits. Minimum fitting size is 20 with both classes present, an operational guard rather than a statistical guarantee.
- Preserve `brier` and equal-width `calibration_error`. Add quantile ECE and binned Brier diagnostics under a separate `diagnostics` object. Never silently redefine historical metric names.

### File ownership and dependency order

| File | Responsibility / task |
| --- | --- |
| `core/pre_match_dataset.py` | Shared normalization, validated exact-ID loading, patch split (1, 2) |
| `core/experiment.py` (new) | Manifest construction, validation, fingerprints and frozen replay (1) |
| `scripts/train_pre_match_model.py`, `cli.py` | Shared experiment use, CLI forwarding, optional calibration (1, 2, 5) |
| `scripts/train_tree_baseline.py` (new) | Tree extraction, validation selection and comparable report (3) |
| `core/calibration.py` (new) | Numeric temperature functions and strict artifact validation (4) |
| `core/model_factory.py`, `scripts/continuous_trainer.py` | Loading, single application and invalidation (6) |
| `app/core.py`, `app/routes/api.py` | Consistent pre-match normalization and error mapping (6) |
| `core/training_metrics.py` | Additive diagnostics (7) |
| `docs/pre_match_training.md` | User-facing experiment, calibration and bundle instructions, updated alongside each task |

Tasks run in order. Do not parallelize edits to the trainers or metrics across tasks. At each gate, review the final interfaces and passing tests before starting the next group. Do not replace the neural architecture, tune all its heads, add a generic model registry, or rebuild analytics as part of this work.

### Preflight

- [ ] Read this complete document, inspect the working tree and confirm the listed symbols still exist. Adapt moved locations without weakening contracts; record substantive design conflicts.

```powershell
git status --short
rg -n 'def (load_records|chronological_split|to_pre_match_features|train_model|evaluate_winner|predict|run_training_step)' core scripts
.\.venv312\Scripts\python.exe -m pytest tests/test_training_metrics.py tests/test_pre_match_dataset.py -q
```

Expected reference from the review: 61 tests passed. Re-run; do not treat that historical result as current verification. Record pre-existing failures separately.

- [ ] Before committing any task, inspect the diff and stage only owned changes. The task commit messages below are suggestions, not permission to commit unrelated existing work. If a modified file contains inseparable pre-existing changes, leave it unstaged and report that fact.

---

### Task 1: Share the feature contract and replay a frozen experiment

**Files:** Modify `core/pre_match_dataset.py`, `scripts/train_pre_match_model.py`; create `core/experiment.py`, `tests/test_experiment.py`; update `tests/test_pre_match_run.py`, `docs/pre_match_training.md`.

**Interfaces:**

- `core.pre_match_dataset.to_pre_match_features(feat: dict) -> dict`: move the current implementation unchanged; import it into the trainer so existing imports from the script still work.
- `core.pre_match_dataset.load_records_by_ids(db_path: str, match_ids: list[str]) -> list[dict]`: same eligibility/decoding rules as `load_records`, exact IDs, sorted chronologically; error if any ID is missing, invalid or duplicated in the result.
- `core.experiment.build_manifest(splits: dict, *, source_db: str | None, split_metadata: dict) -> dict`.
- `core.experiment.restore_splits(records: list[dict], manifest: dict) -> dict`: reconstruct exact membership/order and verify time ranges, features, labels and fingerprints.
- `core.experiment.load_frozen_splits(db_path: str, manifest: dict) -> dict`: call the exact-ID loader, then `restore_splits`.
- `core.experiment.read_manifest(path: str | Path) -> dict`: parse JSON and validate supported schema and structure.

**Step 1 — add regression tests before moving code.**

- [ ] Add the following tests to `tests/test_experiment.py`. Use `copy.deepcopy` whenever mutating a fixture. Its local `records()` fixture is defined here, so the file does not import tests from another test module.

```python
from copy import deepcopy
import pytest
from core.pre_match_dataset import chronological_split, to_pre_match_features
from core.experiment import build_manifest, restore_splits


def records(n=40):
    from core.feature_labels import LABEL_KEYS
    rows = []
    for i in range(n):
        labels = dict.fromkeys(LABEL_KEYS, 0)
        labels['winner'] = i % 2
        player = {'championName': 'Ahri', 'teamPosition': 'MIDDLE',
                  'gold': 10000, 'career_win_rate': 0.99}
        rows.append({'match_id': f'M_{i:04d}', 'game_creation_time': 1000 + i,
                     'winner_label': i % 2, 'labels_json': labels,
                     'feature_json': {'game_version': '16.18.1',
                                      'team_a': [dict(player) for _ in range(5)],
                                      'team_b': [dict(player) for _ in range(5)]}})
    return rows


def manifest_for(rows):
    return build_manifest(chronological_split(rows), source_db=None,
        split_metadata={'requested_strategy': 'chronological',
                        'effective_strategy': 'chronological', 'fallback_reason': None})


def test_normalization_preserves_aliases_and_excludes_undated_stats():
    clean = to_pre_match_features(records()[0]['feature_json'])
    assert clean['team_a'][0] == {'champion': 'Ahri', 'role': 'MIDDLE'}


def test_replay_uses_manifest_membership_even_with_newer_rows():
    rows = records()
    manifest = manifest_for(rows)
    restored = restore_splits(list(reversed(records(45))), manifest)
    assert [r['match_id'] for r in restored['test']] == [
        r['match_id'] for r in chronological_split(rows)['test']]


@pytest.mark.parametrize('mutation', ['missing', 'timestamp', 'label', 'feature', 'duplicate'])
def test_replay_rejects_changed_source(mutation):
    rows = records()
    manifest = manifest_for(rows)
    changed = deepcopy(rows)
    if mutation == 'missing':
        changed.pop()
    elif mutation == 'timestamp':
        changed[-1]['game_creation_time'] += 100
    elif mutation == 'label':
        changed[-1]['labels_json']['winner'] ^= 1
        changed[-1]['winner_label'] ^= 1
    elif mutation == 'feature':
        changed[-1]['feature_json']['team_a'][0]['championName'] = 'Zed'
    else:
        changed.append(deepcopy(changed[-1]))
    with pytest.raises(ValueError):
        restore_splits(changed, manifest)
```

- [ ] Add an SQLite test beside `_database` in `tests/test_pre_match_dataset.py`: create 40 eligible rows using that helper, request the oldest 4 IDs, assert exactly those 4 are loaded despite newer rows, then delete one in fixture setup and assert replay raises. Snapshot source bytes before/after the read and assert equality. Include a corrupt feature row and duplicate request ID case, each rejected with `ValueError`.

- [ ] Run the new tests and confirm missing interface failures, rather than dependency or syntax errors.

```powershell
.\.venv312\Scripts\python.exe -m pytest tests/test_experiment.py tests/test_pre_match_dataset.py -q
```

**Step 2 — implement shared loading and the manifest.**

- [ ] Move normalization into `core/pre_match_dataset.py`, importing `json_utils` locally inside its raw-match branch to avoid unnecessary import side effects. Factor existing row decoding/eligibility into `_decode_record(row) -> dict | None`; use it in both loaders without changing `load_records` selection semantics.

For `load_records_by_ids`, reject empty/duplicate/non-string IDs before querying. Open the same URI with `mode=ro` and `PRAGMA query_only=ON`. Query IDs in chunks of 500 with SQL placeholders and the current join/CLASSIC filter; never interpolate IDs into SQL. Count eligible matches by ID and require exactly one per requested ID.

```python
for start in range(0, len(match_ids), 500):
    batch = match_ids[start:start + 500]
    placeholders = ','.join('?' for _ in batch)
    sql = f'''SELECT t.id, t.match_id, m.game_creation_time, t.feature_json,
                     t.labels_json, t.winner_label
              FROM training_dataset AS t
              JOIN matches AS m ON m.match_id = t.match_id
              WHERE m.game_mode = 'CLASSIC' AND t.match_id IN ({placeholders})'''
    for row in conn.execute(sql, batch):
        record = _decode_record(row)
        if record is None:
            raise ValueError(f'Invalid frozen record: {row["match_id"]}')
        selected.append(record)
```

- [ ] Implement schema 2 manifests. Keep the **existing** `dataset_sha256` serialization algorithm byte-for-byte: iterate train, validation, test, then each ordered row; hash canonical JSON plus a newline for `{match_id, game_creation_time, features: normalized_features, labels: labels_json}`. Add `split_sha256` hashing the canonical `splits` object, which includes split names and their ID/timestamp lists, because the existing dataset hash alone does not encode boundary positions.

```python
SPLIT_NAMES = ('train', 'validation', 'test')
FEATURE_POLICY = 'draft_context_only_v1; no post-match or undated career statistics'

# In build_manifest, after validating nonempty ordered splits and strict boundaries:
manifest = {
    'schema_version': 2,
    'source_db': str(Path(source_db).resolve()) if source_db else None,
    'dataset_sha256': fingerprint.hexdigest(),
    'feature_policy': FEATURE_POLICY,
    'selection': 'frozen match membership; see split_metadata',
    'split_metadata': dict(split_metadata),
    'splits': {name: [{'match_id': r['match_id'],
                      'game_creation_time': r['game_creation_time']} for r in splits[name]]
               for name in SPLIT_NAMES},
}
manifest['split_sha256'] = hashlib.sha256(json.dumps(
    manifest['splits'], sort_keys=True, separators=(',', ':'), allow_nan=False
).encode('utf-8')).hexdigest()
```

Import `hashlib`, `json`, `Path`, and the shared normalizer in `core/experiment.py`. Implement validation directly: exact split names, nonempty lists, unique IDs across splits, integer positive timestamps, sorted `(time,id)` order, and `max(left.time) < min(right.time)` for adjacent splits. Reject unknown schemas and feature policies. Accept schema 1 historical manifests by validating the same membership and old dataset hash, deriving `split_sha256` in memory, and marking metadata `effective_strategy='chronological'`, `requested_strategy='chronological'`, `fallback_reason=None`; do not rewrite the historical file.

`restore_splits` rejects duplicates in supplied records, checks every declared ID/time, allows extra source IDs but never includes them, recomputes both hashes, and returns rows in manifest order. `read_manifest` does structural checks before database access; `restore_splits` does data-dependent checks. A schema 2 manifest with a bad stored split hash is always rejected.

**Step 3 — use the shared manifest in the neural trainer.**

- [ ] Replace the trainer's inline fingerprint/manifest construction with `build_manifest`. Preserve model building, selection, result keys, refusal to overwrite and returned model behavior. Keep the old `to_pre_match_features` import path by importing the shared function at module scope.
- [ ] Add keyword `split_manifest: dict | None = None` to `train_model`; use `restore_splits(records, split_manifest)` when supplied, otherwise the existing chronological split. Keep reports tied to the new output's own manifest and include `split_sha256`. Do not copy an old manifest's source path as authoritative: record the actual resolved source used, while retaining frozen IDs and hashes.
- [ ] Update the existing run test to assert schema 2 and a 64-character split hash, while preserving all existing train/validation/test, baseline and `is_calibrated=False` assertions. Document immutable manifest replay.

```powershell
.\.venv312\Scripts\python.exe -m pytest tests/test_experiment.py tests/test_pre_match_dataset.py tests/test_pre_match_training.py tests/test_pre_match_run.py -q
git diff --check
```

Expected: all listed tests pass; no production database/model writes. Suggested commit: `refactor(training): share normalized frozen experiment contracts`.

---

### Task 2: Add honest patch holdouts and CLI replay

**Files:** Modify `core/pre_match_dataset.py`, `scripts/train_pre_match_model.py`, `cli.py`, `tests/test_experiment.py`, `tests/test_config_cli.py`, `docs/pre_match_training.md`; create `tests/test_patch_stratified_split.py`.

**Interfaces:**

- `extract_major_patch(version: str) -> tuple[int, int]`: accept strings whose first two dot-separated components are decimal integers; reject missing, blank, malformed and `UNKNOWN`.
- `split_records(records: list[dict], strategy: str = 'chronological') -> tuple[dict, dict]`: returns splits and metadata with requested/effective strategy, fallback reason and patches by split.
- Extend `train_model` with keyword `split_strategy: str = 'chronological'`.
- Add `--split-strategy {chronological,patch}` and `--split-manifest PATH` to both neural CLI entry points. An explicit manifest is authoritative; reject an explicitly supplied split strategy together with it. Use parser default `None`, then resolve to chronological when no manifest is given.

- [ ] Write tests using a local record helper with unique IDs, valid timestamps and version strings. Pin the boundary case that the original plan missed:

```python
import pytest
from core.pre_match_dataset import split_records


def rows_for(patch, times):
    return [{'match_id': f'{patch}_{t}', 'game_creation_time': t,
             'feature_json': {'game_version': patch}} for t in times]


def test_patch_holdout_rejects_overlapping_time_ranges():
    rows = (rows_for('14.1.1', list(range(1, 19)) + [100])
            + rows_for('14.2.1', [20, 30]) + rows_for('14.3.1', [40, 50]))
    with pytest.raises(ValueError, match='overlap'):
        split_records(rows, 'patch')


def test_fallback_reports_effective_strategy():
    splits, meta = split_records(rows_for('16.18.1', range(1, 41)), 'patch')
    assert meta['requested_strategy'] == 'patch'
    assert meta['effective_strategy'] == 'chronological'
    assert meta['fallback_reason'] == 'fewer_than_three_patches'
    assert [len(splits[k]) for k in ('train', 'validation', 'test')] == [28, 6, 6]


@pytest.mark.parametrize('version', ['UNKNOWN', '', None, '16', 'x.18'])
def test_patch_mode_rejects_unknown_versions(version):
    rows = rows_for('16.18.1', range(1, 41))
    rows[0]['feature_json']['game_version'] = version
    with pytest.raises(ValueError, match='patch'):
        split_records(rows, 'patch')
```

- [ ] Add tests for valid 14.9/14.10/14.11 grouping, duplicate IDs, equal timestamps across patch boundaries, fewer than 20 records, and input permutation invariance. In the valid case assert exact IDs in every split, not only counts.
- [ ] Run the new test file and confirm the new interface is missing.
- [ ] Reuse the chronological splitter's input validation. The minimal implementation can first call `chronological_split(records)` to validate and obtain the fallback, then group by numeric major/minor. Validate final time intervals; do not infer chronology solely from earliest patch timestamps.

```python
ordered_patches = sorted(groups)  # numeric (major, minor) tuples
if len(ordered_patches) < 3:
    return chronological, {
        'requested_strategy': 'patch', 'effective_strategy': 'chronological',
        'fallback_reason': 'fewer_than_three_patches',
        'patches': patches_by_split(chronological),
    }
splits = {
    'train': [r for p in ordered_patches[:-2] for r in groups[p]],
    'validation': list(groups[ordered_patches[-2]]),
    'test': list(groups[ordered_patches[-1]]),
}
for rows in splits.values():
    rows.sort(key=lambda r: (r['game_creation_time'], r['match_id']))
for left, right in (('train', 'validation'), ('validation', 'test')):
    if splits[left][-1]['game_creation_time'] >= splits[right][0]['game_creation_time']:
        raise ValueError('patch time ranges overlap; choose an explicit chronological run')
```

Define `patches_by_split(splits)` in the same module: unique sorted `major.minor` strings per split, and `UNKNOWN` only for chronological metadata. In chronological mode unknown versions remain allowed. Return the same three metadata fields plus patch mapping for successful patch mode with `effective_strategy='patch'`.

- [ ] Wire flags through `cli.run_train` without adding flags to the forwarded argument list when absent; preserve the existing exact-list CLI test. In the script, a supplied manifest uses `read_manifest` and `load_frozen_splits(args.db, manifest)` instead of `fetch_training_records`. `--samples` applies only without a manifest; log that it is ignored for frozen replay. Flatten restored splits in the declared split order when calling `train_model`. With no manifest, `train_model` calls `split_records(records, split_strategy)` and passes its returned metadata to `build_manifest`. With a manifest, it preserves the verified split metadata and never recomputes boundaries; reject a non-default programmatic split strategy alongside a manifest.
- [ ] Add CLI tests asserting new flag forwarding, mutually exclusive strategy/manifest, missing manifest errors, and `--help` without TensorFlow import. Update docs with fallback/failure behavior and frozen replay semantics.

```powershell
.\.venv312\Scripts\python.exe -m pytest tests/test_patch_stratified_split.py tests/test_experiment.py tests/test_config_cli.py tests/test_pre_match_run.py -q
.\.venv312\Scripts\python.exe cli.py train --help
git diff --check
```

Suggested commit: `feat(training): add strict patch holdouts and frozen split replay`.

---

### Task 3: Train a tree baseline and compare the exact held-out cohort

**Files:** Create `scripts/train_tree_baseline.py`, `tests/test_tree_baseline.py`; update `docs/pre_match_training.md`.

**Interfaces:**

- `extract_tabular_features(feat: dict) -> tuple[list[str], list[str]]`: normalized 24-column categorical row and stable names.
- `train_tree_model(records: list[dict], *, output_dir: str, split_strategy: str = 'chronological', split_manifest: dict | None = None, seed: int = 42, compare_report: dict | None = None) -> dict`.
- CLI: `--db` required, `--samples` default 5000, `--output` optional unique run directory, `--seed`, `--split-strategy`, `--split-manifest`, `--compare-to PATH_TO_NEURAL_METRICS`.
- `--compare-to` reads the candidate report's referenced split manifest and replays its exact rows. Resolve relative manifest paths against the report's directory. Reject ambiguous combinations with an explicit split manifest/strategy. `--samples` is irrelevant in this mode.
- The direct Python API requires `split_manifest` whenever `compare_report` is supplied. Validate this before creating output files or fitting the tree. For a historical report without a split hash, validate its referenced manifest and derive the hash from that membership; do not silently substitute a newly computed split.
- Outputs in a fresh directory: `model.joblib`, `model.split.json`, `model.metrics.json`, and `comparison.json` only when a comparison is requested.

- [ ] Write the extraction and training tests before implementation. Copy the local `records(n)` helper from Task 1 into this test file, including its `LABEL_KEYS` import; do not import another test module. Use n=200 for trainer tests.

```python
import numpy as np
import pytest
from scripts.train_tree_baseline import extract_tabular_features, train_tree_model


def test_tabular_contract_preserves_duplicate_roles_and_aliases():
    team = [{'championName': name, 'teamPosition': 'MIDDLE', 'gold': 9999}
            for name in ['Zed', 'Ahri', 'Lux', 'Annie', 'Orianna']]
    values, names = extract_tabular_features({'team_a': team, 'team_b': team,
        'game_version': '16.18.1', 'season_id': 16, 'region': 'EUW'})
    assert len(values) == len(names) == 24
    assert values[0:10:2] == ['Ahri', 'Annie', 'Lux', 'Orianna', 'Zed']
    assert values[1:10:2] == ['MIDDLE'] * 5
    assert '9999' not in values
    assert len(set(names)) == 24


def test_tree_report_keeps_shared_metric_contract(tmp_path):
    report = train_tree_model(records(200), output_dir=str(tmp_path / 'tree'))
    assert report['evaluation_target'] == 'winner'
    assert report['is_calibrated'] is False
    assert set(report['metrics']['test']) == {
        'model', 'baseline', 'accuracy_lift', 'log_loss_improvement'}
    assert np.isfinite(report['metrics']['test']['model']['brier'])
    assert (tmp_path / 'tree' / 'model.split.json').exists()
```

- [ ] Add round-trip `joblib` prediction equality; unseen validation champion encoded as -1; a one-class train rejected before fitting; existing output directory rejected without overwriting; and an altered comparison hash rejected before fitting. Use a spy on `HistGradientBoostingClassifier.fit` to assert it is never reached after a hash mismatch. Verify candidate selection uses validation labels only by asserting each evaluation of a candidate has validation-sized arrays; final test evaluation occurs once after selection.
- [ ] Run `tests/test_tree_baseline.py` and confirm expected missing-module failures.
- [ ] Implement extraction with explicit stable column order. Normalize first; reject a team that does not contain five participants rather than silently changing feature width. Do not build a role-to-champion dict, which drops duplicate roles.

```python
ROLE_ORDER = {'TOP': 0, 'JUNGLE': 1, 'MIDDLE': 2, 'BOTTOM': 3, 'UTILITY': 4}

def extract_tabular_features(feat):
    clean = to_pre_match_features(feat)
    values, names = [], []
    for team in ('team_a', 'team_b'):
        participants = clean.get(team, [])
        if len(participants) != 5:
            raise ValueError('tree features require five participants per team')
        ordered = sorted(participants, key=lambda p: (
            ROLE_ORDER.get(p['role'], 5), p['role'], p['champion']))
        for index, player in enumerate(ordered):
            for field in ('champion', 'role'):
                names.append(f'{team}_{index}_{field}')
                values.append(str(player[field]))
    for key in ('game_mode', 'game_version', 'season_id', 'region'):
        names.append(key)
        value = clean.get(key)
        values.append('UNKNOWN' if value is None else str(value))
    return values, names
```

- [ ] Use the shared split/manifest functions. Fit an `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, max_categories=255)` only on train; the category cap includes an infrequent category where needed and fits HGB's maximum bin count. Save feature names and encoder with the model. Verify this parameter is supported by the installed scikit-learn, whose declared minimum is 1.3; do not silently drop the cap.

```python
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1,
                         max_categories=255)
X_train = encoder.fit_transform(raw_X['train'])
X_validation = encoder.transform(raw_X['validation'])
train_rate = float(y['train'].mean())
best = None
history = []
for iterations in (50, 100, 150):
    model = HistGradientBoostingClassifier(
        categorical_features=list(range(24)), max_iter=iterations,
        max_leaf_nodes=15, min_samples_leaf=20, learning_rate=0.1,
        early_stopping=False, random_state=seed)
    model.fit(X_train, y['train'])
    validation = evaluate_winner(y['validation'],
        model.predict_proba(X_validation)[:, 1], train_rate)
    loss = validation['model']['log_loss']
    history.append({'max_iter': iterations, 'validation': validation})
    if best is None or loss < best[0]:
        best = (loss, iterations, model)
selected = best[2]
X_test = encoder.transform(raw_X['test'])
```

Require train labels to contain both 0 and 1. Test and validation may be single-class for scoring; AUC then remains null. Build the complete report with the same `metrics`, hashes, train prior and split counts as the neural report, plus `model_kind='hist_gradient_boosting'`, feature names, chosen iteration count, selection history and library version. Keep all three splits' baseline metrics. Use UTF-8 JSON with `allow_nan=False` and refuse existing output artifacts. Save the frozen manifest before fitting.

- [ ] When comparing, verify the neural report's `dataset_sha256`, train prior, split counts and referenced manifest membership against the replay; derive the split hash from schema 1 manifests when needed. Do not trust a report's matching sample count as proof of matching rows. Compare raw tree to raw neural metrics; after Task 5, also show calibrated neural metrics in a distinct column. Never pick the minimum of raw/calibrated test losses as the neural score.

```python
comparison = {
    'dataset_sha256': manifest['dataset_sha256'],
    'split_sha256': manifest['split_sha256'],
    'selection_policy': 'settings frozen using validation before test access',
    'test': {
        'constant': report['metrics']['test']['baseline'],
        'tree_raw': report['metrics']['test']['model'],
        'neural_raw': compare_report['metrics']['test']['model'],
    },
    'limitations': ['Point estimates; no significance claim.',
                    'Players may overlap between temporal splits.',
                    'Reusing a previously inspected test is retrospective benchmarking.'],
}
if compare_report.get('calibrated_metrics') is not None:
    comparison['test']['neural_calibrated'] = compare_report[
        'calibrated_metrics']['test']['model']
```

A report alone cannot cryptographically prove how an old run was trained. Replaying/validating its cohort ensures comparability of declared data; label historical comparisons accordingly. Do not auto-promote a winner.

- [ ] Document fresh-tree and `--compare-to` commands with an explicit user-chosen read-only source database. Check `--help` without importing TensorFlow. Run the task tests and shared data tests.

```powershell
.\.venv312\Scripts\python.exe -m pytest tests/test_tree_baseline.py tests/test_experiment.py tests/test_training_metrics.py -q
.\.venv312\Scripts\python.exe scripts/train_tree_baseline.py --help
git diff --check
```

**Benchmark gate:** both trainers have the same declared cohort, normalization and metrics; raw comparison exists before calibration work starts. Suggested commit: `feat(baseline): benchmark categorical trees on frozen neural holdouts`.

---

### Task 4: Implement probability temperature scaling and bound calibration artifacts

**Files:** Create `core/calibration.py`, `tests/test_model_calibration.py`.

**Interfaces:**

- `apply_temperature_to_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray`: input is strictly one-dimensional P(B), not logits; validates finite values in [0,1].
- `fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> dict`: returns a validated fit record, never a bare ambiguous boolean.
- `sha256_file(path: str | Path) -> str`: streaming file hash.
- `save_calibration_artifact(checkpoint_path, split_manifest_path, fit: dict, *, input_dim: int) -> Path`: validates the fit, creates a new sidecar, binds it to finalized files.
- `load_calibration_artifact(checkpoint_path, *, input_dim: int) -> dict | None`: absent/invalid/mismatched artifact returns None and logs a concrete reason; does not build or mutate a Keras model.
- `invalidate_calibration(model, checkpoint_path) -> None`: clear in-memory attributes, remove the sidecar if present; propagate removal errors so callers cannot train while leaving stale metadata on disk.

**Artifact schema, version 1:**

```json
{
  "schema_version": 1,
  "method": "binary_temperature",
  "head": "winner",
  "class_order": ["A", "B"],
  "positive_class": 1,
  "feature_policy": "draft_context_only_v1; no post-match or undated career statistics",
  "input_dim": 20000,
  "checkpoint_sha256": "64 lowercase hexadecimal characters",
  "split_manifest_sha256": "64 lowercase hexadecimal characters",
  "dataset_sha256": "64 lowercase hexadecimal characters",
  "split_sha256": "64 lowercase hexadecimal characters",
  "temperature": 1.0,
  "fit": {
    "status": "fitted",
    "partition": "validation",
    "count": 30,
    "class_counts": [15, 15],
    "optimizer_success": true,
    "boundary_hit": false,
    "log_loss_before": 0.7,
    "log_loss_after": 0.69
  }
}
```

The example illustrates field types, not real calibration evidence. The saver computes hashes from files and copies measured fit values; it must not copy the example numbers. No serialized `is_calibrated` assertion is trusted. That attribute is derived only after verification. Companion manifest path is always `checkpoint.with_suffix('.split.json')`, not an arbitrary path from JSON.

- [ ] Write the numerical regression tests below, then the artifact tests described immediately after them.

```python
import numpy as np
import pytest
from core.calibration import apply_temperature_to_probabilities, fit_temperature


def test_probability_contract_keeps_favorite_and_normalization():
    p = np.array([0.1, 0.5, 0.9])
    q = apply_temperature_to_probabilities(p, 2.0)
    np.testing.assert_allclose(q, [0.25, 0.5, 0.75])
    np.testing.assert_array_equal(q >= 0.5, p >= 0.5)
    np.testing.assert_allclose(np.column_stack((1-q, q)).sum(axis=1), 1)


def test_identity_keeps_endpoints_without_double_sigmoid():
    p = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    np.testing.assert_array_equal(apply_temperature_to_probabilities(p, 1.0), p)


@pytest.mark.parametrize('temperature', [0, -1, np.nan, np.inf, True, '2'])
def test_invalid_temperature_is_rejected(temperature):
    with pytest.raises(ValueError):
        apply_temperature_to_probabilities(np.array([0.1, 0.9]), temperature)


def test_fit_softens_errors_without_worsening_fit_nll():
    p = np.tile([0.01, 0.99, 0.01, 0.99], 10)
    y = np.tile([0, 0, 1, 1], 10)
    fit = fit_temperature(p, y)
    assert fit['temperature'] > 1
    assert fit['log_loss_after'] <= fit['log_loss_before'] + 1e-12
    assert fit['count'] == 40
    assert fit['class_counts'] == [20, 20]


@pytest.mark.parametrize('labels', [np.zeros(20), np.ones(20), np.array([0, 1])])
def test_fit_rejects_unsupported_calibration_samples(labels):
    with pytest.raises(ValueError):
        fit_temperature(np.full(labels.size, 0.5), labels)
```

Also parametrize probability inputs `[]`, `[[0.1],[0.9]]`, `[nan,0.5]`, `[-0.1,0.5]`, `[0.5,1.1]`, mismatched label lengths and nonbinary labels; all must fail. For artifact tests, use arbitrary checkpoint bytes and a real valid schema 2 manifest built by `build_manifest` with at least 20 validation rows; no TensorFlow is needed. Save a fit computed from those validation labels. Assert round-trip fields through `data['fit']`, then tamper separately with checkpoint bytes, manifest bytes, input dimension, method, class order, temperature, sample count, class counts and before/after loss; loader returns None. Missing file returns None; duplicate save raises `FileExistsError`. Simulate optimizer failure and assert no artifact can be saved from a failed fit.

- [ ] Run the tests; confirm missing-module/interface failures.
- [ ] Implement numeric validation before any cast that could hide invalid values. Reject bool/string temperatures. Check shapes, nonempty arrays, binary finite labels, finite [0,1] probabilities and fit count >=20 with both classes. Use clipping only after validation.

```python
EPS = 1e-12
MIN_T, MAX_T = 0.05, 10.0

def apply_temperature_to_probabilities(probabilities, temperature):
    p = _validated_probabilities(probabilities)
    T = _validated_temperature(temperature)
    if T == 1.0:
        return p.copy()
    clipped = np.clip(p, EPS, 1.0 - EPS)
    logits = np.log(clipped) - np.log1p(-clipped)
    return expit(logits / T)
```

Define `_validated_probabilities` and `_validated_temperature` in this module with the checks just specified. General application accepts any finite positive T; loading/saving this schema restricts it to the fit domain. Import `expit` from `scipy.special`. Public functions never guess whether an input is a probability or logit.

- [ ] Fit in log-temperature space using `scipy.optimize.minimize_scalar(method='bounded')`. Derive logits from clipped probabilities, optimize stable Bernoulli NLL, check success and finiteness, compare against identity, and retain T=1 for non-improvement/ties within 1e-12.

```python
logits = np.log(np.clip(p, EPS, 1-EPS)) - np.log1p(-np.clip(p, EPS, 1-EPS))
def objective(log_temperature):
    z = logits / np.exp(log_temperature)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))
result = minimize_scalar(objective, bounds=(np.log(MIN_T), np.log(MAX_T)),
                         method='bounded')
if not result.success or not np.isfinite(result.fun):
    raise ValueError('temperature optimization failed')
T = float(np.exp(result.x))
# Score exactly the public transform, including its identity behavior.
before = float(log_loss(y, p, labels=[0, 1]))
after = float(log_loss(y, apply_temperature_to_probabilities(p, T), labels=[0, 1]))
if after >= before - 1e-12:
    T, after = 1.0, before
```

Return `temperature`, `status='fitted'`, `partition='validation'`, `count`, integer `class_counts`, `optimizer_success=True`, `boundary_hit` (within 1e-3 in log-space of either bound), `log_loss_before`, `log_loss_after`. A boundary hit is recorded, not misreported as an unconstrained optimum. No assertion of improved test performance follows from this fit.

- [ ] Implement strict artifact save/load using JSON `allow_nan=False`. Enforce exactly the method/head/order/feature policy, positive integer dimension, valid hashes, finite temperature in the declared domain, successful fit status, at least 20 samples, two positive integer class counts summing to count, finite nonnegative losses and `after <= before + 1e-12`. Require validation count to match the manifest validation membership; require both manifest fingerprints and byte hash to match. Hash checkpoint contents, not its path. Reject bools where integers/numbers are expected.

Write via exclusive creation (`'x'`) only after all payload validation and serialization succeeds. Partial/truncated artifacts must load as invalid. Catch expected I/O, JSON, schema/type errors in the loader, log the reason and return None; do not catch exceptions around Keras model loading here.

```python
def invalidate_calibration(model, checkpoint_path):
    model.is_calibrated = False
    model.calibration_temperature = None
    model.calibration_metadata = None
    Path(checkpoint_path).with_suffix('.calibration.json').unlink(missing_ok=True)
```

- [ ] Run tests without TensorFlow or database writes.

```powershell
.\.venv312\Scripts\python.exe -m pytest tests/test_model_calibration.py tests/test_experiment.py -q
git diff --check
```

Suggested commit: `feat(calibration): fit binary temperatures and verify artifact provenance`.

---

### Task 5: Calibrate the selected candidate and retain raw evaluation

**Files:** Modify `scripts/train_pre_match_model.py`, `cli.py`, `tests/test_pre_match_run.py`, `tests/test_config_cli.py`, `docs/pre_match_training.md`.

**Interfaces:** `train_model` gains `calibrate: bool = False`; both CLIs gain `--calibrate`. Existing `metrics` remains raw; optional `calibrated_metrics` uses the identical three-argument metrics API on transformed probabilities. A calibrated run emits `model.calibration.json`; a default run emits no sidecar.

- [ ] Extend `tests/test_pre_match_run.py` using its existing `_records` and `_tiny_model` helpers. Preserve the old n=40 default test and its false flag. Add this opt-in test with n=200 (30 validation rows):

```python
def test_calibrated_candidate_keeps_raw_metrics_and_bound_artifact(tmp_path, monkeypatch):
    from scripts import train_pre_match_model as training
    from core.calibration import load_calibration_artifact
    monkeypatch.setattr(training, 'build_multi_output_model', _tiny_model)
    output = tmp_path / 'calibrated.keras'
    model = training.train_model(_records(200), epochs=1, batch_size=16, dim=16,
                                 checkpoint_path=str(output), calibrate=True)
    report = json.loads(output.with_suffix('.metrics.json').read_text())
    artifact = load_calibration_artifact(output, input_dim=16)
    assert artifact is not None
    assert report['is_calibrated'] is True
    assert model.is_calibrated is True
    assert report['calibrated_heads'] == ['winner']
    assert report['calibrated_metrics']['test']['baseline'] == report['metrics']['test']['baseline']
    assert report['metrics']['test']['model']['accuracy'] == report['calibrated_metrics']['test']['model']['accuracy']
    assert artifact['fit']['count'] == 30
```

- [ ] Add spies for `_winner_probabilities` and `fit_temperature`: fit receives only the selected model's validation labels/probabilities; test inference is called once and its cached vector supplies both raw and calibrated reports. Test a pre-existing calibration sidecar alone blocks output before model construction. Test `calibrate=True` with fewer than 20 validation rows or one validation class fails before model building/output creation. Test a simulated fit failure leaves no final checkpoint, sidecar or successful report, while the reserved split manifest may remain for diagnosis.
- [ ] Run these tests and confirm the missing `calibrate` argument/behavior failures.
- [ ] Add the sidecar path to the no-overwrite set regardless of flag. Also forbid output equal to the configured active `NNRIOT_CHECKPOINT_PATH`, in addition to `scripts/model_v2.keras`. The current continuous trainer passes an environment path through unchanged, so a relative value resolves against the process working directory: use `Path(value).resolve()` for the same comparison, and document an absolute path for activation. Do not import the continuous trainer just to resolve this path. When calibration is requested, check its validation sample prerequisites before creating directories or building TensorFlow models.
- [ ] After the best epoch is restored, cache raw probabilities for train and validation. Fit from validation, then obtain test probabilities once. Use the same cache for both metric views. Maintain validation epoch selection on raw log-loss.

```python
raw = {name: _winner_probabilities(model, x[name], batch_size)
       for name in ('train', 'validation')}
fit = fit_temperature(raw['validation'], y['validation']) if calibrate else None
raw['test'] = _winner_probabilities(model, x['test'], batch_size)
metrics = {name: evaluate_winner(y[name], raw[name], train_rate)
           for name in ('train', 'validation', 'test')}
calibrated_metrics = None
if fit is not None:
    calibrated_metrics = {name: evaluate_winner(y[name],
        apply_temperature_to_probabilities(raw[name], fit['temperature']), train_rate)
        for name in ('train', 'validation', 'test')}
```

- [ ] Finalize checkpoint bytes before creating its bound sidecar. Keep the existing fresh-path checks and write the report last as the completion marker. Hashes depend on finalized checkpoint/manifest bytes. A failed run keeps diagnostic artifacts but must be rerun into a fresh directory. Do not pretend that individually written files form a crash-atomic transaction.

```python
candidate.replace(output)
calibration_path = None
if fit is not None:
    calibration_path = save_calibration_artifact(
        output, manifest_path, fit, input_dim=dim)
    verified = load_calibration_artifact(output, input_dim=dim)
    if verified is None:
        raise ValueError('written calibration artifact failed verification')
    model.is_calibrated = True
    model.calibration_temperature = verified['temperature']
    model.calibration_metadata = verified
else:
    model.is_calibrated = False
    model.calibration_temperature = None
    model.calibration_metadata = None
report.update(is_calibrated=model.is_calibrated,
              calibrated_heads=['winner'] if fit is not None else [],
              calibrated_metrics=calibrated_metrics,
              calibration_artifact=str(calibration_path) if calibration_path else None)
_write_json(report_path, report)
```

Keep raw model outputs unchanged: `model.predict` still returns raw Keras probabilities; the verified attributes are consumed by the serving wrapper in Task 6. Update docstrings and report limitations conditionally: no-calibration for default runs; validation reuse and winner-only calibration for opt-in runs. No statement that fitted temperature makes the predictor useful.

- [ ] Forward `--calibrate` only when requested and add a CLI forwarding assertion. Update the tree comparison reader to show `calibrated_metrics` separately, already defined in Task 3.

```powershell
.\.venv312\Scripts\python.exe -m pytest tests/test_pre_match_run.py tests/test_config_cli.py tests/test_model_calibration.py tests/test_tree_baseline.py -q
git diff --check
```

Suggested commit: `feat(training): emit verified optional winner calibration for candidates`.

---

### Task 6: Load honestly, serve once, and invalidate before mutation

**Files:** Modify `core/model_factory.py`, `scripts/continuous_trainer.py`, `app/core.py`, `app/routes/api.py`, `core/calibration.py`, `tests/test_model_calibration_guard.py`, `tests/test_predict_api.py`, `tests/test_multi_output_trainer.py`, `tests/test_api_train_and_stream.py`, `docs/pre_match_training.md`; create `tests/test_calibrated_serving.py`.

**Interfaces:**

- `core.calibration.UncalibratedModelError(RuntimeError)`: explicit unavailable-serving condition.
- Loaded v2 model attributes: `is_calibrated: bool`, `calibration_temperature: float | None`, `calibration_metadata: dict | None`.
- `ContinuousTrainer.predict(x_dense)` returns a new per-head dict with winner calibrated once; raises `UncalibratedModelError` if the model is not verified. Raw offline inference remains `trainer.model.predict`, never advertised as calibrated.
- `ContinuousTrainer` owns a `threading.RLock`; both training and prediction use it. This is separate from the existing web lock, which does not protect training.

- [ ] Replace `test_model_factory_marks_loaded_valid_model_as_calibrated` with two explicit cases: a valid checkpoint without sidecar loads the same weights and stays false; a valid checkpoint with the Task 4 bound sidecar loads true. Monkeypatch the architecture builder to fail in the missing/invalid-sidecar cases so the test proves no unnecessary rebuild occurred. Prefer the existing tiny multi-output test builder with a small input dimension to avoid repeatedly constructing the full model.
- [ ] Add a trainer-level test with fake raw Keras output; no database or TensorFlow initialization is required because `__new__` bypasses construction.

```python
import threading
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.continuous_trainer import ContinuousTrainer
from core.calibration import UncalibratedModelError


def test_trainer_calibrates_winner_once_and_preserves_other_heads():
    raw = {'winner': np.array([[0.9, 0.1], [0.1, 0.9]]),
           'total_kills': np.array([[1.0], [2.0]])}
    trainer = ContinuousTrainer.__new__(ContinuousTrainer)
    trainer._model_lock = threading.RLock()
    trainer.model = SimpleNamespace(is_calibrated=True, calibration_temperature=2.0,
        predict=lambda x, verbose=0: raw)
    out = trainer.predict(np.zeros((2, 16)))
    np.testing.assert_allclose(out['winner'], [[0.75, 0.25], [0.25, 0.75]])
    np.testing.assert_array_equal(raw['winner'], [[0.9, 0.1], [0.1, 0.9]])
    np.testing.assert_array_equal(out['total_kills'], raw['total_kills'])
    trainer.model.is_calibrated = False
    with pytest.raises(UncalibratedModelError):
        trainer.predict(np.zeros((2, 16)))
```

- [ ] Add failed-mutation coverage using a fake repository with an eligible batch and a fake model whose first `train_on_batch` raises. Assert the sidecar is already absent and the flag false **inside** that method, then assert they stay invalid afterward. Simulate sidecar removal permission failure and assert `train_on_batch` is never called. An empty batch must retain its sidecar and calibration. Reuse `_full_labels` patterns in `tests/test_multi_output_trainer.py`; do not contact Riot or a live DB.
- [ ] Add a deterministic lock test using `threading.Event` rather than arbitrary sleeps: the training fake blocks after invalidation; a prediction thread must not enter the fake model while training owns the lock; after training releases, prediction raises uncalibrated. Always release the event and join threads in `finally`, with a bounded timeout to prevent hanging pytest.
- [ ] In `tests/test_predict_api.py`, replace successful `MagicMock` calibration assumptions with explicit `fake_trainer.model.is_calibrated=True`. Reuse `_fake_preds()` for both endpoints with `ContinuousTrainer.predict` bound to a fake raw model. Set T=2 and assert each route receives the same transformed winner pair; compare formatted results within 0.001 rounding, preserve all non-winner response fields, and test unavailable calibration produces the existing error response instead of a success. Keep standalone fake trainers intentional: mocks must not count as verification evidence in production code.
- [ ] Add feature parity tests: spy on vectorization for both endpoints, supply final match stats and career aggregates, and assert the shared normalized input contains champion/role/context only. Preserve existing API payload acceptance; this task does not introduce a new five-player payload requirement into the legacy match endpoint.
- [ ] Run the affected tests and confirm behavior failures before changes.

**Load verified metadata without rebuilding good weights.**

- [ ] In `ModelFactory.get_model`, initialize all three calibration attributes to false/None. After structural validation of loaded v2 weights, call the loader and attach only verified metadata. Keep calibration errors outside the block that triggers architecture rebuilding. For v1 models, retain weight loading but mark uncalibrated; no v1 calibration support is introduced.

```python
candidate.is_calibrated = False
candidate.calibration_temperature = None
candidate.calibration_metadata = None
metadata = load_calibration_artifact(path, input_dim=input_dim) if version == 'v2' else None
if metadata is not None:
    candidate.is_calibrated = True
    candidate.calibration_temperature = metadata['temperature']
    candidate.calibration_metadata = metadata
```

Freshly built models get the same false/None defaults. Do not trust a serialized truthy string or the mere presence of a JSON file.

**Serialize training and inference around the same model.**

- [ ] Initialize `_model_lock = threading.RLock()` in `ContinuousTrainer.__init__`. Keep the public `run_training_step()` signature and move its existing body to `_run_training_step_locked()`, preserving repository and batching logic. Within that body, call `invalidate_calibration(self.model, self.checkpoint_path)` after all no-work exits and successful batch/target preparation, immediately before any first `train_on_batch`. Do not restore calibration in `finally` or after saving.

```python
def run_training_step(self):
    with self._model_lock:
        return self._run_training_step_locked()


def predict(self, x_dense):
    with self._model_lock:
        if getattr(self.model, 'is_calibrated', False) is not True:
            raise UncalibratedModelError('No verified winner calibration is available')
        temperature = self.model.calibration_temperature
        raw = self.model.predict(x_dense, verbose=0)
        winner = np.asarray(raw['winner'], dtype=float)
        if (winner.ndim != 2 or winner.shape[1] != 2 or
                not np.isfinite(winner).all() or (winner < 0).any() or
                (winner > 1).any() or not np.allclose(winner.sum(axis=1), 1.0, atol=1e-6)):
            raise ValueError('winner output must contain normalized [A, B] probabilities')
        p_b = apply_temperature_to_probabilities(winner[:, 1], temperature)
        output = dict(raw)
        output['winner'] = np.column_stack((1.0 - p_b, p_b))
        return output
```

The temperature loader and application helper already validate numeric temperature; do not compute a second sigmoid in routes or formatters. Inspect all callers of `ContinuousTrainer.predict`; raw-model unit checks may use `trainer.model.predict`, but user-facing serving must not bypass the guard.

**Keep inference features and error mapping consistent.**

- [ ] In `final_predict_match_outcome`, replace `extract_match_features(match_data)` with the shared `to_pre_match_features(match_data)`. In the custom route, pass its synthetic raw match through the same normalizer before vectorization. Retain UI/player data fetching unless a separate change is requested; strip undated stats at the ML boundary.
- [ ] Make early web checks default to uncalibrated when attributes are missing. The check inside the trainer lock is authoritative, since training can start after a route's early check. Catch `UncalibratedModelError` separately: legacy match helper returns `{'error': message, 'uncalibrated': True}` for the route's existing mapping; custom prediction returns this body with HTTP 503. Keep other errors mapped as before.
- [ ] Update the misleading custom-route text that equates uncalibrated with untrained: a trained checkpoint can lack a valid calibration. Preserve `uncalibrated: true` as the machine-readable flag. Do not change existing confidence formulas or response keys in this scope.

**Document the complete candidate bundle without performing activation.**

- [ ] Document that `model.keras`, `model.split.json`, `model.calibration.json` and `model.metrics.json` form a candidate bundle. For a later explicitly authorized activation, choose a completed immutable bundle and configure `NNRIOT_CHECKPOINT_PATH` to its `.keras` file, then restart the app so the factory verifies siblings. This avoids partial copying into the default active filename. Do not run a promotion command or change `.env` during this task. Legacy checkpoints without a verified sidecar intentionally become unavailable for calibrated prediction.
- [ ] Document that invoking continuous training invalidates calibration before mutation; serving stays unavailable until a newly evaluated/calibrated bundle is loaded. One winner sidecar does not validate auxiliary heads. Keep the existing API schema while making this limitation explicit in documentation.

```powershell
.\.venv312\Scripts\python.exe -m pytest tests/test_calibrated_serving.py tests/test_model_calibration_guard.py tests/test_predict_api.py tests/test_multi_output_trainer.py tests/test_api_train_and_stream.py tests/test_pre_match_features.py -q
git diff --check
```

**Calibration gate:** both endpoints consume calibrated winner probabilities once, invalid metadata never enables serving, and a failing training step cannot retain old calibration. Suggested commit: `fix(serving): verify winner calibration and invalidate it before training`.

---

### Task 7: Add honest diagnostics without breaking historical metrics

**Files:** Modify `core/training_metrics.py`, `scripts/train_pre_match_model.py`, `scripts/train_tree_baseline.py`, `docs/pre_match_training.md`; create `tests/test_training_metrics_calibration.py`; retain existing `tests/test_training_metrics.py` expectations.

**Interfaces:**

- `evaluate_winner(y_true, probabilities, train_positive_rate: float, *, diagnostics: bool = False) -> dict`. With the default, its full output remains unchanged.
- `compute_calibration_diagnostics(y_true, probabilities, *, num_bins: int = 10) -> dict`: validated one-dimensional inputs, explicit quantile and binned metrics below.
- Private `_quantile_groups(probabilities, num_bins) -> list[np.ndarray]` and `_calibration_slope(labels, probabilities) -> dict` in the same module.
- With `diagnostics=True`, add `diagnostics` inside both `model` and `baseline`, each calculated on its own probabilities. Do not add or rename top-level result keys.

**Diagnostic schema:**

```python
{
    'quantile_ece': 0.0,
    'binning': {'method': 'quantile', 'requested_bins': 10, 'effective_bins': 1},
    'brier_decomposition': {
        'reliability': 0.0, 'resolution': 0.0, 'uncertainty': 0.25,
        'binned_brier': 0.25, 'binning_residual': 0.0,
        'identity_target': 'binned_probabilities',
    },
    'calibration_slope': {'slope': None, 'intercept': None,
                          'status': 'unavailable', 'reason': 'constant_predictions'},
}
```

These are illustrative values. Binned Brier, not raw Brier, equals reliability - resolution + uncertainty. The signed residual is `raw_brier - binned_brier`; it is not a nonnegative calibration error. The original eight-observation example has raw 0.045, binned 0.040 and residual 0.005 **when using two bins**. Quantile binning with tied probabilities may produce fewer effective bins; report that number and never split equal predictions arbitrarily just to force equal counts.

- [ ] Add exact numerical and compatibility tests:

```python
import json
import numpy as np
import pytest
from core.training_metrics import evaluate_winner, compute_calibration_diagnostics


def test_brier_identity_is_explicitly_for_binned_predictions():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.2, 0.3, 0.7, 0.8, 0.8, 0.9])
    d = compute_calibration_diagnostics(y, p, num_bins=2)['brier_decomposition']
    assert np.mean((p-y)**2) == pytest.approx(0.045)
    assert d['binned_brier'] == pytest.approx(0.04)
    assert d['binning_residual'] == pytest.approx(0.005)
    assert d['binned_brier'] == pytest.approx(d['reliability'] - d['resolution'] + d['uncertainty'])


def test_opt_in_diagnostics_preserve_legacy_values():
    old = evaluate_winner([0, 1, 0, 1], [0.2, 0.8, 0.4, 0.6], 0.5)
    new = evaluate_winner([0, 1, 0, 1], [0.2, 0.8, 0.4, 0.6], 0.5, diagnostics=True)
    assert set(new) == set(old)
    for part in ('model', 'baseline'):
        assert {k: new[part][k] for k in old[part]} == old[part]
        assert 'diagnostics' in new[part]
    json.dumps(new, allow_nan=False)


def test_constant_predictions_have_unavailable_slope_not_ideal_slope():
    d = compute_calibration_diagnostics([0, 1] * 20, [0.5] * 40)
    assert d['binning']['effective_bins'] == 1
    assert d['calibration_slope']['slope'] is None
    assert d['calibration_slope']['intercept'] is None
    assert d['calibration_slope']['reason'] == 'constant_predictions'
```

- [ ] Parametrize diagnostics for one-class labels, constant probabilities, exact 0/1, tied quantiles, num_bins=1, invalid num_bins (0, negative, bool, noninteger), empty arrays and invalid probabilities. Reuse the existing validation expectations without truncating fractional labels to integers. Add slope failure/convergence-warning tests via a monkeypatched `LogisticRegression.fit`; assert null values and a nonempty reason. Add a perfectly separated-label case and assert `reason='separation'`.
- [ ] Run the new tests and confirm missing diagnostics interfaces.
- [ ] Refactor current input validation into a private helper if needed, preserving error cases; do not import `core.calibration` back into metrics. Build quantile groups from unique interior quantiles with `np.searchsorted(..., side='left')`. Remove empty groups and report effective count. Use all requested bins, capped by sample count; no undocumented n//5 or n//10 heuristic.

```python
def _quantile_groups(probabilities, num_bins):
    if type(num_bins) is not int or num_bins < 1:
        raise ValueError('num_bins must be a positive integer')
    count = min(num_bins, len(probabilities))
    edges = np.unique(np.quantile(probabilities, np.linspace(0, 1, count + 1)[1:-1]))
    indices = np.searchsorted(edges, probabilities, side='left')
    return [mask for i in range(len(edges) + 1)
            if (mask := indices == i).any()]
```

- [ ] Compute group statistics directly, with explicit binned reconstruction and raw residual:

```python
base_rate = float(labels.mean())
uncertainty = base_rate * (1 - base_rate)
reliability = resolution = quantile_ece = 0.0
binned_predictions = np.empty_like(probabilities, dtype=float)
for mask in groups:
    weight = float(mask.mean())
    mean_p = float(probabilities[mask].mean())
    mean_y = float(labels[mask].mean())
    reliability += weight * (mean_p - mean_y)**2
    resolution += weight * (mean_y - base_rate)**2
    quantile_ece += weight * abs(mean_p - mean_y)
    binned_predictions[mask] = mean_p
binned_brier = float(np.mean((binned_predictions - labels)**2))
raw_brier = float(np.mean((probabilities - labels)**2))
```

Assemble the declared schema using `binning_residual=raw_brier-binned_brier`. This identity tests an independently computed binned score, not a score reconstructed from the same three terms and then asserted equal to itself.

- [ ] Implement slope diagnostics with `LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)` on clipped binary log-odds. Return unavailable with `single_class`, `constant_predictions`, or `separation` before fitting when applicable. Separation includes either class's entire logit range lying strictly above the other's; for touching ranges, handle convergence warnings as unavailable. Convert `ConvergenceWarning` to an exception with `warnings.catch_warnings`; handle known numerical/fit errors, nonfinite coefficients and failure to converge with reason `fit_failed`. Do not catch every arbitrary programming exception and replace it with (1,0). A successful fit returns measured finite slope/intercept and `status='estimated', reason=None`.
- [ ] Add diagnostics only to final raw/calibrated train/validation/test reports in both trainers. Keep epoch/candidate selection calls on the unchanged metrics path. Document that slope/quantile ECE are descriptive estimates, not proof that a calibrator was fitted and not a model-promotion gate.

```powershell
.\.venv312\Scripts\python.exe -m pytest tests/test_training_metrics.py tests/test_training_metrics_calibration.py tests/test_pre_match_run.py tests/test_tree_baseline.py -q
git diff --check
```

Suggested commit: `feat(metrics): add explicit binned calibration diagnostics without API changes`.

## Final verification and handoff

- [ ] Review every changed file, then run the targeted integration suite. Start with the pure data/metric/calibration tests, then run TensorFlow and endpoint tests. Preserve the existing `tests/conftest.py` isolation of database and checkpoint paths.

```powershell
$env:TF_NUM_INTRAOP_THREADS = '2'
$env:TF_NUM_INTEROP_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv312\Scripts\python.exe -m pytest tests/test_experiment.py tests/test_patch_stratified_split.py tests/test_pre_match_dataset.py tests/test_training_metrics.py tests/test_training_metrics_calibration.py tests/test_model_calibration.py tests/test_tree_baseline.py -q
.\.venv312\Scripts\python.exe -m pytest tests/test_pre_match_training.py tests/test_pre_match_run.py tests/test_calibrated_serving.py tests/test_model_calibration_guard.py tests/test_predict_api.py tests/test_multi_output_trainer.py tests/test_api_train_and_stream.py tests/test_config_cli.py tests/test_pre_match_features.py -q
.\.venv312\Scripts\python.exe cli.py train --help
.\.venv312\Scripts\python.exe scripts/train_tree_baseline.py --help
git diff --check
git status --short
```

- [ ] Verify API success payload snapshots/field assertions remain intact, input allowlisting is shared, and old artifacts remain untouched. Confirm no new external service or dependency was introduced.
- [ ] Review the implementation against the five Review Focus cases. Review the real diff, not only the new tests. If a failure exposes a missing task requirement, fix the task implementation and rerun the affected checks; do not expand to unrelated refactors.
- [ ] Record actual test counts and any pre-existing failures. State separately: implemented contracts, synthetic evidence, absent production-quality evidence and candidate activation status. Do not report calibration or tree superiority on real data without an explicitly executed experiment.
- [ ] Hand off commands for a later authorized real experiment, with the actual user-selected source DB and new output paths. This implementation does not run that experiment or activate a model. Changing features/settings after inspecting an old test requires a new future holdout before making generalization claims.

## Self-review record for this plan

- Metrics signature and historical keys are preserved by default; both trainers consume the same schema.
- Frozen cohort loading checks actual records and both data/membership fingerprints; newer database rows do not silently redefine a comparison.
- Patch fallback is explicit, and overlap/unknown versions have assigned failure behavior and tests.
- Winner calibration consumes probabilities with a named API; both classes, raw reports and both serving paths are covered.
- Artifact verification binds the finalized checkpoint and manifest; mutation invalidation occurs before any weight update, under the same lock as inference.
- The Brier identity targets binned probabilities, with the omitted contribution reported as a signed residual; undefined slopes are null.
- No real-model lift, production calibration quality, successful implementation test run, or Gemini execution is claimed by this document.

## Primary references

- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html): distinction between calibration and discrimination, independent fitting data, temperature transform.
- [Guo et al., On Calibration of Modern Neural Networks, ICML 2017](https://proceedings.mlr.press/v70/guo17a.html): post-hoc temperature scaling.
- Repository contracts and historical evidence are named above; external references do not override the installed dependency API or the explicit experiment protocol.
