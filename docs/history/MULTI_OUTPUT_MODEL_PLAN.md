# Multi-Output Prediction Model — Design Plan

**Status:** Proposal — no code changes. Approve sections before implementation.

**Goal:** Extend the current single-output (winner) model into a multi-task model that predicts ~17 betting-market signals per match. Series-level markets (Map 1 / Map 2 / exact score) are explicitly **out of scope** for this iteration — they require data the project doesn't have.

---

## 1. Markets in scope

Mapping of the requested list → what we'll predict and where the label comes from.

| # | Market (FR) | Output type | Label source in Riot match JSON |
|---|---|---|---|
| 1 | Pari sur le vainqueur | binary | `teams[].win` |
| 2 | Pari sur le vainqueur (Kills) | binary | `sum(kills) per team`, larger team wins |
| 3 | Handicap (Kills) | regression (signed int) | `team_a_kills - team_b_kills` |
| 4 | Total (Kills) | regression (int) | `sum(kills) all participants` |
| 5 | Total de l'équipe (Kills) — Team A | regression (int) | `sum(kills) where teamId=100` |
| 6 | Total de l'équipe (Kills) — Team B | regression (int) | `sum(kills) where teamId=200` |
| 7 | Total de kills Impair/Pair | binary | `(total_kills % 2) == 1` |
| 8 | 1er sang versé (First Blood) | 3-class | team A, team B, or none |
| 9 | 1er baron | 3-class | `teams[].objectives.baron.first` |
| 10 | 1er inhibiteur | 3-class | `teams[].objectives.inhibitor.first` |
| 11 | 1er tour (bonus, not in list but cheap) | 3-class | `teams[].objectives.tower.first` |
| 12 | Total de barons tués | regression (small int) | `sum(teams[].objectives.baron.kills)` |
| 13 | Total des dragons élémentaires tués | regression (small int) | `sum(teams[].objectives.dragon.kills)` |
| 14 | Total de tourelles détruites | regression (int) | `sum(teams[].objectives.tower.kills)` |
| 15 | Les deux équipes tuent le baron | binary | both `baron.kills >= 1` |
| 16 | Les deux équipes détruisent un inhibiteur | binary | both `inhibitor.kills >= 1` |
| 17 | Les deux équipes tuent un dragon élémentaire | binary | both `dragon.kills >= 1` |
| 18 | L'une des équipes tuera-t-elle le dragon ancien ? | binary | any team's `dragon.kills >= 4` (proxy — elder requires 4 elemental drakes) OR `teams[].objectives.elderDragon.kills >= 1` if present in newer schema |

**Group B (deferred behind flag — needs timeline endpoint):**
First to 5/10/15/20 kills → 4 additional 3-class outputs. Requires fetching `/lol/match/v5/matches/{id}/timeline`, parsing `CHAMPION_KILL` events, accumulating per-team. Doubles Riot API call cost per match. **Not implemented in this phase.** See §9.

**Out of scope (require new data sources):**
- Pari sur le vainqueur — 1re carte / 2e carte
- (Carte 2) markets — these are about Map 2 of a series, no series concept exists today
- Score exact (2-0, 2-1, etc.)
- First to N kills (Group B, deferred)

---

## 2. Label extraction

New module: `feature_labels.py` (single responsibility — converts raw match JSON into a labels dict).

Sketch:

```python
def extract_labels(match_details: dict) -> dict | None:
    """
    Compute all ~17 multi-output labels for a single match.
    Returns None if the match is malformed (no winner / no participants).
    """
    info = match_details.get("info", match_details)
    teams = info.get("teams", [])
    participants = info.get("participants", [])

    # Validation: both teams present, exactly one winner
    team_a = next((t for t in teams if t.get("teamId") == 100), None)
    team_b = next((t for t in teams if t.get("teamId") == 200), None)
    if not team_a or not team_b:
        return None
    if bool(team_a.get("win")) == bool(team_b.get("win")):
        return None  # draw / both wins / both losses — malformed

    a_kills = sum(p.get("kills", 0) for p in participants if p.get("teamId") == 100)
    b_kills = sum(p.get("kills", 0) for p in participants if p.get("teamId") == 200)
    total_kills = a_kills + b_kills

    obj_a = team_a.get("objectives", {})
    obj_b = team_b.get("objectives", {})

    def first_team(key: str) -> int:
        """0=team A, 1=team B, 2=neither"""
        if obj_a.get(key, {}).get("first"): return 0
        if obj_b.get(key, {}).get("first"): return 1
        return 2

    return {
        "winner":                int(team_b.get("win")),                        # 0/1
        "winner_kills":          int(b_kills > a_kills),                        # 0/1 (ties → 0)
        "kill_handicap":         a_kills - b_kills,                             # int
        "total_kills":           total_kills,                                   # int
        "team_a_kills":          a_kills,                                       # int
        "team_b_kills":          b_kills,                                       # int
        "kills_odd":             total_kills % 2,                               # 0/1
        "first_blood":           first_team("champion"),                        # 0/1/2
        "first_baron":           first_team("baron"),                           # 0/1/2
        "first_inhibitor":       first_team("inhibitor"),                       # 0/1/2
        "first_tower":           first_team("tower"),                           # 0/1/2
        "total_barons":          obj_a.get("baron", {}).get("kills", 0) + obj_b.get("baron", {}).get("kills", 0),
        "total_dragons":         obj_a.get("dragon", {}).get("kills", 0) + obj_b.get("dragon", {}).get("kills", 0),
        "total_towers":          obj_a.get("tower", {}).get("kills", 0) + obj_b.get("tower", {}).get("kills", 0),
        "both_baron":            int(obj_a.get("baron", {}).get("kills", 0) >= 1 and obj_b.get("baron", {}).get("kills", 0) >= 1),
        "both_inhibitor":        int(obj_a.get("inhibitor", {}).get("kills", 0) >= 1 and obj_b.get("inhibitor", {}).get("kills", 0) >= 1),
        "both_dragon":           int(obj_a.get("dragon", {}).get("kills", 0) >= 1 and obj_b.get("dragon", {}).get("kills", 0) >= 1),
        "elder_dragon":          int(_any_elder(obj_a) or _any_elder(obj_b)),
    }
```

`_any_elder` reads `objectives.elderDragon.kills` if present (newer Riot schema), falling back to `objectives.dragon.kills >= 4` as a proxy. Need to verify against a real sample — punt to implementation.

**Test data:** `test_data.get_sample_match()` already produces a 2-participant fixture. Extend it to 10 participants + full `teams[].objectives` shape so we can unit-test every label extraction path.

---

## 3. Database schema changes

Two options. Recommendation: **Option B**.

### Option A — store each label as a column on `training_dataset`
- Pros: queryable in SQL, easy to inspect.
- Cons: ~17 new columns + schema migration; schema churn if we add more markets later.

### Option B — single `labels_json` column (compressed dict) ⭐
- Pros: one column, easy to extend without migrations, fits the existing `feature_json` pattern.
- Cons: not SQL-queryable; need to deserialize for analytics.

Migration v9 → v10:
```sql
ALTER TABLE training_dataset ADD COLUMN labels_json TEXT;
```

`save_training_record` and `save_training_records_batch` get a `labels: dict` parameter. The existing `winner_label` column stays for backward compatibility (it's the same value as `labels["winner"]`).

### Backfill
New script `backfill_labels.py` (~80 LOC):
1. Iterate `matches.raw_json`, decode (gzip+b64).
2. Call `extract_labels(...)`.
3. UPDATE the corresponding `training_dataset` row's `labels_json`.
4. Records where `extract_labels` returns None get flagged or deleted.

Idempotent. Re-runnable. Logged progress every 1000 rows.

---

## 4. Model architecture

Replace `generate_graph.build_keras_model` with a multi-head version.

**Shared trunk (unchanged):**
- Input (VECTOR_DIM = 100000)
- Dense 1024 ReLU
- ResBlock 1024 ×2
- Dense 512 ReLU
- Dense 128 ReLU  ← shared embedding

**Per-task heads (off the 128-dim embedding):**

| Head name | Layers | Activation | Loss |
|---|---|---|---|
| `winner` | Dense 2 | softmax | categorical_crossentropy |
| `winner_kills` | Dense 2 | softmax | categorical_crossentropy |
| `kills_odd` | Dense 1 | sigmoid | binary_crossentropy |
| `both_baron` / `both_inhibitor` / `both_dragon` / `elder_dragon` | Dense 1 | sigmoid | binary_crossentropy (×4) |
| `first_blood` / `first_baron` / `first_inhibitor` / `first_tower` | Dense 3 | softmax | categorical_crossentropy (×4) |
| `total_kills` / `total_barons` / `total_dragons` / `total_towers` | Dense 1 | linear | MSE (×4) |
| `team_a_kills` / `team_b_kills` | Dense 1 | linear | MSE (×2) |
| `kill_handicap` | Dense 1 | linear | MSE |

Total: 17 heads.

Keras model definition:

```python
def build_multi_output_model(input_dim=100000):
    inputs = tf.keras.Input(shape=(input_dim,), name="Input")
    x = tf.keras.layers.Dense(1024, activation="relu")(inputs)
    # ... two residual blocks identical to current ...
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    embedding = tf.keras.layers.Dense(128, activation="relu", name="embedding")(x)

    outputs = {
        "winner": tf.keras.layers.Dense(2, activation="softmax", name="winner")(embedding),
        "winner_kills": tf.keras.layers.Dense(2, activation="softmax", name="winner_kills")(embedding),
        # ... etc
        "total_kills": tf.keras.layers.Dense(1, activation="linear", name="total_kills")(embedding),
        # ...
    }

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=LOSS_PER_HEAD,           # dict: name -> loss fn
        loss_weights=LOSS_WEIGHTS,    # dict: name -> float, see §5
        metrics=METRICS_PER_HEAD,
    )
    return model
```

**Backward compatibility:** the existing single-output `model.keras` checkpoint is **incompatible** with the new architecture. We bump the checkpoint filename to `model_v2.keras`. If `model_v2.keras` doesn't exist, the trainer initializes from scratch — same fallback as today. We do NOT try to hot-swap the old checkpoint; users either train fresh or run the backfill + trainer once.

---

## 5. Loss weighting

Different heads have wildly different scales (binary 0/1 vs `total_kills` ~30-40 vs `kill_handicap` ~-30 to +30). Without weighting, the regression losses dominate.

**Strategy:** normalize regression targets to roughly unit variance at training time, then use uniform `loss_weights=1.0` everywhere. Alternatively, hand-tune weights via the following starting point:

```python
LOSS_WEIGHTS = {
    "winner":             1.0,
    "winner_kills":       1.0,
    "kills_odd":          0.5,
    "first_blood":        0.7,
    "first_baron":        0.7,
    "first_inhibitor":    0.7,
    "first_tower":        0.5,
    "both_baron":         0.4,
    "both_inhibitor":     0.4,
    "both_dragon":        0.4,
    "elder_dragon":       0.3,
    "total_kills":        0.05,   # int ~30, MSE ~9 untuned → scale down
    "total_barons":       0.5,
    "total_dragons":      0.3,
    "total_towers":       0.05,
    "team_a_kills":       0.05,
    "team_b_kills":       0.05,
    "kill_handicap":      0.05,
}
```

These need empirical tuning after first training run — bake the table into `generate_graph.py` and treat as hyperparameter.

---

## 6. Training loop changes (`continuous_trainer.py`)

1. `run_training_step` reads `labels_json` instead of (or in addition to) `winner_label`.
2. Build a target dict instead of a single `y_data` array:
   ```python
   targets = {head: np.zeros((n,), dtype=...) for head in HEADS}
   for i, r in enumerate(records):
       labels = r["labels_json"]
       for head in HEADS:
           targets[head][i] = labels[head]
   # one-hot the categoricals
   targets["winner"] = tf.keras.utils.to_categorical(targets["winner"], 2)
   # ...
   ```
3. `model.train_on_batch(x_batch, target_batch_dict)` — Keras already supports dict targets when the model was built with named outputs.
4. Log per-head losses + accuracy.

The micro-batch generator stays unchanged.

---

## 7. Prediction API changes (`final_web_app.py`)

### `/api/predict` and `/api/predict/custom`
Response shape today (binary):
```json
{"success": true, "predicted_outcome": "WIN", "win_probability": 0.62, "confidence": 0.62}
```

New shape:
```json
{
  "success": true,
  "winner": {"team_a": 0.38, "team_b": 0.62, "predicted": "B", "confidence": 0.62},
  "winner_kills": {"team_a": 0.41, "team_b": 0.59, "predicted": "B"},
  "kills": {
    "total": 36.4,
    "team_a": 16.1,
    "team_b": 20.3,
    "handicap": -4.2,
    "odd_probability": 0.51
  },
  "first": {
    "blood":     {"team_a": 0.48, "team_b": 0.50, "none": 0.02},
    "baron":     {"team_a": 0.42, "team_b": 0.45, "none": 0.13},
    "inhibitor": {"team_a": 0.40, "team_b": 0.43, "none": 0.17},
    "tower":     {"team_a": 0.49, "team_b": 0.49, "none": 0.02}
  },
  "totals": {"barons": 1.8, "dragons": 3.6, "towers": 14.2},
  "both_teams": {"baron": 0.22, "inhibitor": 0.41, "dragon": 0.93},
  "elder_dragon": 0.18
}
```

The `predicted_outcome`/`win_probability` fields stay on the top level too for one release as a backward-compat shim (`winner.predicted == "B" ? "LOSE" : "WIN"`), then removed.

### Frontend
`templates/predictor.html` needs a results panel rewrite to display all the new probabilities. Out of scope for this plan — separate UI task once the API is stable.

---

## 8. Phased rollout

| Phase | Scope | Estimated effort | Dependencies |
|---|---|---|---|
| **P1: Labels + DB** | `feature_labels.py`, migration v9→v10, `save_training_record(s)` updates, `backfill_labels.py`, unit tests for label extraction | 0.5 day | none |
| **P2: Multi-head model** | `generate_graph.build_multi_output_model`, normalization for regression targets, loss weights table | 0.5 day | P1 |
| **P3: Trainer** | `continuous_trainer.py` updated to fit dict targets, per-head logging, `model_v2.keras` filename | 0.5 day | P2 |
| **P4: API** | `final_web_app.py` `/api/predict` and `/api/predict/custom` return the new dict shape (with compat shim) | 0.5 day | P3 |
| **P5: Backfill + first training run** | Run `backfill_labels.py` over existing DB, then run `continuous_trainer.main()` for one cycle, evaluate per-head metrics | 0.5 day (mostly waiting) | P4 |
| **P6 (optional, future): UI** | Predictor page redesign for the new market panel | 1 day | P5 |
| **P7 (optional, future): Timeline ingestion** | `/lol/match/v5/matches/{id}/timeline` fetcher, schema for timeline events, first-to-N-kills labels | 2-3 days | P5 |

Total P1–P5: ~2.5 days of focused work.

---

## 9. Open questions & risks

1. **Elder dragon detection** — verify whether current Riot schema exposes `objectives.elderDragon.kills` or whether we need the `dragon.kills >= 4` proxy. Quick sample-data check at implementation time.
2. **Regression head bias** — `total_kills` model output will likely converge to dataset mean (~30). Need to verify it's actually learning per-match variance, not just predicting the mean. Track variance of predictions on a held-out set.
3. **Loss weight tuning** — initial weights are guesses. May need 2-3 training cycles to balance. Recommend logging `val_<head>_loss` and `val_<head>_metric` to TensorBoard or stdout so we can spot collapsing heads.
4. **Backfill cost** — ~120 matches in the DB today (per the design review notes). Backfill is trivial. At 100k matches it'd take ~30s; fine.
5. **Catastrophic forgetting** — current trainer uses experience replay. Still works for multi-output, but we need to make sure all heads benefit from replay, not just the winner head. Targets are constructed uniformly so this should be fine.
6. **API compat window** — how long do we keep `predicted_outcome`/`win_probability` as top-level fields? Suggest one full deploy, then remove in the next release with a CHANGELOG note.
7. **Missing labels in old rows** — backfill might fail for some matches (corrupt JSON, missing teams). Those rows should be either deleted from `training_dataset` or flagged with `labels_json = NULL` and filtered out by the trainer. Recommend filter-out.
8. **Series markets (future)** — when/if we want them, we need either:
   - Liquipedia tournament results (scraping, pro-only)
   - A paid esports feed (Bayes, Grid, Sportradar)
   - Manually-curated CSV from the user
   None of these fit into this iteration.

---

## 10. What "done" looks like for this iteration

- ✅ DB schema v10 with `labels_json` column.
- ✅ All existing matches backfilled (or flagged).
- ✅ `model_v2.keras` trained for at least one full pass.
- ✅ `/api/predict` returns the new multi-output dict.
- ✅ Existing winner prediction accuracy is no worse than before (per-head regression test).
- ✅ Each new head reports a non-trivial loss (i.e. not stuck at random / not collapsed to a constant).
- ✅ `tests/test_feature_labels.py` covers the label extraction across normal/edge/malformed matches.
- ❌ Frontend redesign (P6 — future).
- ❌ Timeline / first-to-N-kills (P7 — future).
- ❌ Series markets (out of scope entirely).

---

**Next step:** approve sections 1-8 (or call out changes), then I'll implement P1 first and ship it as one commit before moving to P2.

---

## 11. Running P5 in production

The smoke test (`tests/test_p5_smoke.py`) verifies the pipeline wiring with synthetic data. To execute P5 against the real database, follow these steps. **Back up the DB before starting.**

### Step 1 — Back up the database
```powershell
copy training_data.db training_data.backup-$(Get-Date -Format yyyyMMdd).db
```

### Step 2 — Apply the schema migration
The migration is idempotent and only adds a NULLABLE column. Safe to run on a populated DB.
```bash
python -c "import database; database.init_db()"
```
This bumps `schema_version` from 9 → 10 and adds the `labels_json` column.

### Step 3 — Backfill labels
```bash
python backfill_labels.py
```
The script processes only rows where `labels_json IS NULL`, so it's safe to interrupt and resume. With ~50k records and the gzip+JSON round-trip, expect this to take a few minutes.

Expected output:
```
INFO Backfilling labels for 50243 training records...
INFO Backfilled 50243 records.
Backfill result: {'total': 50243, 'updated': 50243, 'skipped_malformed': 0, 'errors': 0}
```

### Step 4 — Run a small training cycle first (validation)
Before kicking off the full training loop, do a small smoke pass to confirm everything works on real data:
```bash
NNRIOT_BATCH_SIZE=100 NNRIOT_EPOCHS=1 python continuous_trainer.py
```
(On Windows PowerShell, use `$env:NNRIOT_BATCH_SIZE=100; $env:NNRIOT_EPOCHS=1; python continuous_trainer.py`.)

This processes 100 untrained records for 1 epoch — should complete in ~1-2 minutes. Watch for:
- `model_v2.keras` is created
- Per-head logging shows non-NaN losses
- `winner_accuracy` is at least near 0.5 (random baseline)

### Step 5 — Full training cycle
Once the small run looks healthy, run the full trainer:
```bash
python continuous_trainer.py
```
This runs the indefinite training loop with `BATCH_SIZE=500` and re-checks for new records every 10 minutes. Stop with Ctrl-C when done.

### Step 6 — Verify the API
Start the web app:
```bash
python final_web_app.py
```
Then POST a match payload to `/api/predict` and confirm the response contains the new multi-output shape (`winner`, `winner_kills`, `kills`, `first.*`, `totals.*`, `both_teams.*`, `elder_dragon`).

### Rollback
If anything goes wrong:
1. Stop any running trainer / web app
2. Restore the backup: `copy training_data.backup-YYYYMMDD.db training_data.db`
3. The `model.keras` (single-output) checkpoint is still present and untouched; the trainer will fall back to `build_keras_model` only if you also revert the code.

