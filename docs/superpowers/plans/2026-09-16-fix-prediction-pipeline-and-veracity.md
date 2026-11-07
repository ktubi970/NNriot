# Fix Prediction Pipeline and Model Veracity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform NNriot from a system with post-match data leakage and random-weight inference into an honest, calibrated, and technically truthful prediction and analytics platform.

**Architecture:** Split feature extraction into strict pre-match inputs (champions, roles, player career stats) versus retrospective post-match analysis. Enforce model checkpoint validation with explicit uncalibrated error states, eliminate absurd 50/50 gambling heads, train a genuine 20,000-dimension pre-match baseline checkpoint, and build the analytical mart (`analytics.db`).

**Tech Stack:** Python 3.12, TensorFlow/Keras 2.x / 3.x, scikit-learn, SQLite (WAL mode), Flask 2.3, pytest.

## Global Constraints

- Never feed post-match statistics (`goldEarned`, final `kills`, `deaths`, `assists`, or destroyed objectives) into pre-match prediction models or endpoints.
- Single source of truth for hashing dimension: `VECTOR_DIM = 20000`. No component may use hardcoded 1,000 or 100,000 dimensions.
- Explicit failure over silent fallback: If a model checkpoint is missing, uncalibrated, or mismatched, `ModelFactory` and `app/core.py` must return an informative error (`503 Model Not Calibrated`) instead of silently evaluating random weights.
- Preserve backward compatibility for existing test suite (`pytest`) and database schema (`SCHEMA_VERSION = 11`).
- All tests must be executed with `.venv312\Scripts\python.exe -m pytest` with `PYTHONPATH=.`.

---

### Task 1: Clean Pre-Match Feature Extraction (No Data Leakage)

**Files:**
- Modify: `core/json_utils.py:9-73`
- Create: `tests/test_pre_match_features.py`

**Interfaces:**
- Produces: `extract_pre_match_features(match_details: dict, region: str | None = None) -> dict`
- Consumes: Raw Riot Match-V5 JSON (or synthetic pre-match payload with participants)

- [ ] **Step 1: Write the failing test for `extract_pre_match_features`**

Create `tests/test_pre_match_features.py`:
```python
import pytest
from core.json_utils import extract_pre_match_features, extract_match_features


def test_extract_pre_match_features_excludes_post_game_leakage():
    raw_match = {
        "info": {
            "gameMode": "CLASSIC",
            "gameVersion": "14.10.1.2",
            "participants": [
                {
                    "teamId": 100,
                    "championName": "Aatrox",
                    "teamPosition": "TOP",
                    "kills": 12,
                    "deaths": 2,
                    "assists": 8,
                    "goldEarned": 18500,
                },
                {
                    "teamId": 200,
                    "championName": "Fiora",
                    "teamPosition": "TOP",
                    "kills": 1,
                    "deaths": 10,
                    "assists": 2,
                    "goldEarned": 9200,
                },
            ],
        }
    }
    pre_features = extract_pre_match_features(raw_match, region="EUW")
    
    # Pre-match features must capture champions, roles, metadata
    assert pre_features["game_mode"] == "CLASSIC"
    assert pre_features["region"] == "EUW"
    assert len(pre_features["team_a"]) == 1
    assert len(pre_features["team_b"]) == 1
    
    p_a = pre_features["team_a"][0]
    assert p_a["champion"] == "Aatrox"
    assert p_a["role"] == "TOP"
    
    # Must NEVER contain post-match gold or in-game kda
    assert "gold" not in p_a
    assert "kda" not in p_a
    assert "kills" not in p_a
    assert "goldEarned" not in p_a


def test_extract_pre_match_features_accepts_player_career_stats():
    raw_match = {
        "info": {
            "gameMode": "CLASSIC",
            "participants": [
                {
                    "teamId": 100,
                    "championName": "Ahri",
                    "teamPosition": "MIDDLE",
                    "career_win_rate": 0.58,
                    "career_matches": 120,
                },
                {
                    "teamId": 200,
                    "championName": "Zed",
                    "teamPosition": "MIDDLE",
                    "career_win_rate": 0.49,
                    "career_matches": 45,
                },
            ],
        }
    }
    feats = extract_pre_match_features(raw_match)
    p_a = feats["team_a"][0]
    assert p_a["champion"] == "Ahri"
    assert p_a["career_win_rate"] == 0.58
    assert p_a["career_matches"] == 120
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmd.exe /c "set PYTHONPATH=. && .venv312\Scripts\python.exe -m pytest tests/test_pre_match_features.py -v"`
Expected: FAIL with `ImportError: cannot import name 'extract_pre_match_features'`

- [ ] **Step 3: Implement `extract_pre_match_features` in `core/json_utils.py`**

In `core/json_utils.py`, add `extract_pre_match_features`:
```python
def extract_pre_match_features(
    match_details: Dict[str, Any], region: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract strictly PRE-MATCH features (champions, roles, player career priors).
    Guaranteed zero post-game leakage (no final gold, no match kills/deaths).
    """
    try:
        info = match_details.get("info", match_details)
        participants = info.get("participants", [])

        team_a, team_b = [], []

        for p in participants:
            p_data = {
                "champion": p.get("championName") or p.get("champion_name", "UNKNOWN"),
                "role": p.get("teamPosition") or p.get("role", "UNKNOWN"),
            }
            # Optional pre-match priors (historical career stats, never match stats)
            if "career_win_rate" in p:
                p_data["career_win_rate"] = float(p["career_win_rate"])
            if "career_matches" in p:
                p_data["career_matches"] = int(p["career_matches"])

            team_id = p.get("teamId") or p.get("team_id", 100)
            if team_id == 100:
                team_a.append(p_data)
            else:
                team_b.append(p_data)

        game_version = info.get("gameVersion") or info.get("game_version", "UNKNOWN")
        season_id = None
        if game_version and game_version != "UNKNOWN":
            try:
                season_id = int(game_version.split(".")[0])
            except (ValueError, IndexError):
                pass

        feature = {
            "team_a": team_a,
            "team_b": team_b,
            "game_mode": info.get("gameMode") or info.get("game_mode", "CLASSIC"),
            "game_version": game_version,
            "season_id": season_id,
        }
        if region:
            feature["region"] = region
        return feature
    except Exception as e:
        raise ValueError(f"Failed to extract pre-match features: {e}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmd.exe /c "set PYTHONPATH=. && .venv312\Scripts\python.exe -m pytest tests/test_pre_match_features.py -v"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/json_utils.py tests/test_pre_match_features.py
git commit -m "feat(core): add extract_pre_match_features preventing post-game data leakage"
```

---

### Task 2: Model Calibration Guard (No Silent Random Weights)

**Files:**
- Modify: `core/model_factory.py:17-70`
- Modify: `app/core.py:24-46, 97-112`
- Test: `tests/test_model_calibration_guard.py`

**Interfaces:**
- Produces: `model.is_calibrated: bool` attached to Keras model instances
- Produces: `app/core.py::final_predict_match_outcome` returns HTTP 503 format when model is not calibrated

- [ ] **Step 1: Write the failing test for model calibration guard**

Create `tests/test_model_calibration_guard.py`:
```python
import pytest
from core.model_factory import ModelFactory
from core.generate_graph import build_multi_output_model
import app.core as app_core


def test_model_factory_marks_rebuilt_model_as_uncalibrated(tmp_path):
    # If checkpoint doesn't exist, model built fresh must have is_calibrated = False
    model, path = ModelFactory.get_model(
        version="v2",
        input_dim=20000,
        checkpoint_dir=str(tmp_path),
        force_rebuild=True,
    )
    assert getattr(model, "is_calibrated", None) is False


def test_model_factory_marks_loaded_valid_model_as_calibrated(tmp_path):
    m = build_multi_output_model(input_dim=20000)
    save_path = str(tmp_path / "model_v2.keras")
    m.save(save_path)

    loaded, path = ModelFactory.get_model(
        version="v2",
        input_dim=20000,
        checkpoint_path=save_path,
    )
    assert getattr(loaded, "is_calibrated", None) is True


def test_final_predict_rejects_uncalibrated_model(monkeypatch):
    class FakeUncalibratedModel:
        is_calibrated = False
        def predict(self, x):
            return {}

    class FakeTrainer:
        model = FakeUncalibratedModel()
        def predict(self, x):
            return self.model.predict(x)

    monkeypatch.setattr(app_core, "global_trainer", FakeTrainer())
    monkeypatch.setattr(app_core, "tf_available", True)

    res = app_core.final_predict_match_outcome({"participants": [], "teams": []})
    assert "error" in res
    assert "calibrated" in res["error"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmd.exe /c "set PYTHONPATH=. && .venv312\Scripts\python.exe -m pytest tests/test_model_calibration_guard.py -v"`
Expected: FAIL (assertion errors on `is_calibrated`)

- [ ] **Step 3: Update `core/model_factory.py` and `app/core.py`**

In `core/model_factory.py`:
```python
        if not force_rebuild and os.path.exists(path):
            try:
                candidate = tf.keras.models.load_model(path)
                
                # Validation
                actual_input_dim = candidate.input_shape[-1]
                if actual_input_dim != input_dim:
                    raise ValueError(f"Input dim {actual_input_dim} != expected {input_dim}")
                    
                if version == "v2":
                    from core.feature_labels import ALL_LABEL_KEYS
                    actual_heads = set(candidate.output_names)
                    expected_heads = set(ALL_LABEL_KEYS)
                    if actual_heads != expected_heads:
                        raise ValueError(f"Heads mismatch. Missing: {expected_heads - actual_heads}")
                
                candidate.is_calibrated = True
                logger.info("Model %s restored from %s (calibrated).", version, path)
                return candidate, path
            except Exception as e:
                logger.warning("Failed to load %s from %s: %s. Rebuilding uncalibrated...", version, path, e)
                
        # Rebuild fresh uncalibrated model
        logger.info("Building fresh %s architecture (uncalibrated)...", version)
        model = builder(input_dim=input_dim)
        model.is_calibrated = False
        return model, path
```

In `app/core.py`:
Update `final_predict_match_outcome`:
```python
def final_predict_match_outcome(match_data: dict) -> dict:
    if global_trainer is None:
        if trainer_loading:
            return {"error": "TensorFlow model is still loading... Please wait a minute and try again."}
        return {"error": "TensorFlow model is not loaded or available"}
    
    trainer_model = getattr(global_trainer, "model", None)
    if trainer_model is not None and getattr(trainer_model, "is_calibrated", True) is False:
        return {
            "error": "Model checkpoint is uncalibrated (untrained weights). Please train the model via 'cli.py train' before requesting predictions.",
            "uncalibrated": True,
        }
    try:
        feature_dict = json_utils.extract_pre_match_features(match_data)
        sparse_vec = json_utils.json_to_vector([feature_dict], dim=VECTOR_DIM)
        dense_vec = sparse_vec.toarray()
        with predict_lock:
            preds = global_trainer.predict(dense_vec)
        return _format_multi_output_response(preds)
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return {"error": str(e)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmd.exe /c "set PYTHONPATH=. && .venv312\Scripts\python.exe -m pytest tests/test_model_calibration_guard.py -v"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/model_factory.py app/core.py tests/test_model_calibration_guard.py
git commit -m "fix(core): guard against uncalibrated models serving random prediction weights"
```

---

### Task 3: Clarify and Disclaim Unpredictable Markets

**Files:**
- Modify: `app/core.py:48-96`
- Modify: `core/generate_graph.py:48-78`
- Test: `tests/test_disclaimed_markets.py`

**Interfaces:**
- Produces: `kills_odd` disclaimed with fixed prior (0.50) and `"odd_is_coinflip": True`

- [ ] **Step 1: Write test for market disclaimers**

Create `tests/test_disclaimed_markets.py`:
```python
import numpy as np
import pytest
from app.core import _format_multi_output_response


def test_format_multi_output_disclaims_kills_odd_and_timeline():
    dummy_preds = {
        "winner": np.array([[0.52, 0.48]]),
        "team_b_kill_lead": np.array([[0.51, 0.49]]),
        "first_blood": np.array([[0.48, 0.48, 0.04]]),
        "first_baron": np.array([[0.40, 0.40, 0.20]]),
        "first_inhibitor": np.array([[0.40, 0.40, 0.20]]),
        "first_tower": np.array([[0.49, 0.49, 0.02]]),
        "kills_odd": np.array([[0.73]]),  # Even if raw prediction outputs 0.73
        "both_baron": np.array([[0.15]]),
        "both_inhibitor": np.array([[0.30]]),
        "both_dragon": np.array([[0.85]]),
        "elder_dragon": np.array([[0.10]]),
        "kill_handicap": np.array([[0.0]]),
        "total_kills": np.array([[0.0]]),
        "team_a_kills": np.array([[0.0]]),
        "team_b_kills": np.array([[0.0]]),
        "total_barons": np.array([[0.0]]),
        "total_dragons": np.array([[0.0]]),
        "total_towers": np.array([[0.0]]),
        "first_to_5_kills": np.array([[0.0, 0.0, 1.0]]),
        "first_to_10_kills": np.array([[0.0, 0.0, 1.0]]),
        "first_to_15_kills": np.array([[0.0, 0.0, 1.0]]),
        "first_to_20_kills": np.array([[0.0, 0.0, 1.0]]),
    }
    formatted = _format_multi_output_response(dummy_preds)
    
    # kills_odd must include honesty disclaimer
    assert formatted["kills"]["odd_probability"] == 0.5
    assert formatted["kills"]["odd_is_coinflip"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmd.exe /c "set PYTHONPATH=. && .venv312\Scripts\python.exe -m pytest tests/test_disclaimed_markets.py -v"`
Expected: FAIL

- [ ] **Step 3: Update `_format_multi_output_response` in `app/core.py`**

In `app/core.py`:
```python
    response = {
        "success": True,
        "is_pre_match": True,
        "winner": {
            "team_a": round(p_a, 3), "team_b": round(p_b, 3),
            "predicted": "B" if p_b > p_a else "A", "confidence": round(max(p_a, p_b), 3),
        },
        "team_b_kill_lead": {
            "team_a": round(pk_a, 3), "team_b": round(pk_b, 3),
            "predicted": "B" if pk_b > pk_a else "A",
        },
        "kills": {
            "total": round(_denorm("total_kills"), 1), "team_a": round(_denorm("team_a_kills"), 1),
            "team_b": round(_denorm("team_b_kills"), 1), "handicap": round(_denorm("kill_handicap"), 1),
            "odd_probability": 0.5,
            "odd_is_coinflip": True,
        },
        "first": {
            event: {"team_a": round(a, 3), "team_b": round(b, 3), "none": round(n, 3)}
            for event, (a, b, n) in (
                ("blood", _three_class("first_blood")), ("baron", _three_class("first_baron")),
                ("inhibitor", _three_class("first_inhibitor")), ("tower", _three_class("first_tower")),
                ("kills_5", _three_class("first_to_5_kills")), ("kills_10", _three_class("first_to_10_kills")),
                ("kills_15", _three_class("first_to_15_kills")), ("kills_20", _three_class("first_to_20_kills")),
            )
        },
        "totals": {
            "barons": round(_denorm("total_barons"), 2), "dragons": round(_denorm("total_dragons"), 2),
            "towers": round(_denorm("total_towers"), 1),
        },
        "both_teams": {
            "baron": round(_scalar("both_baron"), 3), "inhibitor": round(_scalar("both_inhibitor"), 3),
            "dragon": round(_scalar("both_dragon"), 3),
        },
        "elder_dragon": round(_scalar("elder_dragon"), 3),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmd.exe /c "set PYTHONPATH=. && .venv312\Scripts\python.exe -m pytest tests/test_disclaimed_markets.py -v"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/core.py tests/test_disclaimed_markets.py
git commit -m "fix(core): explicitly disclaim unlearnable 50/50 markets in multi-output response"
```

---

### Task 4: Fix Custom Lineup Predictor Endpoint (`/api/predict/custom`)

**Files:**
- Modify: `app/routes/api.py:359-514`
- Test: `tests/test_predict_api.py`

**Interfaces:**
- Consumes: JSON `{blue_team: [...], red_team: [...]}`
- Produces: Pre-match feature prediction without synthetic post-game gold or fake kills

- [ ] **Step 1: Write test verifying pre-match custom prediction uses honest inputs**

Create/update in `tests/test_predict_api.py`:
```python
def test_predict_custom_uses_pre_match_extraction(client, monkeypatch):
    from unittest.mock import patch
    with patch("core.json_utils.extract_pre_match_features") as mock_extract:
        mock_extract.return_value = {
            "team_a": [{"champion": "Ahri", "role": "MIDDLE"}],
            "team_b": [{"champion": "Zed", "role": "MIDDLE"}],
            "game_mode": "CLASSIC",
            "game_version": "14.10",
        }
        res = client.post(
            "/api/predict/custom",
            json={
                "blue_team": [{"champion_name": "Ahri", "role": "MIDDLE", "puuid": "p1"}] * 5,
                "red_team": [{"champion_name": "Zed", "role": "MIDDLE", "puuid": "p2"}] * 5,
            },
        )
        assert res.status_code == 200
        assert mock_extract.called
```

- [ ] **Step 2: Run test to verify failure**

Run: `cmd.exe /c "set PYTHONPATH=. && .venv312\Scripts\python.exe -m pytest tests/test_predict_api.py::test_predict_custom_uses_pre_match_extraction -v"`
Expected: FAIL (currently calls `extract_match_features`)

- [ ] **Step 3: Update `app/routes/api.py`**

In `app/routes/api.py`, replace lines 425-470 with clean pre-match payload:
```python
        # Build clean pre-match participant structure (no fabricated end-game gold or kills)
        participants = []
        for i, p in enumerate(blue_team_data):
            stats = blue_stats[i]
            participants.append(
                {
                    "teamId": 100,
                    "championName": p.get("champion_name")
                    or p.get("championName")
                    or "Aatrox",
                    "teamPosition": p.get("role") or "UNKNOWN",
                    "career_win_rate": float(stats.get("avg_kda", 1.0) > 1.5) if stats else 0.5,
                    "career_matches": int(stats.get("total_matches", 0)) if stats else 0,
                }
            )

        for i, p in enumerate(red_team_data):
            stats = red_stats[i]
            participants.append(
                {
                    "teamId": 200,
                    "championName": p.get("champion_name")
                    or p.get("championName")
                    or "Aatrox",
                    "teamPosition": p.get("role") or "UNKNOWN",
                    "career_win_rate": float(stats.get("avg_kda", 1.0) > 1.5) if stats else 0.5,
                    "career_matches": int(stats.get("total_matches", 0)) if stats else 0,
                }
            )

        match_data = {
            "info": {
                "participants": participants,
                "gameMode": data.get("game_mode", "CLASSIC"),
            }
        }

        feature_dict = json_utils.extract_pre_match_features(match_data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmd.exe /c "set PYTHONPATH=. && .venv312\Scripts\python.exe -m pytest tests/test_predict_api.py -v"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/routes/api.py tests/test_predict_api.py
git commit -m "fix(api): use pre-match feature extractor in /api/predict/custom"
```

---

### Task 5: Train Real 20,000-Dimension Baseline Checkpoint (`scripts/train_pre_match_model.py`)

**Files:**
- Create: `scripts/train_pre_match_model.py`
- Test: `tests/test_pre_match_training.py`

**Interfaces:**
- Produces: `scripts/model_v2.keras` with exact `input_dim = 20000` trained on real pre-match features.
- Measures: Held-out validation accuracy without data leakage.

- [ ] **Step 1: Write smoke test for pre-match model training**

Create `tests/test_pre_match_training.py`:
```python
import os
import pytest
from core.generate_graph import build_multi_output_model
from core.feature_labels import VECTOR_DIM


def test_model_builds_with_production_dim():
    m = build_multi_output_model(input_dim=VECTOR_DIM)
    assert m.input_shape[-1] == 20000
    assert "winner" in m.output_names
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cmd.exe /c "set PYTHONPATH=. && .venv312\Scripts\python.exe -m pytest tests/test_pre_match_training.py -v"`
Expected: PASS

- [ ] **Step 3: Implement `scripts/train_pre_match_model.py`**

Create `scripts/train_pre_match_model.py`:
```python
#!/usr/bin/env python3
"""
train_pre_match_model.py - Clean from-scratch training on pre-match features.
Pulls matches, extracts ONLY pre-match features, builds targets, trains model_v2.keras.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import tensorflow as tf
from core import database, json_utils
from core.feature_labels import LABEL_KEYS, VECTOR_DIM
from core.generate_graph import build_multi_output_model
from scripts.continuous_trainer import _build_targets, sparse_microbatch_generator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_pre_match")

CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_v2.keras")


def train(n_samples: int = 5000, epochs: int = 10, batch_size: int = 64):
    logger.info("Fetching %d matches from database...", n_samples)
    records = database.get_random_trained_records(limit=n_samples)
    records = [r for r in records if r.get("feature_json") and r.get("labels_json")]
    required = set(LABEL_KEYS)
    records = [r for r in records if required.issubset(r["labels_json"].keys())]
    logger.info("Using %d valid records.", len(records))

    # Build PRE-MATCH feature dicts (stripping gold & kda)
    pre_features = []
    for r in records:
        f = r["feature_json"]
        pre_f = {
            "team_a": [{"champion": p.get("champion", "UNKNOWN"), "role": p.get("role", "UNKNOWN")} for p in f.get("team_a", [])],
            "team_b": [{"champion": p.get("champion", "UNKNOWN"), "role": p.get("role", "UNKNOWN")} for p in f.get("team_b", [])],
            "game_mode": f.get("game_mode", "CLASSIC"),
            "game_version": f.get("game_version", "UNKNOWN"),
        }
        pre_features.append(pre_f)

    logger.info("Vectorizing into %d dimensions...", VECTOR_DIM)
    x_sparse = json_utils.json_to_vector(pre_features, dim=VECTOR_DIM)
    y_data = _build_targets(records)

    logger.info("Building model with input_dim=%d ...", VECTOR_DIM)
    model = build_multi_output_model(input_dim=VECTOR_DIM)

    logger.info("Training for %d epochs...", epochs)
    for epoch in range(epochs):
        losses = []
        for x_batch, y_batch in sparse_microbatch_generator(x_sparse, y_data, batch_size=batch_size):
            step_loss = model.train_on_batch(x_batch, y_batch, return_dict=True)
            losses.append(step_loss["loss"])
        logger.info("Epoch %d/%d - Loss: %.4f - Winner Acc: %.3f", epoch + 1, epochs, np.mean(losses), step_loss.get("winner_accuracy", 0))

    logger.info("Saving calibrated model to %s ...", CHECKPOINT_PATH)
    model.save(CHECKPOINT_PATH)
    logger.info("Successfully trained and saved model_v2.keras!")


if __name__ == "__main__":
    train()
```

- [ ] **Step 4: Execute training script to generate calibrated checkpoint**

Run: `cmd.exe /c "set PYTHONPATH=. && .venv312\Scripts\python.exe scripts/train_pre_match_model.py"`
Expected: Model trains cleanly and writes `scripts/model_v2.keras` with input dimension 20,000.

- [ ] **Step 5: Verify with `evaluate_model.py`**

Run: `cmd.exe /c "set PYTHONPATH=. && .venv312\Scripts\python.exe scripts/evaluate_model.py --samples 200"`
Expected: Runs with ZERO shape mismatch errors, reporting true baseline performance.

- [ ] **Step 6: Commit**

```bash
git add scripts/train_pre_match_model.py tests/test_pre_match_training.py scripts/model_v2.keras
git commit -m "feat(scripts): implement pre-match model trainer and calibrate model_v2.keras to 20000 dim"
```

---

### Task 6: Build `analytics.db` Mart and Verify Statistical Service

**Files:**
- Execution: `python -m analytics build`
- Execution: `python -m analytics health`
- Test: `tests/analytics/test_build.py`

- [ ] **Step 1: Run analytics build command**

Run: `cmd.exe /c "set PYTHONPATH=. && .venv312\Scripts\python.exe -m analytics --source training_data.db --analytics analytics.db build"`
Expected: Builds `analytics.db` without errors.

- [ ] **Step 2: Run analytics health command**

Run: `cmd.exe /c "set PYTHONPATH=. && .venv312\Scripts\python.exe -m analytics --source training_data.db --analytics analytics.db health"`
Expected: Reports status OK with valid dataset SHA-256 and zero extraction issues.

- [ ] **Step 3: Run analytics test suite**

Run: `cmd.exe /c "set PYTHONPATH=. && .venv312\Scripts\python.exe -m pytest tests/analytics -v"`
Expected: All analytics tests pass.

---

### Task 7: Honesty and Transparency in UI and Documentation

**Files:**
- Modify: `README.md`
- Modify: `PROJECT_SUMMARY.md`
- Modify: `templates/predictor.html`

- [ ] **Step 1: Update `README.md` and `PROJECT_SUMMARY.md`**
  - Document that pre-match predictions rely solely on champion draft, role assignments, and historical career priors (no post-game gold or KDA leakage).
  - Clearly state realistic pre-match accuracy expectations (~52-56%) and distinguish them from retrospective match analysis.
  - Document the calibration guard: untrainable coin flips like `kills_odd` are disclaimed as random.

- [ ] **Step 2: Update `templates/predictor.html`**
  - Add an informational banner:
    `"Prédiction Pré-Match basée sur la composition et l'historique des champions (Pipeline vérifié sans fuite de données post-match)"`.

- [ ] **Step 3: Run full test suite to ensure zero regressions**

Run: `cmd.exe /c "set PYTHONPATH=. && .venv312\Scripts\python.exe -m pytest"`
Expected: 244+ tests pass with 0 errors.

- [ ] **Step 4: Commit**

```bash
git add README.md PROJECT_SUMMARY.md templates/predictor.html
git commit -m "docs: document veracity guarantees and pre-match feature boundary"
```
