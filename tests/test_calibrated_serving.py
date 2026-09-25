"""Tests for calibrated serving, lock serialization, and mutation invalidation."""

import json
from pathlib import Path
import threading
from types import SimpleNamespace
import numpy as np
import pytest

from core.calibration import UncalibratedModelError
from core.feature_labels import LABEL_KEYS
from scripts.continuous_trainer import ContinuousTrainer


def _full_labels(**overrides) -> dict:
    base = {k: 0 for k in LABEL_KEYS}
    base.update(overrides)
    return base


def _sample_record(record_id: int = 1):
    return {
        "id": record_id,
        "match_id": f"M_{record_id}",
        "feature_json": {
            "team_a": [{"champion": "Ahri", "role": "MIDDLE"}],
            "team_b": [{"champion": "Zed", "role": "MIDDLE"}],
            "game_mode": "CLASSIC",
            "game_version": "14.10",
        },
        "labels_json": _full_labels(),
    }


def test_trainer_calibrates_winner_once_and_preserves_other_heads():
    raw = {
        "winner": np.array([[0.9, 0.1], [0.1, 0.9]]),
        "total_kills": np.array([[1.0], [2.0]]),
    }
    trainer = ContinuousTrainer.__new__(ContinuousTrainer)
    trainer._model_lock = threading.RLock()
    trainer.model = SimpleNamespace(
        is_calibrated=True,
        calibration_temperature=2.0,
        predict=lambda x, verbose=0: raw,
    )
    out = trainer.predict(np.zeros((2, 16)))
    np.testing.assert_allclose(out["winner"], [[0.75, 0.25], [0.25, 0.75]])
    np.testing.assert_array_equal(raw["winner"], [[0.9, 0.1], [0.1, 0.9]])
    np.testing.assert_array_equal(out["total_kills"], raw["total_kills"])
    trainer.model.is_calibrated = False
    with pytest.raises(UncalibratedModelError):
        trainer.predict(np.zeros((2, 16)))


def test_failed_mutation_sidecar_invalidation(tmp_path):
    checkpoint = tmp_path / "model_v2.keras"
    checkpoint.touch()
    sidecar = tmp_path / "model_v2.calibration.json"
    sidecar.write_text("{}", encoding="utf-8")

    class FakeRepo:
        def get_untrained_records(self, limit):
            return [_sample_record(1)]

        def get_random_trained_records(self, limit):
            return []

        def mark_as_trained(self, ids):
            pass

    class FakeModel:
        def __init__(self):
            self.is_calibrated = True
            self.calibration_temperature = 1.5
            self.calibration_metadata = {"fake": True}

        def train_on_batch(self, x, y, return_dict=True):
            # Assert sidecar is already absent and flag is False inside train_on_batch
            assert not sidecar.exists(), "Sidecar must be removed before train_on_batch"
            assert self.is_calibrated is False, "Model must be marked uncalibrated before train_on_batch"
            raise RuntimeError("Simulated failure during train_on_batch")

        def save(self, path):
            pass

    trainer = ContinuousTrainer.__new__(ContinuousTrainer)
    trainer._model_lock = threading.RLock()
    trainer.checkpoint_path = str(checkpoint)
    trainer.repo = FakeRepo()
    fake_model = FakeModel()
    trainer.model = fake_model

    with pytest.raises(RuntimeError, match="Simulated failure"):
        trainer.run_training_step()

    # Assert they stay invalid afterward
    assert not sidecar.exists()
    assert fake_model.is_calibrated is False
    assert fake_model.calibration_temperature is None
    assert fake_model.calibration_metadata is None


def test_sidecar_removal_permission_failure_prevents_training(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model_v2.keras"
    checkpoint.touch()
    sidecar = tmp_path / "model_v2.calibration.json"
    sidecar.write_text("{}", encoding="utf-8")

    class FakeRepo:
        def get_untrained_records(self, limit):
            return [_sample_record(1)]

        def get_random_trained_records(self, limit):
            return []

    class FakeModel:
        def __init__(self):
            self.is_calibrated = True
            self.calibration_temperature = 1.5
            self.calibration_metadata = {"fake": True}
            self.trained = False

        def train_on_batch(self, x, y, return_dict=True):
            self.trained = True
            return {"loss": 0.1}

    fake_model = FakeModel()
    trainer = ContinuousTrainer.__new__(ContinuousTrainer)
    trainer._model_lock = threading.RLock()
    trainer.checkpoint_path = str(checkpoint)
    trainer.repo = FakeRepo()
    trainer.model = fake_model

    # Simulate permission failure on sidecar deletion
    original_unlink = Path.unlink
    def _fail_unlink(self, *args, **kwargs):
        if self.name.endswith(".calibration.json"):
            raise PermissionError("Access denied: read-only filesystem")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _fail_unlink)

    with pytest.raises(PermissionError, match="Access denied"):
        trainer.run_training_step()

    assert fake_model.trained is False, "train_on_batch must never be called when sidecar unlink fails"


def test_empty_batch_retains_calibration_and_sidecar(tmp_path):
    checkpoint = tmp_path / "model_v2.keras"
    checkpoint.touch()
    sidecar = tmp_path / "model_v2.calibration.json"
    sidecar.write_text('{"temperature": 1.5}', encoding="utf-8")

    class FakeRepo:
        def get_untrained_records(self, limit):
            return []

    class FakeModel:
        def __init__(self):
            self.is_calibrated = True
            self.calibration_temperature = 1.5
            self.calibration_metadata = {"temperature": 1.5}

    fake_model = FakeModel()
    trainer = ContinuousTrainer.__new__(ContinuousTrainer)
    trainer._model_lock = threading.RLock()
    trainer.checkpoint_path = str(checkpoint)
    trainer.repo = FakeRepo()
    trainer.model = fake_model

    processed = trainer.run_training_step()
    assert processed == 0
    assert sidecar.exists(), "Sidecar must remain intact when batch is empty"
    assert fake_model.is_calibrated is True
    assert fake_model.calibration_temperature == 1.5


def test_trainer_lock_deterministic_with_event(tmp_path):
    checkpoint = tmp_path / "model_v2.keras"
    checkpoint.touch()

    in_training_event = threading.Event()
    release_training_event = threading.Event()

    class FakeRepo:
        def get_untrained_records(self, limit):
            return [_sample_record(1)]

        def get_random_trained_records(self, limit):
            return []

        def mark_as_trained(self, ids):
            pass

    class FakeModel:
        def __init__(self):
            self.is_calibrated = True
            self.calibration_temperature = 1.5
            self.calibration_metadata = {"fake": True}

        def train_on_batch(self, x, y, return_dict=True):
            in_training_event.set()
            # Block until test allows training to complete
            if not release_training_event.wait(timeout=3.0):
                raise TimeoutError("Timed out waiting for release_training_event")
            return {"loss": 0.5}

        def predict(self, x, verbose=0):
            return {"winner": np.array([[0.5, 0.5]])}

        def save(self, path):
            pass

    trainer = ContinuousTrainer.__new__(ContinuousTrainer)
    trainer._model_lock = threading.RLock()
    trainer.checkpoint_path = str(checkpoint)
    trainer.repo = FakeRepo()
    fake_model = FakeModel()
    trainer.model = fake_model

    predict_exception = []
    training_exception = []

    def _train_worker():
        try:
            trainer.run_training_step()
        except Exception as exc:
            training_exception.append(exc)

    def _predict_worker():
        try:
            trainer.predict(np.zeros((1, 16)))
        except Exception as exc:
            predict_exception.append(exc)

    t_train = threading.Thread(target=_train_worker, daemon=True)
    t_predict = threading.Thread(target=_predict_worker, daemon=True)

    try:
        t_train.start()
        # Wait until training owns the lock and is inside train_on_batch
        assert in_training_event.wait(timeout=3.0), "Training did not enter train_on_batch"

        # Start predict thread while training owns the lock
        t_predict.start()
        # Give predict thread a moment to try to acquire the lock (it will block)
        t_predict.join(timeout=0.1)
        assert t_predict.is_alive(), "Prediction thread should block while training owns the lock"

        # Now release training
        release_training_event.set()
        t_train.join(timeout=3.0)
        assert not t_train.is_alive(), "Training thread did not finish"

        # Now prediction thread should finish and raise UncalibratedModelError
        t_predict.join(timeout=3.0)
        assert not t_predict.is_alive(), "Prediction thread did not finish"

        assert len(predict_exception) == 1
        assert isinstance(predict_exception[0], UncalibratedModelError)
    finally:
        release_training_event.set()
        if t_train.is_alive():
            t_train.join(timeout=1.0)
        if t_predict.is_alive():
            t_predict.join(timeout=1.0)


def test_endpoint_feature_parity_and_calibration(monkeypatch):
    from app import app as flask_app
    from app import core as app_core
    from core import json_utils

    captured_vectors = []
    original_json_to_vector = json_utils.json_to_vector

    def _spy_json_to_vector(features_list, dim):
        captured_vectors.append(features_list)
        return original_json_to_vector(features_list, dim)

    monkeypatch.setattr(json_utils, "json_to_vector", _spy_json_to_vector)

    raw_preds = {
        "winner": np.array([[0.8, 0.2]], dtype=np.float32),
        "team_b_kill_lead": np.array([[0.5, 0.5]], dtype=np.float32),
        "first_blood": np.array([[0.33, 0.33, 0.34]], dtype=np.float32),
        "first_baron": np.array([[0.33, 0.33, 0.34]], dtype=np.float32),
        "first_inhibitor": np.array([[0.33, 0.33, 0.34]], dtype=np.float32),
        "first_tower": np.array([[0.33, 0.33, 0.34]], dtype=np.float32),
        "kills_odd": np.array([[0.5]], dtype=np.float32),
        "both_baron": np.array([[0.1]], dtype=np.float32),
        "both_inhibitor": np.array([[0.1]], dtype=np.float32),
        "both_dragon": np.array([[0.1]], dtype=np.float32),
        "elder_dragon": np.array([[0.1]], dtype=np.float32),
        "kill_handicap": np.array([[0.0]], dtype=np.float32),
        "total_kills": np.array([[30.0]], dtype=np.float32),
        "team_a_kills": np.array([[15.0]], dtype=np.float32),
        "team_b_kills": np.array([[15.0]], dtype=np.float32),
        "total_barons": np.array([[1.0]], dtype=np.float32),
        "total_dragons": np.array([[2.0]], dtype=np.float32),
        "total_towers": np.array([[10.0]], dtype=np.float32),
        "first_to_5_kills": np.array([[0.33, 0.33, 0.34]], dtype=np.float32),
        "first_to_10_kills": np.array([[0.33, 0.33, 0.34]], dtype=np.float32),
        "first_to_15_kills": np.array([[0.33, 0.33, 0.34]], dtype=np.float32),
        "first_to_20_kills": np.array([[0.33, 0.33, 0.34]], dtype=np.float32),
    }

    class FakeModel:
        is_calibrated = True
        calibration_temperature = 2.0
        calibration_metadata = {"temperature": 2.0}

        def predict(self, x, verbose=0):
            return raw_preds

    trainer = ContinuousTrainer.__new__(ContinuousTrainer)
    trainer._model_lock = threading.RLock()
    trainer.model = FakeModel()

    monkeypatch.setattr(app_core, "global_trainer", trainer)
    monkeypatch.setattr(app_core, "tf_available", True)

    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    # 1. Test legacy match endpoint /api/predict with extra post-match stats
    legacy_payload = {
        "participants": [
            {"teamId": 100, "championName": "Ahri", "teamPosition": "MIDDLE", "kills": 10, "deaths": 2, "goldEarned": 15000},
            {"teamId": 200, "championName": "Zed", "teamPosition": "MIDDLE", "kills": 2, "deaths": 10, "goldEarned": 8000},
        ],
        "teams": [{"teamId": 100, "win": True}, {"teamId": 200, "win": False}],
    }
    resp1 = client.post("/api/predict", json=legacy_payload)
    assert resp1.status_code == 200
    data1 = resp1.get_json()

    # 2. Test custom predictor endpoint /api/predict/custom with career stats
    custom_payload = {
        "blue_team": [{"champion_name": "Ahri", "role": "MIDDLE", "career_win_rate": 0.8, "career_matches": 100, "avg_kda": 3.5}] * 5,
        "red_team": [{"champion_name": "Zed", "role": "MIDDLE", "career_win_rate": 0.2, "career_matches": 50, "avg_kda": 1.0}] * 5,
    }
    resp2 = client.post("/api/predict/custom", json=custom_payload)
    assert resp2.status_code == 200
    data2 = resp2.get_json()

    # Check feature parity: verify that both vectorizations received ONLY clean pre-match draft features
    assert len(captured_vectors) == 2
    for call_features in captured_vectors:
        feat = call_features[0]
        # Only context and draft keys allowed
        assert set(feat.keys()).issubset({"game_mode", "game_version", "season_id", "region", "team_a", "team_b"})
        for p in feat["team_a"] + feat["team_b"]:
            assert set(p.keys()) == {"champion", "role"}

    # Both routes should receive the same calibrated winner probabilities:
    # raw is [0.8, 0.2], with T=2.0 -> scaled P(B) is ~0.332, so [0.668, 0.332]
    # Check that both endpoints match within 0.001
    p_a1, p_b1 = data1["winner"]["team_a"], data1["winner"]["team_b"]
    p_a2, p_b2 = data2["nn_results"]["blue_win_probability"], data2["nn_results"]["red_win_probability"]
    assert abs(p_a1 - p_a2) <= 0.001
    assert abs(p_b1 - p_b2) <= 0.001
    assert abs(p_a1 - 0.668) <= 0.002

    # Check uncalibrated model yields 503 on custom route
    trainer.model.is_calibrated = False
    resp_uncal = client.post("/api/predict/custom", json=custom_payload)
    assert resp_uncal.status_code == 503
    assert resp_uncal.get_json().get("uncalibrated") is True
