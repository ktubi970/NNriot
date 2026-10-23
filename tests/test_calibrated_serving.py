"""Tests for calibrated serving, lock serialization, and mutation invalidation."""

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


def test_endpoint_feature_parity_and_calibration(monkeypatch, tmp_path):
    from copy import deepcopy
    from app import app as flask_app
    import core.prediction_service as serving
    from tests.serving_fixture import custom_match, full_match, install_serving_fixture

    captured_features = []
    captured_inputs = []
    original_extract = serving.extract_aligned_inputs

    def capture_inputs(match, *args, **kwargs):
        captured_features.append(deepcopy(match))
        tensors = original_extract(match, *args, **kwargs)
        captured_inputs.append({key: value.copy() for key, value in tensors.items()})
        return tensors

    monkeypatch.setattr(serving, 'extract_aligned_inputs', capture_inputs)
    raw = {
        'winner': np.array([[.8, .2]], dtype=np.float32),
        'total_kills': np.array([[30.0]], dtype=np.float32),
    }
    install_serving_fixture(monkeypatch, raw, tmp_path, temperature=2.0)
    flask_app.config['TESTING'] = True
    client = flask_app.test_client()

    legacy_payload = full_match()
    for participant in legacy_payload['participants']:
        participant.update(deaths=4444, assists=5555, win=True)
    legacy_payload['teams'][0]['win'] = True
    legacy_payload['teams'][1]['win'] = False
    legacy_payload['winner'] = 100

    custom_payload = custom_match()
    for key in ('blue_team', 'red_team'):
        for participant in custom_payload[key]:
            participant.update(
                career_win_rate=.9, career_matches=100, avg_kda=3.5,
                kills=7777, goldEarned=88888, win=True,
            )

    legacy_response = client.post('/api/predict', json=legacy_payload)
    custom_response = client.post('/api/predict/custom', json=custom_payload)
    assert legacy_response.status_code == 200
    assert custom_response.status_code == 200
    legacy = legacy_response.get_json()
    custom = custom_response.get_json()

    # Both routes reach the real aligned encoder with only pre-match facts.
    assert len(captured_features) == len(captured_inputs) == 2
    assert captured_features[0] == captured_features[1]
    for features in captured_features:
        assert set(features) <= {
            'game_mode', 'game_version', 'season_id', 'region',
            'team_a', 'team_b', 'prediction_time_ms',
        }
        for key in ('team_a', 'team_b'):
            assert len(features[key]) == 5
            assert {p['role'] for p in features[key]} == {
                'TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY',
            }
            for player in features[key]:
                assert set(player) == {'champion', 'role', 'puuid'}
    assert set(captured_inputs[0]) == set(captured_inputs[1])
    for key in captured_inputs[0]:
        np.testing.assert_array_equal(captured_inputs[0][key], captured_inputs[1][key])
    assert captured_inputs[0]['team_a'].shape == (1, 5, 14)
    assert captured_inputs[0]['team_b'].shape == (1, 5, 14)

    assert legacy['probabilities_raw'] == custom['probabilities_raw']
    assert legacy['probabilities_raw']['team_b'] == pytest.approx(.2)
    assert legacy['probabilities_calibrated'] == custom['probabilities_calibrated']
    # T=2 scales the raw odds 1:4 to 1:2, so calibrated P(B) is 1/3.
    assert legacy['probabilities_calibrated']['team_b'] == pytest.approx(1 / 3)
    assert legacy['winner']['team_a'] == custom['nn_results']['blue_win_probability'] == .6667
    assert legacy['winner']['team_b'] == custom['nn_results']['red_win_probability'] == .3333
    assert legacy['heads_raw']['total_kills'] == [30.0]
    assert legacy['calibration'] == custom['calibration'] == {
        'status': 'artifact_verified', 'temperature': 2.0,
    }
    assert legacy['model']['serving_validation']['status'] == 'validated'
    assert legacy['selective_decision'] == custom['selective_decision']
    assert legacy['selective_decision']['abstained'] is False
