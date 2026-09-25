import json
from pathlib import Path
import tensorflow as tf
import pytest
from core.model_factory import ModelFactory
from core.feature_labels import LABEL_KEYS, TIMELINE_LABEL_KEYS
from core.experiment import build_manifest
from core.calibration import save_calibration_artifact
import app.core as app_core


def _tiny_model(input_dim: int = 16):
    inputs = tf.keras.Input(shape=(input_dim,), sparse=False)
    outputs = {}
    for k in LABEL_KEYS + TIMELINE_LABEL_KEYS:
        if k in ("winner", "team_b_kill_lead"):
            outputs[k] = tf.keras.layers.Dense(2, activation="softmax", name=k)(inputs)
        elif k in (
            "first_blood", "first_baron", "first_inhibitor", "first_tower",
            "first_to_5_kills", "first_to_10_kills", "first_to_15_kills", "first_to_20_kills"
        ):
            outputs[k] = tf.keras.layers.Dense(3, activation="softmax", name=k)(inputs)
        elif k in ("kills_odd", "both_baron", "both_inhibitor", "both_dragon", "elder_dragon"):
            outputs[k] = tf.keras.layers.Dense(1, activation="sigmoid", name=k)(inputs)
        else:
            outputs[k] = tf.keras.layers.Dense(1, activation="linear", name=k)(inputs)
    m = tf.keras.Model(inputs=inputs, outputs=outputs)
    m.compile(optimizer="adam", loss="mse")
    return m


def test_model_factory_marks_rebuilt_model_as_uncalibrated(tmp_path, monkeypatch):
    import core.model_factory as mf
    monkeypatch.setattr(mf, "build_multi_output_model", lambda input_dim: _tiny_model(input_dim))
    model, path = ModelFactory.get_model(
        version="v2",
        input_dim=16,
        checkpoint_dir=str(tmp_path),
        force_rebuild=True,
    )
    assert getattr(model, "is_calibrated", None) is False
    assert getattr(model, "calibration_temperature", None) is None
    assert getattr(model, "calibration_metadata", None) is None


def test_model_factory_valid_model_without_sidecar_stays_uncalibrated(tmp_path, monkeypatch):
    dim = 16
    m = _tiny_model(input_dim=dim)
    save_path = str(tmp_path / "model_v2.keras")
    m.save(save_path)

    import core.model_factory as mf
    def _fail_rebuild(*args, **kwargs):
        raise AssertionError("Unnecessary rebuild was triggered for valid weights")
    monkeypatch.setattr(mf, "build_multi_output_model", _fail_rebuild)

    loaded, path = ModelFactory.get_model(
        version="v2",
        input_dim=dim,
        checkpoint_path=save_path,
    )
    assert getattr(loaded, "is_calibrated", None) is False
    assert getattr(loaded, "calibration_temperature", None) is None
    assert getattr(loaded, "calibration_metadata", None) is None


def test_model_factory_valid_model_with_sidecar_is_calibrated(tmp_path, monkeypatch):
    dim = 16
    m = _tiny_model(input_dim=dim)
    save_path = Path(tmp_path / "model_v2.keras")
    m.save(save_path)

    manifest_path = save_path.with_suffix(".split.json")
    splits = {
        "train": [{"match_id": f"M_{i}", "game_creation_time": 1000 + i} for i in range(50)],
        "validation": [{"match_id": f"V_{i}", "game_creation_time": 2000 + i} for i in range(25)],
        "test": [{"match_id": f"T_{i}", "game_creation_time": 3000 + i} for i in range(25)],
    }
    manifest_data = build_manifest(splits, source_db=None, split_metadata={"strategy": "chronological"})
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    fit = {
        "status": "fitted",
        "partition": "validation",
        "count": 25,
        "class_counts": [12, 13],
        "temperature": 1.45,
        "optimizer_success": True,
        "boundary_hit": False,
        "log_loss_before": 0.693,
        "log_loss_after": 0.650,
    }
    save_calibration_artifact(save_path, manifest_path, fit, input_dim=dim)

    import core.model_factory as mf
    def _fail_rebuild(*args, **kwargs):
        raise AssertionError("Unnecessary rebuild was triggered")
    monkeypatch.setattr(mf, "build_multi_output_model", _fail_rebuild)

    loaded, path = ModelFactory.get_model(
        version="v2",
        input_dim=dim,
        checkpoint_path=str(save_path),
    )
    assert getattr(loaded, "is_calibrated", None) is True
    assert loaded.calibration_temperature == 1.45
    assert loaded.calibration_metadata["fit"]["count"] == 25


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
    assert res.get("uncalibrated") is True
