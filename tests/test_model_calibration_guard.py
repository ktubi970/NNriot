import pytest
from core.model_factory import ModelFactory
from core.generate_graph import build_multi_output_model
import app.core as app_core


def test_model_factory_marks_rebuilt_model_as_uncalibrated(tmp_path):
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
