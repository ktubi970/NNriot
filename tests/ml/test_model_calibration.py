import pytest

pytestmark = [pytest.mark.ml, pytest.mark.slow]


@pytest.fixture(scope="module")
def graph_module():
    from core import generate_graph

    return generate_graph


def test_model_factory_marks_rebuilt_model_as_uncalibrated(tmp_path):
    from core.model_factory import ModelFactory

    model, path = ModelFactory.get_model(
        version="v2",
        input_dim=20000,
        checkpoint_dir=str(tmp_path),
        force_rebuild=True,
    )
    assert getattr(model, "is_calibrated", None) is False


def test_model_factory_marks_loaded_valid_model_as_calibrated(tmp_path, graph_module):
    from core.model_factory import ModelFactory

    m = graph_module.build_multi_output_model(input_dim=20000)
    save_path = str(tmp_path / "model_v2.keras")
    m.save(save_path)

    loaded, path = ModelFactory.get_model(
        version="v2",
        input_dim=20000,
        checkpoint_path=save_path,
    )
    assert getattr(loaded, "is_calibrated", None) is True


def test_final_predict_rejects_uncalibrated_model(monkeypatch):
    import app.core as app_core

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
