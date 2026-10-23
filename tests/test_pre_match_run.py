import json
from pathlib import Path

import numpy as np
import pytest


def _records(n=40):
    from core.feature_labels import LABEL_KEYS

    records = []
    for i in range(n):
        labels = dict.fromkeys(LABEL_KEYS, 0)
        labels.update(winner=i % 2, total_kills=30, team_a_kills=15,
                      team_b_kills=15, total_towers=10, total_dragons=3)
        player = {"champion": "Ahri", "role": "MIDDLE", "gold": 10000,
                  "career_win_rate": 0.99, "career_matches": 500}
        records.append({
            "id": n - i, "match_id": f"MATCH_{i:03d}",
            "game_creation_time": 1700000000000 + i * 1000,
            "winner_label": i % 2, "labels_json": labels,
            "feature_json": {"game_mode": "CLASSIC", "game_version": "16.18",
                             "team_a": [dict(player) for _ in range(5)],
                             "team_b": [dict(player) for _ in range(5)]},
        })
    return records


def _tiny_model(input_dim):
    import tensorflow as tf
    from core.generate_graph import LOSS_PER_HEAD, LOSS_WEIGHTS

    inputs = tf.keras.Input((input_dim,))
    outputs = {}
    for name, loss in LOSS_PER_HEAD.items():
        if loss == "categorical_crossentropy":
            units = 2 if name in ("winner", "team_b_kill_lead") else 3
            activation = "softmax"
        else:
            units = 1
            activation = "sigmoid" if loss == "binary_crossentropy" else "linear"
        outputs[name] = tf.keras.layers.Dense(units, activation=activation,
                                             name=name)(inputs)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss=LOSS_PER_HEAD, loss_weights=LOSS_WEIGHTS)
    return model


def test_pre_match_run_excludes_undated_career_statistics():
    from scripts.train_pre_match_model import to_pre_match_features

    clean = to_pre_match_features(_records(1)[0]["feature_json"])
    assert clean["team_a"][0] == {"champion": "Ahri", "role": "MIDDLE"}
    raw = {"info": {"participants": [{"teamId": 100, "championName": "Ahri",
            "teamPosition": "MIDDLE", "career_win_rate": 0.99, "career_matches": 500}]}}
    assert to_pre_match_features(raw)["team_a"][0] == clean["team_a"][0]


def test_run_persists_disjoint_splits_and_held_out_metrics(tmp_path, monkeypatch):
    from scripts import train_pre_match_model as training

    monkeypatch.setattr(training, "build_multi_output_model", _tiny_model)
    output = tmp_path / "pilot.keras"
    model = training.train_model(_records(), epochs=2, batch_size=8,
                                 checkpoint_path=str(output), dim=16)
    assert output.exists()
    assert model.input_shape[-1] == 16
    report = json.loads(output.with_suffix(".metrics.json").read_text())
    manifest = json.loads(output.with_suffix(".split.json").read_text())
    assert report["split_counts"] == {"train": 28, "validation": 6, "test": 6}
    assert report["is_calibrated"] is False
    assert report["selection_metric"] == "validation.winner.log_loss"
    assert report["selected_epoch"] in (1, 2)
    groups = manifest["splits"]
    ids = [row["match_id"] for rows in groups.values() for row in rows]
    assert len(ids) == len(set(ids)) == 40
    assert max(r["game_creation_time"] for r in groups["train"]) < min(
        r["game_creation_time"] for r in groups["validation"])
    assert max(r["game_creation_time"] for r in groups["validation"]) < min(
        r["game_creation_time"] for r in groups["test"])
    assert report["train_positive_rate"] == 0.5
    assert report["metrics"]["test"]["baseline"]["log_loss"] == pytest.approx(np.log(2))
    assert report["metrics"]["test"]["baseline"]["brier"] == pytest.approx(0.25)
    assert np.isfinite(report["metrics"]["test"]["model"]["log_loss"])
    assert manifest["schema_version"] == 2
    assert len(manifest["dataset_sha256"]) == 64
    assert len(manifest["split_sha256"]) == 64
    assert len(report["split_sha256"]) == 64
    from tensorflow import keras
    restored = keras.models.load_model(output)
    np.testing.assert_allclose(model.get_weights()[0], restored.get_weights()[0])


def test_existing_checkpoint_is_never_overwritten(tmp_path):
    from scripts.train_pre_match_model import train_model

    output = tmp_path / "existing.keras"
    output.write_bytes(b"keep existing model")
    with pytest.raises(FileExistsError):
        train_model(_records(), epochs=1, checkpoint_path=str(output), dim=16)
    assert output.read_bytes() == b"keep existing model"


@pytest.mark.parametrize("kwargs", [{"epochs": 0}, {"batch_size": 0}, {"dim": 0}])
def test_invalid_run_settings_fail_before_building_model(tmp_path, monkeypatch, kwargs):
    from scripts import train_pre_match_model as training

    def forbidden_builder(**kwargs):
        pytest.fail("invalid settings must be rejected before model creation")

    monkeypatch.setattr(training, "build_multi_output_model", forbidden_builder)
    with pytest.raises(ValueError):
        training.train_model(_records(), checkpoint_path=str(tmp_path / "bad.keras"), **kwargs)
    assert not list(tmp_path.iterdir())


def test_fetch_uses_project_relative_database_outside_working_directory(tmp_path, monkeypatch):
    from core import config, database
    from scripts import train_pre_match_model as training

    project = tmp_path / "project"
    project.mkdir()
    path = project / "source.db"
    database.init_db(str(path))
    monkeypatch.setattr(config, "PROJECT_ROOT", project)
    monkeypatch.chdir(tmp_path)
    assert training.fetch_training_records(db_path="source.db") == []
    with pytest.raises(FileNotFoundError, match="missing.db"):
        training.fetch_training_records(db_path="missing.db")
    assert not (project / "missing.db").exists()


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


def test_preexisting_calibration_sidecar_blocks_output(tmp_path, monkeypatch):
    from scripts import train_pre_match_model as training

    def forbidden_builder(**kwargs):
        pytest.fail("builder must not be called when calibration sidecar already exists")

    monkeypatch.setattr(training, 'build_multi_output_model', forbidden_builder)
    output = tmp_path / 'candidate.keras'
    output.with_suffix('.calibration.json').write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        training.train_model(_records(40), epochs=1, checkpoint_path=str(output), dim=16)


def test_calibrate_prerequisites_fail_before_model_building_or_output_creation(tmp_path, monkeypatch):
    from scripts import train_pre_match_model as training

    def forbidden_builder(**kwargs):
        pytest.fail("builder must not be called when validation prerequisites fail")

    monkeypatch.setattr(training, 'build_multi_output_model', forbidden_builder)
    output = tmp_path / 'bad_cal.keras'

    # Fewer than 20 validation rows (n=40 gives 6 validation rows)
    with pytest.raises(ValueError, match="20"):
        training.train_model(_records(40), epochs=1, checkpoint_path=str(output), dim=16, calibrate=True)
    assert not output.parent.exists() or not list(output.parent.iterdir())

    # 200 rows but single class in validation
    rows_single_class = _records(200)
    for r in rows_single_class[140:170]:  # validation slice
        r['labels_json']['winner'] = 0
        r['winner_label'] = 0
    with pytest.raises(ValueError, match="both classes"):
        training.train_model(rows_single_class, epochs=1, checkpoint_path=str(output), dim=16, calibrate=True)


def test_simulated_fit_failure_leaves_no_checkpoint_or_sidecar(tmp_path, monkeypatch):
    from scripts import train_pre_match_model as training

    monkeypatch.setattr(training, 'build_multi_output_model', _tiny_model)

    def failing_fit(*args, **kwargs):
        raise ValueError("simulated optimizer failure")

    monkeypatch.setattr(training, 'fit_temperature', failing_fit)
    output = tmp_path / 'fail_fit.keras'

    with pytest.raises(ValueError, match="simulated optimizer failure"):
        training.train_model(_records(200), epochs=1, batch_size=16, dim=16,
                             checkpoint_path=str(output), calibrate=True)

    assert not output.exists()
    assert not output.with_suffix('.calibration.json').exists()
    assert not output.with_suffix('.metrics.json').exists()
    assert output.with_suffix('.split.json').exists()  # manifest remains for diagnosis


def test_calibrated_candidate_spies_and_inference_count(tmp_path, monkeypatch):
    from scripts import train_pre_match_model as training
    monkeypatch.setattr(training, 'build_multi_output_model', _tiny_model)

    prob_calls = []
    original_prob = training._winner_probabilities
    def spy_winner_probabilities(model, x, batch_size):
        prob_calls.append(x.shape[0])
        return original_prob(model, x, batch_size)
    monkeypatch.setattr(training, '_winner_probabilities', spy_winner_probabilities)

    fit_calls = []
    original_fit = training.fit_temperature
    def spy_fit(p, y):
        fit_calls.append((len(p), len(y)))
        return original_fit(p, y)
    monkeypatch.setattr(training, 'fit_temperature', spy_fit)

    output = tmp_path / 'calibrated_spies.keras'
    training.train_model(_records(200), epochs=1, batch_size=16, dim=16,
                         checkpoint_path=str(output), calibrate=True)

    # 1 in training loop on validation (30), after reload: 1 on train (140), 1 on val (30), 1 on test (30)
    assert prob_calls.count(30) == 3
    assert len(fit_calls) == 1
    assert fit_calls[0] == (30, 30)
