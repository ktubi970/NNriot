from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest

from core.experiment import build_manifest
from core.pre_match_dataset import chronological_split
from core.calibration import (
    apply_temperature_to_probabilities,
    fit_temperature,
    invalidate_calibration,
    load_calibration_artifact,
    save_calibration_artifact,
    sha256_file,
)


def _records(n=40):
    from core.feature_labels import LABEL_KEYS
    rows = []
    for i in range(n):
        labels = dict.fromkeys(LABEL_KEYS, 0)
        labels['winner'] = i % 2
        labels['total_kills'] = 30
        labels['team_a_kills'] = 15
        labels['team_b_kills'] = 15
        labels['total_towers'] = 10
        labels['total_dragons'] = 3
        player = {'championName': 'Ahri', 'teamPosition': 'MIDDLE',
                  'gold': 10000, 'career_win_rate': 0.99}
        rows.append({'match_id': f'M_{i:04d}', 'game_creation_time': 1000 + i,
                     'winner_label': i % 2, 'labels_json': labels,
                     'feature_json': {'game_version': '16.18.1',
                                      'season_id': 16,
                                      'region': 'EUW',
                                      'team_a': [dict(player) for _ in range(5)],
                                      'team_b': [dict(player) for _ in range(5)]}})
    return rows


def test_probability_contract_keeps_favorite_and_normalization():
    p = np.array([0.1, 0.5, 0.9])
    q = apply_temperature_to_probabilities(p, 2.0)
    np.testing.assert_allclose(q, [0.25, 0.5, 0.75])
    np.testing.assert_array_equal(q >= 0.5, p >= 0.5)
    np.testing.assert_allclose(np.column_stack((1 - q, q)).sum(axis=1), 1)


def test_identity_keeps_endpoints_without_double_sigmoid():
    p = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    np.testing.assert_array_equal(apply_temperature_to_probabilities(p, 1.0), p)


@pytest.mark.parametrize('temperature', [0, -1, np.nan, np.inf, True, '2'])
def test_invalid_temperature_is_rejected(temperature):
    with pytest.raises(ValueError):
        apply_temperature_to_probabilities(np.array([0.1, 0.9]), temperature)


@pytest.mark.parametrize('prob_input', [
    [],
    [[0.1], [0.9]],
    [np.nan, 0.5],
    [-0.1, 0.5],
    [0.5, 1.1],
])
def test_invalid_probability_inputs_are_rejected(prob_input):
    with pytest.raises(ValueError):
        apply_temperature_to_probabilities(np.array(prob_input), 1.5)


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


def test_fit_rejects_mismatched_lengths_and_nonbinary_labels():
    with pytest.raises(ValueError):
        fit_temperature(np.full(25, 0.5), np.zeros(20))
    with pytest.raises(ValueError):
        fit_temperature(np.full(20, 0.5), np.array([0, 2] * 10))


def test_save_and_load_calibration_artifact_roundtrip_and_tampering(tmp_path):
    rows = _records(140)  # gives 21 validation rows
    splits = chronological_split(rows)
    assert len(splits['validation']) >= 20

    manifest = build_manifest(splits, source_db=None,
                              split_metadata={'requested_strategy': 'chronological',
                                              'effective_strategy': 'chronological',
                                              'fallback_reason': None})

    ckpt_path = tmp_path / "model.keras"
    ckpt_path.write_bytes(b"fake_checkpoint_bytes_for_testing")
    manifest_path = tmp_path / "model.split.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    val_y = np.array([r['labels_json']['winner'] for r in splits['validation']])
    val_p = np.full(len(val_y), 0.5)
    fit = fit_temperature(val_p, val_y)

    # Save
    sidecar_path = save_calibration_artifact(ckpt_path, manifest_path, fit, input_dim=16)
    assert sidecar_path.exists()

    # Duplicate save raises FileExistsError
    with pytest.raises(FileExistsError):
        save_calibration_artifact(ckpt_path, manifest_path, fit, input_dim=16)

    # Valid load
    loaded = load_calibration_artifact(ckpt_path, input_dim=16)
    assert loaded is not None
    assert loaded['temperature'] == fit['temperature']
    assert loaded['fit']['count'] == fit['count']
    assert loaded['fit']['class_counts'] == fit['class_counts']

    # Missing sidecar returns None
    assert load_calibration_artifact(tmp_path / "missing.keras", input_dim=16) is None

    # Tamper with checkpoint bytes -> returns None
    tampered_ckpt = tmp_path / "tampered_ckpt.keras"
    tampered_ckpt.write_bytes(b"tampered")
    # copy companion manifest and sidecar
    manifest_copy = tmp_path / "tampered_ckpt.split.json"
    manifest_copy.write_bytes(manifest_path.read_bytes())
    sidecar_copy = tmp_path / "tampered_ckpt.calibration.json"
    sidecar_copy.write_bytes(sidecar_path.read_bytes())
    assert load_calibration_artifact(tampered_ckpt, input_dim=16) is None

    # Helper to test sidecar field tampering
    def assert_tampered_sidecar_fails(mutator):
        test_dir = tmp_path / f"tamper_{np.random.randint(1000000)}"
        test_dir.mkdir()
        c = test_dir / "m.keras"
        c.write_bytes(ckpt_path.read_bytes())
        m = test_dir / "m.split.json"
        m.write_bytes(manifest_path.read_bytes())
        sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        mutator(sidecar_data)
        (test_dir / "m.calibration.json").write_text(json.dumps(sidecar_data), encoding="utf-8")
        assert load_calibration_artifact(c, input_dim=16) is None

    assert_tampered_sidecar_fails(lambda d: d.update(input_dim=32))
    assert_tampered_sidecar_fails(lambda d: d.update(method="unknown_method"))
    assert_tampered_sidecar_fails(lambda d: d.update(class_order=["B", "A"]))
    assert_tampered_sidecar_fails(lambda d: d.update(temperature=-1.0))
    assert_tampered_sidecar_fails(lambda d: d['fit'].update(count=15))
    assert_tampered_sidecar_fails(lambda d: d['fit'].update(class_counts=[10, 5]))
    assert_tampered_sidecar_fails(lambda d: d['fit'].update(log_loss_after=d['fit']['log_loss_before'] + 1.0))


def test_invalidate_calibration(tmp_path):
    class FakeModel:
        is_calibrated = True
        calibration_temperature = 1.5
        calibration_metadata = {"test": 1}

    model = FakeModel()
    ckpt = tmp_path / "inv.keras"
    sidecar = tmp_path / "inv.calibration.json"
    sidecar.write_text("{}", encoding="utf-8")

    invalidate_calibration(model, ckpt)
    assert model.is_calibrated is False
    assert model.calibration_temperature is None
    assert model.calibration_metadata is None
    assert not sidecar.exists()
