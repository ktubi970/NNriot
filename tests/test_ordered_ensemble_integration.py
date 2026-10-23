"""Compatibility and causal checks for ensemble evaluation."""
from __future__ import annotations

import copy
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from core.calibration import sha256_file
from core.experiment import (
    build_ordered_feature_contract, read_manifest, write_feature_contract,
)
from core.features.ordered_matchup import extract_ordered_features
from scripts import evaluate_ensemble as evaluator
from scripts.evaluate_ensemble import _model_input_policy, _restore_cached_splits
from tests.ordered_integration_fixture import make_ordered_fixture


class _Inputs:
    def __init__(self, *specs):
        self.inputs = [SimpleNamespace(name=name, shape=shape) for name, shape in specs]


def test_input_names_define_policy_even_when_macro_is_first():
    model = _Inputs(
        ("macro_clash", (None, 15)),
        ("team_b", (None, 5, 14)),
        ("team_a", (None, 5, 14)),
    )
    assert _model_input_policy(model) == (14, "v3")
    assert _model_input_policy(_Inputs(("team_b", (None, 5, 8)),
                                       ("team_a", (None, 5, 8)))) == (8, "tensors")
    with pytest.raises(ValueError, match="macro_clash"):
        _model_input_policy(_Inputs(("macro_clash", (None, 15)),
                                    ("team_a", (None, 5, 8)),
                                    ("team_b", (None, 5, 8))))


def test_cache_is_restored_against_manifest(tmp_path):
    fixture = make_ordered_fixture(tmp_path / "fixture")
    manifest = read_manifest(fixture["manifest"])
    restored = _restore_cached_splits(fixture["cache"], manifest)
    assert [r["match_id"] for r in restored["train"]] == [
        r["match_id"] for r in fixture["splits"]["train"]]

    foreign = copy.deepcopy(fixture["splits"])
    foreign["validation"][0]["feature_json"]["team_a"][0]["priors"][0] += 0.01
    foreign_path = tmp_path / "foreign.pkl"
    with foreign_path.open("wb") as handle:
        pickle.dump(foreign, handle)
    # The historical manifest cannot see an attached-prior change, but the
    # ordered contract's enriched hash must reject it in the full evaluator.
    assert _restore_cached_splits(foreign_path, manifest) == foreign

    foreign["train"][0]["game_creation_time"] += 1
    with foreign_path.open("wb") as handle:
        pickle.dump(foreign, handle)
    with pytest.raises(ValueError, match="timestamp mismatch"):
        _restore_cached_splits(foreign_path, manifest)


def test_missing_explicit_cache_is_rejected_before_database_read(tmp_path, monkeypatch):
    fixture = make_ordered_fixture(tmp_path / "fixture")
    checkpoint = _checkpoint_from_fixture(fixture, tmp_path, with_contract=False)
    monkeypatch.setattr(evaluator, "load_enriched_records", lambda **_kwargs: pytest.fail(
        "explicit cache absence must stop before database reconstruction"))
    with pytest.raises(FileNotFoundError, match="Explicit splits cache"):
        evaluator.evaluate_ensemble(
            str(checkpoint), str(fixture["db"]), samples=140,
            splits_cache=str(tmp_path / "missing.pkl"),
            output_dir=str(tmp_path / "output"),
        )
    assert not (tmp_path / "output").exists()


def test_ordered_ensemble_features_ignore_holdout_labels(tmp_path):
    fixture = make_ordered_fixture(tmp_path / "fixture")
    splits = fixture["splits"]
    ends = {row["match_id"]: row["game_creation_time"] + 30_000
            for row in splits["train"]}
    original, swapped, _ = extract_ordered_features(
        splits, ends, representation="tabular", matchup_weight=15.0)
    altered = copy.deepcopy(splits)
    for split in ("validation", "test"):
        for row in altered[split]:
            row["labels_json"]["winner"] = 1 - row["labels_json"]["winner"]
    changed, changed_swapped, _ = extract_ordered_features(
        altered, ends, representation="tabular", matchup_weight=15.0)
    for split in ("train", "validation", "test"):
        np.testing.assert_array_equal(original[split], changed[split])
        np.testing.assert_array_equal(swapped[split], changed_swapped[split])


def test_swapped_probabilities_are_antisymmetric():
    forward = np.array([0.1, 0.8, 0.4])
    swapped = np.array([0.7, 0.2, 0.6])
    p = 0.5 * (forward + 1.0 - swapped)
    reverse = 0.5 * (swapped + 1.0 - forward)
    np.testing.assert_allclose(p + reverse, 1.0)


def _checkpoint_from_fixture(fixture, tmp_path, *, with_contract):
    checkpoint = tmp_path / "model.keras"
    checkpoint.write_bytes(b"test checkpoint")
    checkpoint.with_suffix(".split.json").write_bytes(fixture["manifest"].read_bytes())
    if with_contract:
        ends = {row["match_id"]: row["game_creation_time"] + 30_000
                for row in fixture["splits"]["train"]}
        _, _, diagnostics = extract_ordered_features(
            fixture["splits"], ends, representation="tensors", matchup_weight=5.0)
        contract = build_ordered_feature_contract(
            fixture["splits"], result_end_times_sha256=sha256_file(fixture["ends"]),
            representation="tensors", matchup_weight=5.0, missing_policy="error",
            diagnostics=diagnostics, source_cache_sha256=sha256_file(fixture["cache"]),
        )
        write_feature_contract(checkpoint, contract)
        checkpoint.with_suffix(".result_end_times.json").write_bytes(fixture["ends"].read_bytes())
    return checkpoint


def test_fourteen_without_contract_requires_retraining(tmp_path, monkeypatch):
    fixture = make_ordered_fixture(tmp_path / "fixture")
    checkpoint = _checkpoint_from_fixture(fixture, tmp_path, with_contract=False)
    monkeypatch.setattr(evaluator.tf.keras.models, "load_model", lambda *_args, **_kwargs: _Inputs(
        ("team_a", (None, 5, 14)), ("team_b", (None, 5, 14))))
    with pytest.raises(ValueError, match="retraining is required"):
        evaluator.evaluate_ensemble(str(checkpoint), str(fixture["db"]), samples=140,
                                    splits_cache=str(fixture["cache"]), output_dir=str(tmp_path / "output"))
    assert not (tmp_path / "output").exists()


def test_calibration_bound_to_feature_contract(tmp_path, monkeypatch):
    fixture = make_ordered_fixture(tmp_path / "fixture")
    checkpoint = _checkpoint_from_fixture(fixture, tmp_path, with_contract=True)
    checkpoint.with_suffix(".calibration.json").write_text(json.dumps({
        "temperature": 1.0, "feature_contract_sha256": "0" * 64,
    }), encoding="utf-8")

    class Model(_Inputs):
        def predict(self, x, **_kwargs):
            return {"winner": np.tile(np.array([[0.5, 0.5]]), (len(x["team_a"]), 1))}

    monkeypatch.setattr(evaluator.tf.keras.models, "load_model", lambda *_args, **_kwargs: Model(
        ("team_a", (None, 5, 14)), ("team_b", (None, 5, 14))))
    with pytest.raises(ValueError, match="calibration feature_contract_sha256 mismatch"):
        evaluator.evaluate_ensemble(str(checkpoint), str(fixture["db"]), samples=140,
                                    splits_cache=str(fixture["cache"]), output_dir=str(tmp_path / "output"))
    assert not (tmp_path / "output").exists()
    checkpoint.with_suffix(".calibration.json").write_text(json.dumps({
        "feature_contract_sha256": sha256_file(checkpoint.with_suffix(".features.json")),
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="missing temperature"):
        evaluator.evaluate_ensemble(str(checkpoint), str(fixture["db"]), samples=140,
                                    splits_cache=str(fixture["cache"]), output_dir=str(tmp_path / "output"))
    assert not (tmp_path / "output").exists()


def test_enriched_cache_hash_mismatch_is_rejected(tmp_path, monkeypatch):
    fixture = make_ordered_fixture(tmp_path / "fixture")
    checkpoint = _checkpoint_from_fixture(fixture, tmp_path, with_contract=True)
    foreign = copy.deepcopy(fixture["splits"])
    foreign["train"][0]["feature_json"]["team_a"][0]["priors"][0] += 0.01
    foreign_path = tmp_path / "foreign.pkl"
    with foreign_path.open("wb") as handle:
        pickle.dump(foreign, handle)
    monkeypatch.setattr(evaluator.tf.keras.models, "load_model", lambda *_args, **_kwargs: _Inputs(
        ("team_a", (None, 5, 14)), ("team_b", (None, 5, 14))))
    with pytest.raises(ValueError, match="enriched_dataset_sha256"):
        evaluator.evaluate_ensemble(str(checkpoint), str(fixture["db"]), samples=140,
                                    splits_cache=str(foreign_path), output_dir=str(tmp_path / "output"))
    assert not (tmp_path / "output").exists()


def test_legacy_eight_without_contract_uses_historical_tensors(tmp_path, monkeypatch):
    fixture = make_ordered_fixture(tmp_path / "fixture")
    checkpoint = _checkpoint_from_fixture(fixture, tmp_path, with_contract=False)

    class Model(_Inputs):
        def predict(self, x, **_kwargs):
            assert x["team_a"].shape[-1] == 8
            return {"winner": np.tile(np.array([[0.45, 0.55]]), (len(x["team_a"]), 1))}

    monkeypatch.setattr(evaluator.tf.keras.models, "load_model", lambda *_args, **_kwargs: Model(
        ("team_b", (None, 5, 8)), ("team_a", (None, 5, 8))))
    source_hashes = {p: sha256_file(p) for p in (checkpoint, fixture["cache"], fixture["ends"])}
    output = tmp_path / "ensemble_run"
    report = evaluator.evaluate_ensemble(
        str(checkpoint), str(fixture["db"]), samples=140,
        splits_cache=str(fixture["cache"]), result_end_times=str(fixture["ends"]),
        output_dir=str(output),
    )
    assert report["neural_matchup_weight"] is None
    assert report["tree_matchup_weight"] == 15.0
    assert np.isfinite(report["tri_blend_ensemble_metrics"]["log_loss"])
    assert json.loads((output / "ensemble_comparison.json").read_text()) == report
    with pytest.raises(FileExistsError):
        evaluator.evaluate_ensemble(
            str(checkpoint), str(fixture["db"]), samples=140,
            splits_cache=str(fixture["cache"]), result_end_times=str(fixture["ends"]),
            output_dir=str(output),
        )
    assert {p: sha256_file(p) for p in source_hashes} == source_hashes


def test_real_fourth_consumer_with_saved_ordered_checkpoint(tmp_path):
    from scripts.train_deepsets_v2 import train_deepsets_v2

    fixture = make_ordered_fixture(tmp_path / "fixture")
    model_dir = tmp_path / "v2"
    train_deepsets_v2(
        str(fixture["cache"]), str(model_dir), epochs=1, batch_size=32,
        result_end_times=str(fixture["ends"]),
    )
    checkpoint = model_dir / "model.keras"
    source_paths = [
        checkpoint, checkpoint.with_suffix(".split.json"),
        checkpoint.with_suffix(".calibration.json"),
        checkpoint.with_suffix(".features.json"),
        checkpoint.with_suffix(".result_end_times.json"),
        fixture["cache"], fixture["ends"],
    ]
    source_hashes = {path: sha256_file(path) for path in source_paths}
    output = tmp_path / "ensemble"
    real_extract = evaluator.extract_ordered_features
    calls = []

    def counted_extract(*args, **kwargs):
        calls.append((kwargs["representation"], kwargs["matchup_weight"]))
        return real_extract(*args, **kwargs)

    from unittest.mock import patch
    with patch.object(evaluator, "extract_ordered_features", side_effect=counted_extract):
        report = evaluator.evaluate_ensemble(
            str(checkpoint), str(fixture["db"]), samples=140,
            splits_cache=str(fixture["cache"]), output_dir=str(output),
        )
    assert calls == [("tensors", 5.0), ("tabular", 15.0)]
    assert report["split_counts"] == {"train": 98, "validation": 21, "test": 21}
    assert report["neural_matchup_weight"] == 5.0
    assert report["tree_matchup_weight"] == 15.0
    assert report["calibration_applied"]
    for key in ("deepsets_metrics", "lightgbm_metrics", "gbdt_metrics",
                "tri_blend_ensemble_metrics"):
        assert all(np.isfinite(value) for value in report[key].values())
    assert json.loads((output / "ensemble_comparison.json").read_text()) == report
    with pytest.raises(FileExistsError):
        evaluator.evaluate_ensemble(
            str(checkpoint), str(fixture["db"]), samples=140,
            splits_cache=str(fixture["cache"]), output_dir=str(output),
        )
    assert {path: sha256_file(path) for path in source_paths} == source_hashes


def test_ordered_model_without_calibration_reports_raw_probabilities(tmp_path, monkeypatch):
    fixture = make_ordered_fixture(tmp_path / "fixture")
    checkpoint = _checkpoint_from_fixture(fixture, tmp_path, with_contract=True)

    class Model(_Inputs):
        def predict(self, x, **_kwargs):
            return {"winner": np.tile(np.array([[0.4, 0.6]]), (len(x["team_a"]), 1))}

    monkeypatch.setattr(evaluator.tf.keras.models, "load_model", lambda *_args, **_kwargs: Model(
        ("team_a", (None, 5, 14)), ("team_b", (None, 5, 14))))
    report = evaluator.evaluate_ensemble(
        str(checkpoint), str(fixture["db"]), samples=140,
        splits_cache=str(fixture["cache"]), output_dir=str(tmp_path / "output"),
    )
    assert report["calibration_applied"] is False
    assert report["calibration_sha256"] is None


def test_ordered_model_restores_from_database_without_training_cache(tmp_path, monkeypatch):
    fixture = make_ordered_fixture(tmp_path / "fixture")
    checkpoint = _checkpoint_from_fixture(fixture, tmp_path, with_contract=True)
    assert not (tmp_path / "splits_cache_140.pkl").exists()

    class Model(_Inputs):
        def predict(self, x, **_kwargs):
            return {"winner": np.tile(np.array([[0.4, 0.6]]), (len(x["team_a"]), 1))}

    monkeypatch.setattr(evaluator.tf.keras.models, "load_model", lambda *_args, **_kwargs: Model(
        ("team_a", (None, 5, 14)), ("team_b", (None, 5, 14))))
    monkeypatch.setattr(evaluator, "load_enriched_records", lambda **_kwargs: fixture["records"])
    report = evaluator.evaluate_ensemble(
        str(checkpoint), str(fixture["db"]), samples=140,
        output_dir=str(tmp_path / "database_run"),
    )
    assert report["source_cache_sha256"] is None
    assert report["enriched_dataset_sha256"]

    changed = copy.deepcopy(fixture["records"])
    changed[0]["feature_json"]["team_a"][0]["priors"][0] += 0.01
    monkeypatch.setattr(evaluator, "load_enriched_records", lambda **_kwargs: changed)
    with pytest.raises(ValueError, match="enriched_dataset_sha256"):
        evaluator.evaluate_ensemble(
            str(checkpoint), str(fixture["db"]), samples=140,
            output_dir=str(tmp_path / "tampered_database_run"),
        )
    assert not (tmp_path / "tampered_database_run").exists()
