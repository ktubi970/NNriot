"""End-to-end checks for ordered neural training consumers."""
from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from core.experiment import (build_ordered_feature_contract, read_feature_contract,
                             read_manifest, restore_splits, write_feature_contract)
from core.calibration import sha256_file
from core.features.ordered_matchup import extract_ordered_features
from tests.ordered_integration_fixture import make_ordered_fixture


def test_v2_rejects_occupied_output_before_training(tmp_path):
    from scripts.train_deepsets_v2 import train_deepsets_v2
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    sentinel = occupied / "model.keras"
    sentinel.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="occupied"):
        train_deepsets_v2("missing.pkl", str(occupied), epochs=1)
    assert sentinel.read_bytes() == b"existing"


def test_v3_rejects_mismatched_cache_and_manifest(tmp_path):
    from scripts.train_deepsets_v3 import train_deepsets_v3
    data = make_ordered_fixture(tmp_path / "data")
    manifest = json.loads(data["manifest"].read_text(encoding="utf-8"))
    manifest["splits"]["train"][0]["match_id"] = "WRONG"
    from core.experiment import compute_split_sha256
    manifest["split_sha256"] = compute_split_sha256(manifest["splits"])
    wrong = tmp_path / "wrong.split.json"
    wrong.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "v3"
    with pytest.raises(ValueError):
        train_deepsets_v3(str(data["db"]), str(output), manifest_path=str(wrong),
                          splits_cache=str(data["cache"]), epochs=1)
    assert not output.exists()


def test_v3_missing_explicit_cache_fails_before_streaming(tmp_path, monkeypatch):
    from scripts import train_deepsets_v3 as trainer
    def unexpected_stream(**kwargs):
        raise AssertionError("database streaming should not be reached")
    monkeypatch.setattr(trainer, "load_enriched_records", unexpected_stream)
    with pytest.raises(FileNotFoundError, match="Splits cache"):
        trainer.train_deepsets_v3("missing.db", str(tmp_path / "output"),
                                  splits_cache=str(tmp_path / "missing.pkl"), epochs=1)
    assert not (tmp_path / "output").exists()


def test_replay_rejects_changed_prior_with_companion_contract(tmp_path):
    from scripts.train_deepsets_v3 import train_deepsets_v3
    from scripts.train_pre_match_model import main as general_main
    data = make_ordered_fixture(tmp_path / "data")
    companion_checkpoint = data["manifest"].with_name("splits.keras")
    contract = build_ordered_feature_contract(
        data["splits"], result_end_times_sha256=sha256_file(data["ends"]),
        representation="tensors", matchup_weight=5.0, missing_policy="error",
        diagnostics={},
    )
    write_feature_contract(companion_checkpoint, contract)
    changed = pickle.loads(data["cache"].read_bytes())
    changed["train"][0]["feature_json"]["team_a"][0]["priors"][0] += 0.01
    changed_cache = tmp_path / "changed.pkl"
    changed_cache.write_bytes(pickle.dumps(changed))
    with pytest.raises(ValueError, match="enriched_dataset_sha256"):
        train_deepsets_v3(str(data["db"]), str(tmp_path / "v3"),
                          manifest_path=str(data["manifest"]),
                          splits_cache=str(changed_cache), epochs=1)
    assert not (tmp_path / "v3").exists()
    with pytest.raises(SystemExit) as ex:
        general_main(["--db", str(data["db"]), "--split-manifest", str(data["manifest"]),
                      "--splits-cache", str(changed_cache), "--feature-version", "deepsets_v2",
                      "--output", str(tmp_path / "general" / "model.keras")])
    assert ex.value.code == 2
    assert not (tmp_path / "general").exists()


def test_general_cli_rejects_cache_without_manifest(tmp_path, monkeypatch):
    from scripts import train_pre_match_model as trainer
    def unexpected_loader(*args, **kwargs):
        raise AssertionError("database should not be loaded")
    monkeypatch.setattr(trainer, "fetch_training_records", unexpected_loader)
    with pytest.raises(SystemExit) as ex:
        trainer.main(["--db", "unused.db", "--splits-cache", "unused.pkl"])
    assert ex.value.code == 2


def test_general_cli_replays_enriched_cache_without_rebuilding_priors(tmp_path, monkeypatch):
    from scripts import train_pre_match_model as trainer
    data = make_ordered_fixture(tmp_path / "data")
    captured = {}
    def observe(records, **kwargs):
        captured["records"] = records
        captured["source_cache"] = kwargs["source_cache"]
        captured["manifest"] = kwargs["split_manifest"]
    monkeypatch.setattr(trainer, "train_model", observe)
    trainer.main(["--db", str(data["db"]), "--split-manifest", str(data["manifest"]),
                  "--splits-cache", str(data["cache"]), "--feature-version", "deepsets_v2"])
    assert captured["source_cache"] == str(data["cache"])
    assert len(captured["records"]) == 140
    assert captured["records"][0]["feature_json"]["team_a"][0]["priors"] == [0.5] * 10
    assert captured["manifest"]["split_sha256"]


def test_v3_without_cache_or_manifest_uses_chronological_split(tmp_path, monkeypatch):
    from scripts import train_deepsets_v3 as trainer
    data = make_ordered_fixture(tmp_path / "data")
    monkeypatch.setattr(trainer, "load_enriched_records", lambda **kwargs: data["records"])
    class ReachedFirstBatch(Exception):
        pass
    def stop_at_batch(*args, **kwargs):
        raise ReachedFirstBatch
    monkeypatch.setattr(trainer, "pcgrad_step", stop_at_batch)
    output = tmp_path / "v3_fresh"
    with pytest.raises(ReachedFirstBatch):
        trainer.train_deepsets_v3(str(data["db"]), str(output),
                                  manifest_path=None, splits_cache=None,
                                  samples=140, epochs=1,
                                  result_end_times=str(data["ends"]))
    manifest = read_manifest(output / "model.split.json")
    assert len(restore_splits(data["records"], manifest)["train"]) == 98
    assert read_feature_contract(output / "model.keras")["representation"] == "v3"


def test_legacy_eight_dim_uses_historical_extractor_without_ends(tmp_path, monkeypatch):
    from scripts.train_pre_match_model import train_model
    import core.deepsets_dataset as legacy
    data = make_ordered_fixture(tmp_path / "data")
    calls = []
    original = legacy.batch_extract_team_tensors
    def observe(records):
        calls.append(len(records))
        return original(records)
    monkeypatch.setattr(legacy, "batch_extract_team_tensors", observe)
    ckpt = tmp_path / "legacy.keras"
    result = train_model(data["records"][:40], epochs=1, batch_size=16,
                         checkpoint_path=str(ckpt), feature_version="deepsets",
                         use_pcgrad=True)
    assert calls == [28, 6, 6]
    assert ckpt.exists()
    assert result.model.input_shape[0][-1] == 8
    assert not ckpt.with_suffix(".features.json").exists()


def test_real_ordered_training_paths(tmp_path, monkeypatch):
    from scripts import train_deepsets_v2 as v2_trainer
    from scripts import train_deepsets_v3 as v3_trainer
    from scripts.train_pre_match_model import train_model
    import core.pcgrad as pcgrad_module
    data = make_ordered_fixture(tmp_path / "data")
    ends = {r["match_id"]: r["game_creation_time"] + 30_000
            for r in data["splits"]["train"]}
    observed = {"v2": [], "v3": [], "general": []}
    original_step = pcgrad_module.pcgrad_step
    def capture(path):
        def wrapped(model, optimizer, xb, yb, loss_fns, loss_weights):
            observed[path].append({k: np.asarray(v).copy() for k, v in xb.items()})
            return original_step(model, optimizer, xb, yb, loss_fns,
                                 loss_weights=loss_weights)
        return wrapped
    monkeypatch.setattr(v2_trainer, "pcgrad_step", capture("v2"))
    monkeypatch.setattr(v3_trainer, "pcgrad_step", capture("v3"))
    monkeypatch.setattr(pcgrad_module, "pcgrad_step", capture("general"))
    v2 = tmp_path / "v2"
    v2_trainer.train_deepsets_v2(str(data["cache"]), str(v2), epochs=1, batch_size=32,
                                 result_end_times=str(data["ends"]))
    v3 = tmp_path / "v3"
    v3_trainer.train_deepsets_v3(str(data["db"]), str(v3), manifest_path=str(data["manifest"]),
                                 splits_cache=str(data["cache"]), epochs=1, batch_size=32,
                                 result_end_times=str(data["ends"]))
    general = tmp_path / "general" / "model.keras"
    train_model(data["records"], epochs=1, batch_size=32,
                checkpoint_path=str(general), feature_version="deepsets_v2",
                source_db=str(data["db"]), result_end_times=str(data["ends"]),
                calibrate=True, use_pcgrad=True)
    for path, representation, weight in (("v2", "tensors", 5.0),
                                         ("v3", "v3", 5.0),
                                         ("general", "tensors", 15.0)):
        expected = extract_ordered_features(data["splits"], ends,
                                            representation=representation,
                                            matchup_weight=weight)[0]["train"]
        assert observed[path]
        actual_a = np.concatenate([batch["team_a"] for batch in observed[path]])
        # v2/v3 shuffle each epoch; compare sorted matchup deltas received by real batches.
        np.testing.assert_allclose(np.sort(actual_a[:, :, 12], axis=0),
                                   np.sort(expected["team_a"][:, :, 12], axis=0))
        if representation == "v3":
            actual_macro = np.concatenate([batch["macro_clash"] for batch in observed[path]])
            assert actual_macro.shape == expected["macro_clash"].shape
    for ckpt, representation, weight in ((v2 / "model.keras", "tensors", 5.0),
                                         (v3 / "model.keras", "v3", 5.0),
                                         (general, "tensors", 15.0)):
        assert ckpt.exists()
        contract = read_feature_contract(ckpt)
        assert contract["representation"] == representation
        assert contract["matchup_weight"] == weight
        assert contract["diagnostics"]["train_results_eligible"] > 0
        saved_manifest = read_manifest(ckpt.with_suffix(".split.json"))
        assert len(restore_splits(data["records"], saved_manifest)["train"]) > 0
        calibration = json.loads(ckpt.with_suffix(".calibration.json").read_text())
        assert calibration["feature_contract_sha256"] == sha256_file(ckpt.with_suffix(".features.json"))
    import tensorflow as tf
    for ckpt in (v2 / "model.keras", v3 / "model.keras", general):
        loaded = tf.keras.models.load_model(ckpt)
        contract = read_feature_contract(ckpt)
        x = extract_ordered_features(data["splits"], ends,
                                     representation=contract["representation"],
                                     matchup_weight=contract["matchup_weight"])[0]["test"]
        pred = loaded.predict({k: v[:3] for k, v in x.items()}, verbose=0)
        assert np.isfinite(np.asarray(pred["winner"])).all()
