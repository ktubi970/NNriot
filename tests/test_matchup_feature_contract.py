"""Compatibility and persistence of ordered matchup model contracts."""
import json
import math

import pytest

from core.experiment import read_feature_contract, write_feature_contract


def _contract(**updates):
    contract = {
        "schema_version": 1, "policy": "ordered_matchup_v1", "representation": "tensors",
        "player_feature_dim": 14, "macro_feature_dim": 0,
        "matchup_weight": 5.0, "global_weight": 50.0,
        "split_sha256": "a"*64, "enriched_dataset_sha256": "b"*64,
        "result_end_times_sha256": "c"*64,
        "availability_assumption": "explicit_result_end_proxy", "missing_policy": "error",
        "diagnostics": {"requested": 2, "valid": 2, "missing": 0},
        "artifact_scope": "training_evaluation",
        "ordered_matchup_sha256": "d"*64, "champion_registry_sha256": "e"*64,
        "source_sha256": {name: "f"*64 for name in (
            "ordered_matchup.py", "pipeline.py", "matchup_matrix.py",
            "draft_composition.py", "static_meta.py", "champion_registry.py")},
    }
    contract.update(updates)
    return contract


def test_missing_feature_contract_fails_explicitly(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_feature_contract(tmp_path / "model.keras")


def test_round_trip_and_written_bytes_hash(tmp_path):
    import hashlib
    checkpoint = tmp_path / "model.keras"
    digest = write_feature_contract(checkpoint, _contract())
    path = checkpoint.with_suffix(".features.json")
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert read_feature_contract(checkpoint) == _contract()
    with pytest.raises(FileExistsError):
        write_feature_contract(checkpoint, _contract())


@pytest.mark.parametrize("updates", [
    {"schema_version": 2}, {"policy": "greedy"}, {"matchup_weight": 0},
    {"schema_version": True},
    {"matchup_weight": -1}, {"matchup_weight": math.inf}, {"matchup_weight": math.nan},
    {"global_weight": 0}, {"player_feature_dim": 8},
    {"representation": "v3", "macro_feature_dim": 0},
    {"representation": "tensors", "macro_feature_dim": 15},
    {"result_end_times_sha256": "bad"}, {"artifact_scope": "serving"},
])
def test_incompatible_contract_is_rejected(tmp_path, updates):
    with pytest.raises(ValueError):
        write_feature_contract(tmp_path / "model.keras", _contract(**updates))


def test_weights_are_preserved_as_distinct_contracts(tmp_path):
    five = tmp_path / "five.keras"
    fifteen = tmp_path / "fifteen.keras"
    assert write_feature_contract(five, _contract(matchup_weight=5.0)) != \
           write_feature_contract(fifteen, _contract(matchup_weight=15.0))
    assert read_feature_contract(five)["matchup_weight"] == 5.0
    assert read_feature_contract(fifteen)["matchup_weight"] == 15.0


def test_builder_records_audit_hashes_without_using_them_as_compatibility_gate():
    from test_matchup_provenance import _splits
    from core.experiment import build_ordered_feature_contract
    contract = build_ordered_feature_contract(
        _splits(), result_end_times_sha256="c"*64, representation="v3",
        matchup_weight=5.0, missing_policy="error",
        diagnostics={"requested": 2, "found": 2, "invalid": 0, "valid": 2,
                     "missing": 0, "excluded": 0},
    )
    assert contract["macro_feature_dim"] == 15
    assert set(contract["source_sha256"]) == {
        "ordered_matchup.py", "pipeline.py", "matchup_matrix.py",
        "draft_composition.py", "static_meta.py", "champion_registry.py",
    }


def test_invalid_json_and_tampered_contract_are_rejected(tmp_path):
    path = (tmp_path / "model.keras").with_suffix(".features.json")
    path.write_text("{")
    with pytest.raises(ValueError, match="JSON"):
        read_feature_contract(tmp_path / "model.keras")
    path.write_text(json.dumps(_contract(policy="unknown")))
    with pytest.raises(ValueError, match="policy"):
        read_feature_contract(tmp_path / "model.keras")
