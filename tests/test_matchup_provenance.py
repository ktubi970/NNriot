"""Strict provenance for ordered matchup result availability."""
from copy import deepcopy
import json
import sqlite3

import pytest

from core.experiment import (
    compute_dataset_sha256, compute_enriched_dataset_sha256,
    resolve_result_end_times, write_result_end_times_artifact,
)


def _splits():
    def row(mid, creation):
        player = {"championName": "Ahri", "teamPosition": "MIDDLE", "priors": [0.5]}
        return {"match_id": mid, "game_creation_time": creation,
                "feature_json": {"team_a": [deepcopy(player)], "team_b": [deepcopy(player)]},
                "labels_json": {"winner": 1}}
    return {"train": [row("a", 100), row("b", 200)],
            "validation": [row("c", 300)], "test": [row("d", 400)]}


def _artifact(tmp_path, splits=None, ends=None):
    splits = splits or _splits()
    ends = {"a": 150, "b": 250} if ends is None else ends
    path = tmp_path / "ends.json"
    provenance = {"availability_assumption": "explicit_result_end_proxy",
                  "missing_policy": "exclude" if len(ends) < 2 else "error",
                  "diagnostics": {"requested": 2, "found": len(ends), "invalid": 0,
                                  "valid": len(ends), "missing": 2-len(ends),
                                  "excluded": 2-len(ends)}}
    write_result_end_times_artifact(path, splits, ends, provenance)
    return path


def test_prior_change_invalidates_enriched_provenance_but_not_historical_hash(tmp_path):
    splits = _splits()
    path = _artifact(tmp_path, splits)
    changed = deepcopy(splits)
    changed["train"][0]["feature_json"]["team_a"][0]["priors"][0] += .01
    assert compute_dataset_sha256(changed) == compute_dataset_sha256(splits)
    assert compute_enriched_dataset_sha256(changed) != compute_enriched_dataset_sha256(splits)
    with pytest.raises(ValueError, match="enriched_dataset_sha256"):
        resolve_result_end_times(changed, artifact_path=path)


def test_strict_missing_and_explicit_partial_exclusion(tmp_path):
    path = _artifact(tmp_path, ends={"a": 150})
    with pytest.raises(ValueError, match="b"):
        resolve_result_end_times(_splits(), artifact_path=path)
    ends, provenance = resolve_result_end_times(_splits(), artifact_path=path, missing="exclude")
    assert ends == {"a": 150}
    assert provenance["diagnostics"]["missing"] == 1
    assert provenance["diagnostics"]["excluded"] == 1
    assert provenance["missing_policy"] == "exclude"


def test_zero_usable_results_refused_even_when_excluding(tmp_path):
    with pytest.raises(ValueError, match="no usable"):
        _artifact(tmp_path, ends={})
    path = _artifact(tmp_path)
    payload = json.loads(path.read_text())
    for row in payload["train_matches"]:
        row["end_ms"] = None
    payload["diagnostics"].update(valid=0, missing=2)
    payload["diagnostics"].update(found=0, excluded=2)
    payload["missing_policy"] = "exclude"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="no usable|zero usable"):
        resolve_result_end_times(_splits(), artifact_path=path, missing="exclude")


@pytest.mark.parametrize("field,value", [
    ("schema_version", 0), ("timestamp_unit", "s"), ("split_sha256", "0"*64),
    ("enriched_dataset_sha256", "0"*64),
])
def test_artifact_metadata_tampering_is_rejected(tmp_path, field, value):
    path = _artifact(tmp_path)
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        resolve_result_end_times(_splits(), artifact_path=path)


@pytest.mark.parametrize("mutate", [
    lambda p: p["train_matches"][0].update(match_id="other"),
    lambda p: p["train_matches"][0].update(game_creation_time=101),
    lambda p: p["train_matches"][0].update(end_ms=99),
    lambda p: p["train_matches"][0].update(end_ms=float("nan")),
    lambda p: p["train_matches"][0].update(end_ms=True),
    lambda p: p["train_matches"][1].update(match_id="a"),
])
def test_artifact_train_match_tampering_is_rejected(tmp_path, mutate):
    path = _artifact(tmp_path)
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload, allow_nan=True))
    with pytest.raises(ValueError):
        resolve_result_end_times(_splits(), artifact_path=path)


def test_missing_source_and_truncated_json_fail_explicitly(tmp_path):
    with pytest.raises(ValueError, match="db_path|artifact_path"):
        resolve_result_end_times(_splits())
    with pytest.raises(FileNotFoundError):
        resolve_result_end_times(_splits(), artifact_path=tmp_path / "absent.json")
    path = tmp_path / "truncated.json"
    path.write_text('{"schema_version":')
    with pytest.raises(ValueError, match="JSON"):
        resolve_result_end_times(_splits(), artifact_path=path)


def test_database_resolution_uses_explicit_train_ends(tmp_path):
    path = tmp_path / "source.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE matches (match_id TEXT PRIMARY KEY, game_creation_time INTEGER, raw_json TEXT)")
        conn.executemany("INSERT INTO matches VALUES (?,?,?)", [
            ("a", 100, json.dumps({"info": {"gameStartTimestamp": 110, "gameEndTimestamp": 150}})),
            ("b", 200, json.dumps({"info": {"gameStartTimestamp": 210, "gameEndTimestamp": 250}})),
        ])
    ends, provenance = resolve_result_end_times(_splits(), db_path=path)
    assert ends == {"a": 150, "b": 250}
    assert provenance["diagnostics"] == {"requested": 2, "found": 2, "invalid": 0,
                                           "valid": 2, "missing": 0, "excluded": 0}


def test_invalid_database_metadata_preserves_missing_and_invalid_counts(tmp_path):
    path = tmp_path / "source.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE matches (match_id TEXT PRIMARY KEY, game_creation_time INTEGER, raw_json TEXT)")
        conn.execute("INSERT INTO matches VALUES (?,?,?)", ("a", 100, "{}"))
    with pytest.raises(ValueError, match="no usable"):
        resolve_result_end_times(_splits(), db_path=path, missing="exclude")
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE matches SET raw_json=? WHERE match_id='a'",
                     (json.dumps({"info": {"gameStartTimestamp": 110, "gameEndTimestamp": 150}}),))
        conn.execute("INSERT INTO matches VALUES (?,?,?)", ("b", 200, "{}"))
    ends, provenance = resolve_result_end_times(_splits(), db_path=path, missing="exclude")
    assert ends == {"a": 150}
    assert provenance["diagnostics"] == {"requested": 2, "found": 2, "invalid": 1,
                                           "valid": 1, "missing": 0, "excluded": 1}


def test_boolean_schema_and_diagnostics_are_not_accepted_as_integer_counts(tmp_path):
    path = _artifact(tmp_path)
    payload = json.loads(path.read_text())
    payload["schema_version"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema_version"):
        resolve_result_end_times(_splits(), artifact_path=path)
    payload["schema_version"] = 1
    payload["diagnostics"]["valid"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="diagnostics"):
        resolve_result_end_times(_splits(), artifact_path=path)


def test_missing_end_field_is_not_interpreted_as_known_absence(tmp_path):
    path = _artifact(tmp_path)
    payload = json.loads(path.read_text())
    del payload["train_matches"][0]["end_ms"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="end_ms"):
        resolve_result_end_times(_splits(), artifact_path=path)


def test_artifact_writer_refuses_overwrite_and_conflicting_provenance(tmp_path):
    path = _artifact(tmp_path)
    with pytest.raises(FileExistsError):
        write_result_end_times_artifact(path, _splits(), {"a": 150, "b": 250}, {})
    with pytest.raises(ValueError, match="split_sha256"):
        write_result_end_times_artifact(tmp_path / "other.json", _splits(),
            {"a": 150, "b": 250}, {"split_sha256": "0"*64})


def test_replay_validates_source_feature_contract_when_present(tmp_path):
    from core.experiment import (build_ordered_feature_contract, write_feature_contract,
                                 validate_replay_feature_provenance)
    splits = _splits()
    manifest_path = tmp_path / 'model.split.json'
    # Historical A3 manifests have no feature companion and remain usable.
    validate_replay_feature_provenance(manifest_path, splits)
    contract = build_ordered_feature_contract(
        splits, result_end_times_sha256='a'*64, representation='v3',
        matchup_weight=5., missing_policy='error', diagnostics={},
    )
    write_feature_contract(tmp_path / 'model.keras', contract)
    validate_replay_feature_provenance(manifest_path, splits)
    changed = deepcopy(splits)
    changed['train'][0]['feature_json']['team_a'][0]['priors'][0] += .1
    assert compute_dataset_sha256(changed) == compute_dataset_sha256(splits)
    with pytest.raises(ValueError, match='enriched_dataset_sha256'):
        validate_replay_feature_provenance(manifest_path, changed)
    (tmp_path / 'model.features.json').write_text('{', encoding='utf-8')
    with pytest.raises(ValueError, match='JSON'):
        validate_replay_feature_provenance(manifest_path, splits)
