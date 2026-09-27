"""Manifest construction, validation, fingerprints, and frozen replay."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from core.pre_match_dataset import load_records_by_ids, to_pre_match_features

SPLIT_NAMES = ("train", "validation", "test")
FEATURE_POLICY = "draft_context_only_v1; no post-match or undated career statistics"
ORDERED_AVAILABILITY_ASSUMPTION = "explicit_result_end_proxy"


def _validate_splits_structure(splits: dict) -> None:
    if not isinstance(splits, dict) or tuple(splits.keys()) != SPLIT_NAMES:
        raise ValueError(f"splits must contain exactly keys: {SPLIT_NAMES}")
    all_seen_ids = set()
    last_split_max_time = None
    for name in SPLIT_NAMES:
        rows = splits[name]
        if not isinstance(rows, list) or len(rows) == 0:
            raise ValueError(f"split '{name}' must be a non-empty list")
        prev_time = -1
        prev_id = ""
        for r in rows:
            if not isinstance(r, dict):
                raise ValueError(f"split row in '{name}' must be a dict")
            mid = r.get("match_id")
            if not isinstance(mid, str) or not mid.strip():
                raise ValueError(f"record has missing or invalid match_id in '{name}'")
            if mid in all_seen_ids:
                raise ValueError(f"duplicate match_id across splits: {mid}")
            all_seen_ids.add(mid)
            t = r.get("game_creation_time")
            if type(t) is not int or t <= 0:
                raise ValueError(f"record {mid} has non-positive integer game_creation_time")
            if (t, mid) < (prev_time, prev_id):
                raise ValueError(f"split '{name}' is not sorted by (game_creation_time, match_id)")
            prev_time, prev_id = t, mid
        if last_split_max_time is not None:
            if rows[0]["game_creation_time"] <= last_split_max_time:
                raise ValueError(
                    f"split time ranges must be strictly increasing: boundary conflict before '{name}'"
                )
        last_split_max_time = rows[-1]["game_creation_time"]


def compute_dataset_sha256(splits: dict) -> str:
    fingerprint = hashlib.sha256()
    for name in SPLIT_NAMES:
        for record in splits[name]:
            norm_features = to_pre_match_features(record.get("feature_json", {}))
            labels = record.get("labels_json", {})
            payload = {
                "match_id": record["match_id"],
                "game_creation_time": record["game_creation_time"],
                "features": norm_features,
                "labels": labels,
            }
            fingerprint.update(
                json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")
            )
            fingerprint.update(b"\n")
    return fingerprint.hexdigest()


def compute_split_sha256(splits_repr: dict) -> str:
    return hashlib.sha256(
        json.dumps(splits_repr, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _split_membership(splits: dict) -> dict:
    _validate_splits_structure(splits)
    return {
        name: [
            {"match_id": row["match_id"], "game_creation_time": row["game_creation_time"]}
            for row in splits[name]
        ] for name in SPLIT_NAMES
    }


def compute_enriched_dataset_sha256(splits: dict) -> str:
    """Fingerprint full point-in-time rows, including attached priors."""
    _validate_splits_structure(splits)
    payload = {
        name: [
            {key: row[key] for key in (
                "match_id", "game_creation_time", "feature_json", "labels_json")}
            for row in splits[name]
        ] for name in SPLIT_NAMES
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, allow_nan=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _ordered_fingerprints(splits: dict) -> tuple[str, str]:
    return compute_split_sha256(_split_membership(splits)), compute_enriched_dataset_sha256(splits)


def _validate_ends(splits: dict, ends: dict) -> dict:
    if not isinstance(ends, dict):
        raise ValueError("result ends must be a mapping")
    train = {row["match_id"]: row for row in splits["train"]}
    if set(ends) - set(train):
        raise ValueError("result ends contain match_id outside train")
    for mid, end in ends.items():
        creation = train[mid]["game_creation_time"]
        if type(end) is not int or not creation < end <= creation + 24 * 60 * 60 * 1000:
            raise ValueError(f"invalid end_ms for match_id {mid}")
    return train


def _check_missing_policy(splits: dict, ends: dict, missing: str) -> None:
    if missing not in ("error", "exclude"):
        raise ValueError("missing must be 'error' or 'exclude'")
    missing_ids = [r["match_id"] for r in splits["train"] if r["match_id"] not in ends]
    if not ends:
        raise ValueError("no usable result end times for train")
    if missing_ids and missing == "error":
        raise ValueError(
            f"missing result ends for {len(missing_ids)} train matches: "
            + ", ".join(missing_ids[:5])
        )


def _validate_end_diagnostics(diagnostics: dict, requested: int, valid: int) -> None:
    if not isinstance(diagnostics, dict):
        raise ValueError("result end artifact missing diagnostics")
    for key in ("requested", "found", "invalid", "valid", "missing", "excluded"):
        if type(diagnostics.get(key)) is not int or diagnostics[key] < 0:
            raise ValueError(f"result end diagnostics {key} must be a nonnegative integer")
    if (diagnostics["requested"] != requested or diagnostics["valid"] != valid or
            diagnostics["requested"] != diagnostics["found"] + diagnostics["missing"] or
            diagnostics["found"] != diagnostics["valid"] + diagnostics["invalid"] or
            diagnostics["excluded"] != requested - valid):
        raise ValueError("result end diagnostics counts mismatch")


def _validated_end_artifact(data: dict, splits: dict) -> tuple[dict, dict]:
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("unsupported result end artifact schema_version")
    if data.get("timestamp_unit") != "ms":
        raise ValueError("result end artifact timestamp_unit must be ms")
    split_sha, enriched_sha = _ordered_fingerprints(splits)
    for key, expected in (("split_sha256", split_sha),
                          ("enriched_dataset_sha256", enriched_sha),
                          ("availability_assumption", ORDERED_AVAILABILITY_ASSUMPTION)):
        if data.get(key) != expected:
            raise ValueError(f"result end artifact {key} mismatch")
    if data.get("missing_policy") not in ("error", "exclude"):
        raise ValueError("invalid result end artifact missing_policy")
    rows = data.get("train_matches")
    if not isinstance(rows, list) or len(rows) != len(splits["train"]):
        raise ValueError("result end artifact train_matches mismatch")
    ends = {}
    for actual, expected in zip(rows, splits["train"]):
        if not isinstance(actual, dict) or actual.get("match_id") != expected["match_id"]:
            raise ValueError("result end artifact train match_id mismatch or duplicate")
        if type(actual.get("game_creation_time")) is not int or \
                actual["game_creation_time"] != expected["game_creation_time"]:
            raise ValueError("result end artifact game_creation_time mismatch")
        if "end_ms" not in actual:
            raise ValueError("result end artifact missing end_ms")
        end = actual.get("end_ms")
        if end is not None:
            ends[expected["match_id"]] = end
    _validate_ends(splits, ends)
    diagnostics = data.get("diagnostics")
    _validate_end_diagnostics(diagnostics, len(rows), len(ends))
    return ends, diagnostics


def resolve_result_end_times(splits: dict, *, db_path=None, artifact_path=None,
                             missing: str = "error") -> tuple[dict, dict]:
    """Resolve explicit train results and enforce the requested missing policy."""
    if missing not in ("error", "exclude"):
        raise ValueError("missing must be 'error' or 'exclude'")
    split_sha, enriched_sha = _ordered_fingerprints(splits)
    if artifact_path is not None:
        try:
            with Path(artifact_path).open("r", encoding="utf-8") as handle:
                data = json.load(handle, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {x}")))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid result end artifact JSON: {exc}") from exc
        ends, diagnostics = _validated_end_artifact(data, splits)
    elif db_path is not None:
        from core.point_in_time_dataset import load_result_end_times
        ends, diagnostics = load_result_end_times(db_path, splits["train"])
        _validate_ends(splits, ends)
    else:
        raise ValueError("provide artifact_path or db_path for result end times")
    _check_missing_policy(splits, ends, missing)
    diagnostics = dict(diagnostics)
    diagnostics["excluded"] = len(splits["train"]) - len(ends)
    _validate_end_diagnostics(diagnostics, len(splits["train"]), len(ends))
    provenance = {
        "schema_version": 1, "timestamp_unit": "ms", "split_sha256": split_sha,
        "enriched_dataset_sha256": enriched_sha,
        "availability_assumption": ORDERED_AVAILABILITY_ASSUMPTION,
        "missing_policy": missing, "diagnostics": diagnostics,
    }
    return ends, provenance


def write_result_end_times_artifact(path, splits: dict, ends: dict, provenance: dict) -> str:
    """Write a versioned, full-data-bound result-end sidecar exclusively."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be an object")
    split_sha, enriched_sha = _ordered_fingerprints(splits)
    _validate_ends(splits, ends)
    for key, expected in (("split_sha256", split_sha),
                          ("enriched_dataset_sha256", enriched_sha),
                          ("availability_assumption", ORDERED_AVAILABILITY_ASSUMPTION),
                          ("timestamp_unit", "ms"), ("schema_version", 1)):
        if key in provenance and provenance[key] != expected:
            raise ValueError(f"provenance {key} mismatch")
    policy = provenance.get("missing_policy", "error")
    _check_missing_policy(splits, ends, policy)
    diagnostics = dict(provenance.get("diagnostics", {}))
    _validate_end_diagnostics(diagnostics, len(splits["train"]), len(ends))
    data = {
        "schema_version": 1, "timestamp_unit": "ms", "split_sha256": split_sha,
        "enriched_dataset_sha256": enriched_sha,
        "availability_assumption": ORDERED_AVAILABILITY_ASSUMPTION,
        "train_matches": [
            {"match_id": row["match_id"], "game_creation_time": row["game_creation_time"],
             "end_ms": ends.get(row["match_id"])} for row in splits["train"]
        ], "missing_policy": policy, "diagnostics": diagnostics,
    }
    raw = (json.dumps(data, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(raw)
    return hashlib.sha256(raw).hexdigest()


def _validate_feature_contract(contract: dict) -> None:
    if not isinstance(contract, dict):
        raise ValueError("feature contract must be a JSON object")
    required = (
        "schema_version", "policy", "representation", "player_feature_dim",
        "macro_feature_dim", "matchup_weight", "global_weight", "split_sha256",
        "enriched_dataset_sha256", "result_end_times_sha256",
        "availability_assumption", "missing_policy", "diagnostics", "artifact_scope",
        "ordered_matchup_sha256", "champion_registry_sha256",
        "source_sha256",
    )
    for key in required:
        if key not in contract:
            raise ValueError(f"feature contract missing {key}")
    for key, expected in (("schema_version", 1), ("policy", "ordered_matchup_v1"),
                          ("player_feature_dim", 14), ("global_weight", 50.0),
                          ("availability_assumption", ORDERED_AVAILABILITY_ASSUMPTION),
                          ("artifact_scope", "training_evaluation")):
        if type(contract[key]) is bool or contract[key] != expected:
            raise ValueError(f"incompatible feature contract {key}")
    rep = contract["representation"]
    if rep not in ("tensors", "v3") or type(contract["macro_feature_dim"]) is not int or \
            contract["macro_feature_dim"] != (15 if rep == "v3" else 0):
        raise ValueError("incompatible feature contract representation or macro_feature_dim")
    weight = contract["matchup_weight"]
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("incompatible feature contract matchup_weight")
    if contract["missing_policy"] not in ("error", "exclude"):
        raise ValueError("incompatible feature contract missing_policy")
    if not isinstance(contract["diagnostics"], dict):
        raise ValueError("incompatible feature contract diagnostics")
    for key in ("split_sha256", "enriched_dataset_sha256", "result_end_times_sha256",
                "ordered_matchup_sha256", "champion_registry_sha256", "source_cache_sha256"):
        if key in contract and (not isinstance(contract[key], str) or len(contract[key]) != 64 or
                                any(ch not in "0123456789abcdef" for ch in contract[key])):
            raise ValueError(f"incompatible feature contract {key}")
    sources = contract["source_sha256"]
    expected_sources = ("ordered_matchup.py", "pipeline.py", "matchup_matrix.py",
                        "draft_composition.py", "static_meta.py", "champion_registry.py")
    if not isinstance(sources, dict) or set(sources) != set(expected_sources):
        raise ValueError("incompatible feature contract source_sha256")
    for digest in sources.values():
        if not isinstance(digest, str) or len(digest) != 64 or \
                any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("incompatible feature contract source_sha256")


def build_ordered_feature_contract(
    splits: dict, *, result_end_times_sha256: str, representation: str,
    matchup_weight: float, missing_policy: str, diagnostics: dict,
    availability_assumption: str = ORDERED_AVAILABILITY_ASSUMPTION,
    source_cache_sha256: str | None = None,
) -> dict:
    """Build the saved compatibility contract and audit source hashes."""
    split_sha, enriched_sha = _ordered_fingerprints(splits)
    module_root = Path(__file__).resolve().parent
    contract = {
        "schema_version": 1, "policy": "ordered_matchup_v1",
        "representation": representation, "player_feature_dim": 14,
        "macro_feature_dim": 15 if representation == "v3" else 0,
        "matchup_weight": matchup_weight, "global_weight": 50.0,
        "split_sha256": split_sha, "enriched_dataset_sha256": enriched_sha,
        "result_end_times_sha256": result_end_times_sha256,
        "availability_assumption": availability_assumption,
        "missing_policy": missing_policy, "diagnostics": diagnostics,
        "artifact_scope": "training_evaluation",
        "ordered_matchup_sha256": hashlib.sha256(
            (module_root / "features" / "ordered_matchup.py").read_bytes()
        ).hexdigest(),
        "champion_registry_sha256": hashlib.sha256(
            (module_root / "champion_registry.py").read_bytes()
        ).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in {
                **{name: module_root / "features" / name for name in (
                    "ordered_matchup.py", "pipeline.py", "matchup_matrix.py",
                    "draft_composition.py", "static_meta.py")},
                "champion_registry.py": module_root / "champion_registry.py",
            }.items()
        },
    }
    if source_cache_sha256 is not None:
        contract["source_cache_sha256"] = source_cache_sha256
    _validate_feature_contract(contract)
    return contract


def write_feature_contract(checkpoint_path, contract: dict) -> str:
    """Validate and exclusively write the checkpoint's feature sidecar."""
    _validate_feature_contract(contract)
    raw = (json.dumps(contract, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    path = Path(checkpoint_path).with_suffix(".features.json")
    with path.open("xb") as handle:
        handle.write(raw)
    return hashlib.sha256(raw).hexdigest()


def read_feature_contract(checkpoint_path) -> dict:
    """Read a compatible versioned feature contract for an ordered model."""
    path = Path(checkpoint_path).with_suffix(".features.json")
    try:
        with path.open("r", encoding="utf-8") as handle:
            contract = json.load(handle, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {x}")))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid feature contract JSON: {exc}") from exc
    _validate_feature_contract(contract)
    return contract


def build_manifest(splits: dict, *, source_db: str | None, split_metadata: dict) -> dict:
    """Construct and validate a schema 2 experiment manifest."""
    _validate_splits_structure(splits)
    if not isinstance(split_metadata, dict):
        raise ValueError("split_metadata must be a dict")

    splits_repr = {
        name: [
            {"match_id": r["match_id"], "game_creation_time": r["game_creation_time"]}
            for r in splits[name]
        ]
        for name in SPLIT_NAMES
    }
    dataset_sha256 = compute_dataset_sha256(splits)
    split_sha256 = compute_split_sha256(splits_repr)

    return {
        "schema_version": 2,
        "source_db": str(Path(source_db).resolve()) if source_db else None,
        "dataset_sha256": dataset_sha256,
        "feature_policy": FEATURE_POLICY,
        "selection": "frozen match membership; see split_metadata",
        "split_metadata": dict(split_metadata),
        "splits": splits_repr,
        "split_sha256": split_sha256,
    }


def read_manifest(path: str | Path) -> dict:
    """Parse JSON and validate supported manifest schema and structure."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("manifest content must be a JSON object")

    schema_version = data.get("schema_version")
    if schema_version not in (1, 2):
        raise ValueError(f"unsupported manifest schema_version: {schema_version}")

    if data.get("feature_policy") != FEATURE_POLICY:
        raise ValueError(f"unsupported feature_policy: {data.get('feature_policy')}")

    splits = data.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("manifest missing splits object")
    _validate_splits_structure(splits)

    dataset_sha256 = data.get("dataset_sha256")
    if not isinstance(dataset_sha256, str) or len(dataset_sha256) != 64:
        raise ValueError("manifest dataset_sha256 must be a 64-character hex string")

    computed_split_sha = compute_split_sha256(splits)
    if schema_version == 2:
        stored_split_sha = data.get("split_sha256")
        if not isinstance(stored_split_sha, str) or len(stored_split_sha) != 64:
            raise ValueError("schema 2 manifest missing 64-character split_sha256")
        if stored_split_sha != computed_split_sha:
            raise ValueError("manifest split_sha256 verification failed")
        if not isinstance(data.get("split_metadata"), dict):
            raise ValueError("schema 2 manifest missing split_metadata dictionary")
    else:  # schema_version == 1
        data["split_sha256"] = computed_split_sha
        data.setdefault(
            "split_metadata",
            {
                "requested_strategy": "chronological",
                "effective_strategy": "chronological",
                "fallback_reason": None,
            },
        )

    return data


def restore_splits(records: list[dict], manifest: dict) -> dict[str, list[dict]]:
    """Reconstruct exact membership/order and verify time ranges, features, labels and fingerprints."""
    if not isinstance(manifest, dict) or "splits" not in manifest:
        raise ValueError("invalid manifest object")

    manifest_splits = manifest["splits"]
    _validate_splits_structure(manifest_splits)

    seen_ids = set()
    records_by_id: dict[str, dict] = {}
    for r in records:
        if not isinstance(r, dict):
            raise ValueError("source record must be a dict")
        mid = r.get("match_id")
        if not isinstance(mid, str) or not mid.strip():
            raise ValueError("source record has invalid match_id")
        if mid in seen_ids:
            raise ValueError(f"duplicate match_id in source records: {mid}")
        seen_ids.add(mid)
        records_by_id[mid] = r

    restored: dict[str, list[dict]] = {}
    for name in SPLIT_NAMES:
        partition: list[dict] = []
        for expected in manifest_splits[name]:
            mid = expected["match_id"]
            if mid not in records_by_id:
                raise ValueError(f"missing match_id in source records: {mid}")
            rec = records_by_id[mid]
            if rec.get("game_creation_time") != expected["game_creation_time"]:
                raise ValueError(
                    f"timestamp mismatch for match_id {mid}: expected {expected['game_creation_time']}, got {rec.get('game_creation_time')}"
                )
            partition.append(rec)
        restored[name] = partition

    _validate_splits_structure(restored)

    # Verify split_sha256
    restored_splits_repr = {
        name: [
            {"match_id": r["match_id"], "game_creation_time": r["game_creation_time"]}
            for r in restored[name]
        ]
        for name in SPLIT_NAMES
    }
    recomputed_split_sha = compute_split_sha256(restored_splits_repr)
    expected_split_sha = manifest.get("split_sha256")
    if expected_split_sha and recomputed_split_sha != expected_split_sha:
        raise ValueError(f"split_sha256 mismatch: expected {expected_split_sha}, got {recomputed_split_sha}")

    # Verify dataset_sha256
    recomputed_dataset_sha = compute_dataset_sha256(restored)
    if recomputed_dataset_sha != manifest.get("dataset_sha256"):
        raise ValueError(
            f"dataset_sha256 mismatch: expected {manifest.get('dataset_sha256')}, got {recomputed_dataset_sha}"
        )

    return restored


def load_frozen_splits(db_path: str, manifest: dict) -> dict[str, list[dict]]:
    """Call exact-ID loader on database, then restore splits."""
    if not isinstance(manifest, dict) or "splits" not in manifest:
        raise ValueError("invalid manifest object")
    all_ids = [entry["match_id"] for name in SPLIT_NAMES for entry in manifest["splits"][name]]
    records = load_records_by_ids(db_path, all_ids)
    return restore_splits(records, manifest)


def validate_replay_feature_provenance(manifest_path: str | Path, splits: dict) -> None:
    """Verify enriched replay against a source contract, if that run has one.

    Historical manifests without an ordered contract can be certified for the
    first time. A destination may intentionally change representation or weight;
    only the source cohort and complete attached data must remain identical.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.name.endswith('.split.json'):
        return
    checkpoint = manifest_path.with_name(manifest_path.name[:-len('.split.json')] + '.keras')
    if not checkpoint.with_suffix('.features.json').exists():
        return
    contract = read_feature_contract(checkpoint)
    split_sha, enriched_sha = _ordered_fingerprints(splits)
    if contract['split_sha256'] != split_sha:
        raise ValueError('source feature contract split_sha256 mismatch during replay')
    if contract['enriched_dataset_sha256'] != enriched_sha:
        raise ValueError('source feature contract enriched_dataset_sha256 mismatch during replay')
