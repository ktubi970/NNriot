"""Manifest construction, validation, fingerprints, and frozen replay."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from core.pre_match_dataset import load_records_by_ids, to_pre_match_features

SPLIT_NAMES = ("train", "validation", "test")
FEATURE_POLICY = "draft_context_only_v1; no post-match or undated career statistics"


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
