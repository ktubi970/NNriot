"""Preparation and provenance for result-ordered neural training runs."""
from __future__ import annotations

from pathlib import Path

from core.experiment import (
    build_ordered_feature_contract,
    resolve_result_end_times,
    write_feature_contract,
    write_result_end_times_artifact,
)
from core.features.ordered_matchup import extract_ordered_features
from core.calibration import sha256_file


def prepare_ordered_training(
    splits: dict,
    checkpoint_path: str | Path,
    *,
    db_path: str | None,
    result_end_times: str | None,
    missing_result_end: str,
    representation: str,
    matchup_weight: float,
    source_cache: str | None = None,
    resolved: tuple[dict, dict] | None = None,
) -> tuple[dict, dict, str]:
    """Validate input facts, extract features, and record this run's contract."""
    checkpoint = Path(checkpoint_path)
    ends, provenance = resolved if resolved is not None else resolve_result_end_times(
        splits, db_path=db_path, artifact_path=result_end_times,
        missing=missing_result_end,
    )
    x, _, diagnostics = extract_ordered_features(
        splits, ends, representation=representation,
        matchup_weight=matchup_weight,
    )
    end_sha = write_result_end_times_artifact(
        checkpoint.with_suffix(".result_end_times.json"), splits, ends, provenance,
    )
    contract = build_ordered_feature_contract(
        splits, result_end_times_sha256=end_sha,
        representation=representation, matchup_weight=matchup_weight,
        missing_policy=missing_result_end, diagnostics=diagnostics,
        availability_assumption=provenance["availability_assumption"],
        source_cache_sha256=sha256_file(source_cache) if source_cache else None,
    )
    contract_sha = write_feature_contract(checkpoint, contract)
    return x, diagnostics, contract_sha
