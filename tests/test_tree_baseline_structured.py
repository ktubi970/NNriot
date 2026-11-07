import json
from pathlib import Path
import joblib
import numpy as np
import pytest

from core.draft_features import TABULAR_FEATURE_DIM
from core.experiment import build_manifest
from core.pre_match_dataset import chronological_split
from scripts.train_tree_baseline import train_tree_model


def make_synthetic_record(mid: int, winner: int, a_top: str = "Aatrox", b_top: str = "Darius"):
    return {
        "match_id": f"TEST_{mid:04d}",
        "game_creation_time": 1700000000 + mid * 100,
        "feature_json": {
            "team_a": [
                {"champion": a_top, "role": "TOP"},
                {"champion": "LeeSin", "role": "JUNGLE"},
                {"champion": "Ahri", "role": "MIDDLE"},
                {"champion": "Jinx", "role": "BOTTOM"},
                {"champion": "Thresh", "role": "UTILITY"},
            ],
            "team_b": [
                {"champion": b_top, "role": "TOP"},
                {"champion": "Elise", "role": "JUNGLE"},
                {"champion": "Zed", "role": "MIDDLE"},
                {"champion": "Ezreal", "role": "BOTTOM"},
                {"champion": "Lulu", "role": "UTILITY"},
            ],
            "game_mode": "CLASSIC",
        },
        "labels_json": {"winner": winner},
        "winner_label": winner,
    }


def test_tree_model_trains_with_structured_features(tmp_path):
    records = []
    # Create 60 records (42 train, 9 val, 9 test) with synthetic bias
    for i in range(60):
        # Bias: If Darius is on team B, winner is mostly 1
        b_top = "Darius" if i % 2 == 0 else "Garen"
        winner = 1 if b_top == "Darius" else 0
        records.append(make_synthetic_record(i, winner, a_top="Aatrox", b_top=b_top))

    out_dir = str(tmp_path / "tree_run")
    report = train_tree_model(records, output_dir=out_dir, split_strategy="chronological")

    assert "model" in report["metrics"]["train"]
    assert "test" in report["metrics"]
    assert (tmp_path / "tree_run" / "model.joblib").exists()
    assert (tmp_path / "tree_run" / "model.metrics.json").exists()

    # Verify structured features and bundle contract
    bundle = joblib.load(tmp_path / "tree_run" / "model.joblib")
    assert bundle["encoder"] is None
    assert len(bundle["feature_names"]) == TABULAR_FEATURE_DIM
    assert bundle["feature_names"][0] == "team_a_TOP"
    assert report["feature_names"] == bundle["feature_names"]

    # Model inference verification with structured features
    clf = bundle["model"]
    dummy_input = np.zeros((1, TABULAR_FEATURE_DIM), dtype=np.float32)
    probs = clf.predict_proba(dummy_input)
    assert probs.shape == (1, 2)
    assert np.isclose(probs.sum(), 1.0)


def test_tree_model_rejects_single_class_train(tmp_path):
    records = [make_synthetic_record(i, winner=0) for i in range(60)]
    out_dir = str(tmp_path / "tree_single_class")
    with pytest.raises(ValueError, match="train partition must contain both classes"):
        train_tree_model(records, output_dir=out_dir)


def test_tree_model_refuses_to_overwrite_existing_artifacts(tmp_path):
    records = [make_synthetic_record(i, winner=i % 2) for i in range(60)]
    out_dir = tmp_path / "existing_artifacts"
    out_dir.mkdir(parents=True)
    (out_dir / "model.joblib").write_bytes(b"dummy")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        train_tree_model(records, output_dir=str(out_dir))


def test_tree_model_comparison_reporting(tmp_path):
    records = [make_synthetic_record(i, winner=i % 2) for i in range(60)]
    splits = chronological_split(records)
    manifest = build_manifest(splits, source_db=None, split_metadata={
        "requested_strategy": "chronological",
        "effective_strategy": "chronological",
        "fallback_reason": None,
    })

    fake_neural_report = {
        "dataset_sha256": manifest["dataset_sha256"],
        "split_sha256": manifest["split_sha256"],
        "train_positive_rate": 0.5,
        "split_counts": {"train": 42, "validation": 9, "test": 9},
        "metrics": {
            "test": {
                "baseline": {"accuracy": 0.5, "log_loss": 0.693, "brier": 0.25},
                "model": {"accuracy": 0.55, "log_loss": 0.680, "brier": 0.24},
            }
        },
    }

    out_dir = str(tmp_path / "tree_compare")
    report = train_tree_model(
        records,
        output_dir=out_dir,
        split_manifest=manifest,
        compare_report=fake_neural_report,
    )

    comparison_path = tmp_path / "tree_compare" / "comparison.json"
    assert comparison_path.exists()
    with comparison_path.open("r", encoding="utf-8") as f:
        comp_data = json.load(f)
    assert comp_data["test"]["neural_raw"]["accuracy"] == 0.55
    assert "tree_raw" in comp_data["test"]


def test_tree_model_handles_missing_or_messy_participant_fields(tmp_path):
    records = []
    for i in range(60):
        rec = make_synthetic_record(i, winner=i % 2)
        # Introduce role aliases, lowercase, or missing entries
        if i % 3 == 0:
            rec["feature_json"]["team_a"][0]["role"] = "MID"  # alias
            rec["feature_json"]["team_b"][0]["role"] = "SUP"  # alias
        elif i % 3 == 1:
            rec["feature_json"]["team_a"] = rec["feature_json"]["team_a"][:4]  # 4 players
        records.append(rec)

    out_dir = str(tmp_path / "tree_messy")
    report = train_tree_model(records, output_dir=out_dir)
    assert (tmp_path / "tree_messy" / "model.joblib").exists()
    assert report["metrics"]["train"]["model"]["log_loss"] > 0
