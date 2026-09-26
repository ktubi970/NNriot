import os
import pytest
from core.generate_graph import build_multi_output_model
from core.feature_labels import VECTOR_DIM


def test_model_builds_with_production_dim():
    m = build_multi_output_model(input_dim=VECTOR_DIM)
    assert m.input_shape[-1] == 20000
    assert "winner" in m.output_names


def test_to_pre_match_features_strips_leakage():
    from scripts.train_pre_match_model import to_pre_match_features

    sample_feat = {
        "team_a": [{"champion": "Ahri", "role": "MID", "kda": 5.0, "gold": 12000}],
        "team_b": [{"champion": "Zed", "role": "MID", "kda": 2.0, "gold": 9000}],
        "game_mode": "CLASSIC",
        "game_version": "14.1.1",
        "season_id": 14,
        "region": "EUW",
    }
    cleaned = to_pre_match_features(sample_feat)
    assert cleaned["team_a"][0] == {"champion": "Ahri", "role": "MID"}
    assert cleaned["team_b"][0] == {"champion": "Zed", "role": "MID"}
    assert "kda" not in cleaned["team_a"][0]
    assert "gold" not in cleaned["team_a"][0]
    assert cleaned["game_mode"] == "CLASSIC"
    assert cleaned["region"] == "EUW"


def test_equivariant_deepsets_training_run(tmp_path):
    from scripts.train_pre_match_model import train_model

    # Build 25 synthetic matches for chronological split (min 20 required)
    records = []
    for i in range(25):
        records.append({
            "match_id": f"TEST_{i}",
            "game_creation_time": 1000 + i * 100,
            "winner_label": i % 2,
            "feature_json": {
                "team_a": [{"champion": "Ahri", "role": "MIDDLE"}] * 5,
                "team_b": [{"champion": "Zed", "role": "MIDDLE"}] * 5,
            },
            "labels_json": {
                "winner": i % 2,
                "team_b_kill_lead": i % 2,
                "kills_odd": 1,
                "both_baron": 0,
                "both_inhibitor": 0,
                "both_dragon": 1,
                "elder_dragon": 0,
                "first_blood": 0,
                "first_baron": 0,
                "first_inhibitor": 0,
                "first_tower": 1,
                "kill_handicap": 2,
                "total_kills": 30,
                "team_a_kills": 16,
                "team_b_kills": 14,
                "total_barons": 1,
                "total_dragons": 3,
                "total_towers": 11,
            },
        })

    ckpt_path = str(tmp_path / "model.keras")
    model_report = train_model(
        records,
        epochs=1,
        batch_size=8,
        checkpoint_path=ckpt_path,
        feature_version="deepsets",
        use_pcgrad=True,
    )

    report = model_report.report
    assert report["symmetry_diagnostics"] is not None
    assert report["symmetry_diagnostics"]["is_equivariant"] is True
    assert report["symmetry_diagnostics"]["anti_symmetry_guarantee"] is True
    assert report["symmetry_diagnostics"]["max_symmetry_violation"] < 1e-4
    assert os.path.exists(ckpt_path)

    # Verify selective metrics in report structure and serialized artifact
    assert "selective_metrics" in report["metrics"]["validation"]["model"]["diagnostics"]
    assert "selective_metrics" in report["metrics"]["test"]["model"]["diagnostics"]
    val_selective = report["metrics"]["validation"]["model"]["diagnostics"]["selective_metrics"]
    test_selective = report["metrics"]["test"]["model"]["diagnostics"]["selective_metrics"]
    assert "conviction_ge_4pct" in val_selective
    assert "conviction_ge_4pct" in test_selective
    assert "conviction_ge_0pct" in test_selective
    assert test_selective["conviction_ge_0pct"]["coverage"] == 1.0

    metrics_json_path = ckpt_path.replace(".keras", ".metrics.json")
    assert os.path.exists(metrics_json_path)
    import json
    with open(metrics_json_path, "r", encoding="utf-8") as f:
        saved_report = json.load(f)
    assert "selective_metrics" in saved_report["metrics"]["test"]["model"]["diagnostics"]
    assert "conviction_ge_4pct" in saved_report["metrics"]["test"]["model"]["diagnostics"]["selective_metrics"]


def test_evaluation_includes_selective_metrics():
    from core.training_metrics import evaluate_winner
    import numpy as np

    y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    y_prob = np.array([0.65, 0.35, 0.52, 0.48, 0.70, 0.60, 0.40, 0.30])

    report = evaluate_winner(y_true, y_prob, "winner")
    assert "selective_metrics" in report["diagnostics"]
    assert "conviction_ge_4pct" in report["diagnostics"]["selective_metrics"]
    assert report["diagnostics"]["selective_metrics"]["conviction_ge_4pct"]["accuracy"] >= 0.60



