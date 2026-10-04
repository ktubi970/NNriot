"""Integration tests for Equivariant DeepSets v4 with timeline player priors."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import tensorflow as tf

from core.champion_registry import NUM_CHAMPIONS
from core.equivariant_model import build_equivariant_deepsets_model
from core.features.draft_composition import MACRO_CLASH_FEATURE_DIM
from core.features.pipeline import ENRICHED_PLAYER_TIMELINE_TENSOR_DIM
from scripts.train_deepsets_v4_timeline import train_deepsets_v4_timeline


def _create_synthetic_batch(
    batch_size: int = 4,
    player_dim: int = 18,
    macro_dim: int = 15,
    seed: int = 42,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Generate valid synthetic inputs for DeepSets v4."""
    rng = np.random.default_rng(seed)
    team_a = np.zeros((batch_size, 5, player_dim), dtype=np.float32)
    team_b = np.zeros((batch_size, 5, player_dim), dtype=np.float32)

    for b in range(batch_size):
        for slot in range(5):
            # Champion ID (integer index in range [1, NUM_CHAMPIONS - 1])
            team_a[b, slot, 0] = float(rng.integers(1, NUM_CHAMPIONS - 1))
            # Canonical role ID (0 to 4)
            team_a[b, slot, 1] = float(slot)
            # 16 continuous features (10 base priors + 4 timeline + matchup delta + base wr)
            team_a[b, slot, 2:] = rng.uniform(-1.0, 1.0, size=player_dim - 2)

            team_b[b, slot, 0] = float(rng.integers(1, NUM_CHAMPIONS - 1))
            team_b[b, slot, 1] = float(slot)
            team_b[b, slot, 2:] = rng.uniform(-1.0, 1.0, size=player_dim - 2)

    macro_clash = rng.uniform(-1.0, 1.0, size=(batch_size, macro_dim)).astype(np.float32)
    return tf.constant(team_a), tf.constant(team_b), tf.constant(macro_clash)


def test_build_equivariant_deepsets_v4_compiles():
    """Verify build_equivariant_deepsets_model compiles with 18-dim player inputs."""
    model = build_equivariant_deepsets_model(
        player_feature_dim=ENRICHED_PLAYER_TIMELINE_TENSOR_DIM,
        include_macro_clash=True,
        macro_feature_dim=MACRO_CLASH_FEATURE_DIM,
        use_cross_attention=False,
    )
    assert model is not None
    input_names = [inp.name for inp in model.inputs]
    assert "team_a" in input_names
    assert "team_b" in input_names
    assert "macro_clash" in input_names
    assert "winner" in model.output_names


def test_forward_pass_and_output_shapes():
    """Verify synthetic batch produces (B, 2) winner probabilities."""
    batch_size = 4
    team_a, team_b, macro_clash = _create_synthetic_batch(
        batch_size=batch_size,
        player_dim=ENRICHED_PLAYER_TIMELINE_TENSOR_DIM,
        macro_dim=MACRO_CLASH_FEATURE_DIM,
    )

    model = build_equivariant_deepsets_model(
        player_feature_dim=ENRICHED_PLAYER_TIMELINE_TENSOR_DIM,
        include_macro_clash=True,
        macro_feature_dim=MACRO_CLASH_FEATURE_DIM,
        use_cross_attention=False,
    )

    preds = model({"team_a": team_a, "team_b": team_b, "macro_clash": macro_clash})
    assert "winner" in preds

    winner_probs = preds["winner"].numpy()
    assert winner_probs.shape == (batch_size, 2)
    # Probabilities must be in [0, 1] and sum to 1.0 per row
    assert np.all(winner_probs >= 0.0)
    assert np.all(winner_probs <= 1.0)
    np.testing.assert_allclose(np.sum(winner_probs, axis=-1), np.ones(batch_size), atol=1e-5)


def test_antisymmetry_under_team_swap():
    """Verify team swap anti-symmetry: P(B | A, B) == 1 - P(B | B, A)."""
    batch_size = 4
    team_a, team_b, macro_clash = _create_synthetic_batch(
        batch_size=batch_size,
        player_dim=ENRICHED_PLAYER_TIMELINE_TENSOR_DIM,
        macro_dim=MACRO_CLASH_FEATURE_DIM,
        seed=123,
    )

    model = build_equivariant_deepsets_model(
        player_feature_dim=ENRICHED_PLAYER_TIMELINE_TENSOR_DIM,
        include_macro_clash=True,
        macro_feature_dim=MACRO_CLASH_FEATURE_DIM,
        use_cross_attention=False,
    )

    # Forward: Team A vs Team B with macro clash diff (B - A)
    preds_ab = model({"team_a": team_a, "team_b": team_b, "macro_clash": macro_clash})
    # Swapped: Team B vs Team A with macro clash diff -(B - A) = (A - B)
    preds_ba = model({"team_a": team_b, "team_b": team_a, "macro_clash": -macro_clash})

    win_ab = preds_ab["winner"].numpy()
    win_ba = preds_ba["winner"].numpy()

    # Column 0 is P(Team A wins), Column 1 is P(Team B wins)
    # Swapping teams: P(Team B wins in AB) == P(Team A wins in BA)
    np.testing.assert_allclose(win_ab[:, 1], win_ba[:, 0], atol=1e-5)
    np.testing.assert_allclose(win_ab[:, 0], win_ba[:, 1], atol=1e-5)
    # Exact anti-symmetry: P(Team B wins in AB) == 1 - P(Team B wins in BA)
    np.testing.assert_allclose(win_ab[:, 1], 1.0 - win_ba[:, 1], atol=1e-5)


def _make_dummy_records(n: int = 30) -> list[dict]:
    """Create dummy records matching load_enriched_records output structure."""
    records = []
    roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    champs_a = ["Darius", "LeeSin", "Ahri", "Jinx", "Thresh"]
    champs_b = ["Malphite", "Amumu", "Syndra", "Caitlyn", "Leona"]

    for i in range(n):
        team_a = []
        team_b = []
        for slot in range(5):
            team_a.append({
                "puuid": f"puuid_a_{slot}_{i}",
                "champion": champs_a[slot],
                "role": roles[slot],
                "priors": [0.5] * 14,
            })
            team_b.append({
                "puuid": f"puuid_b_{slot}_{i}",
                "champion": champs_b[slot],
                "role": roles[slot],
                "priors": [0.5] * 14,
            })

        winner = i % 2
        labels = {
            "winner": winner,
            "team_b_kill_lead": winner,
            "kill_handicap": 2.0 if winner == 0 else -2.0,
            "team_a_kills": 20.0,
            "team_b_kills": 18.0,
            "total_kills": 38.0,
            "first_blood": winner,
        }
        records.append({
            "match_id": f"TEST_MATCH_{i:04d}",
            "game_creation_time": 1700000000 + i * 1800,
            "feature_json": {
                "team_a": team_a,
                "team_b": team_b,
                "game_mode": "CLASSIC",
            },
            "labels_json": labels,
            "winner_label": winner,
        })
    return records


def test_train_deepsets_v4_timeline_pipeline(tmp_path: Path):
    """Test train_deepsets_v4_timeline execution and artifact generation."""
    output_dir = tmp_path / "deepsets_v4_test_run"
    dummy_records = _make_dummy_records(140)

    with patch("scripts.train_deepsets_v4_timeline.load_enriched_records", return_value=dummy_records):
        metrics = train_deepsets_v4_timeline(
            db_path=":memory:",
            output_dir=str(output_dir),
            samples=140,
            warmup=0,
            epochs=1,
            batch_size=32,
            seed=42,
            learning_rate=0.001,
        )

    # Verify generated artifacts
    assert (output_dir / "model.keras").is_file()
    assert (output_dir / "model.calibration.json").is_file()
    assert (output_dir / "model.split.json").is_file()
    assert (output_dir / "model.metrics.json").is_file()

    # Verify metrics content
    assert isinstance(metrics, dict)
    assert metrics["player_feature_dim"] == 18
    assert "test_calibrated" in metrics
    assert "test_raw" in metrics
