import os
import pytest

pytestmark = [pytest.mark.ml, pytest.mark.slow]

from core.feature_labels import VECTOR_DIM


@pytest.fixture(scope="module")
def graph_module():
    from core import generate_graph

    return generate_graph


@pytest.mark.integration
def test_model_builds_with_production_dim(graph_module):
    m = graph_module.build_multi_output_model(input_dim=VECTOR_DIM)
    assert m.input_shape[-1] == 20000
    assert "winner" in m.output_names


@pytest.mark.unit
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
