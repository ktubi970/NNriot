import pytest
from scripts.train_pre_match_model import train_model
from core.generate_graph import build_draft_dense_model
from core.draft_features import DENSE_FEATURE_DIM


def make_sample_record(mid: int, winner: int):
    return {
        "match_id": f"DRAFT_NN_{mid}",
        "game_creation_time": 1700000000 + mid * 60,
        "feature_json": {
            "team_a": [
                {"champion": "Aatrox", "role": "TOP"},
                {"champion": "LeeSin", "role": "JUNGLE"},
                {"champion": "Ahri", "role": "MIDDLE"},
                {"champion": "Jinx", "role": "BOTTOM"},
                {"champion": "Thresh", "role": "UTILITY"},
            ],
            "team_b": [
                {"champion": "Darius" if winner == 1 else "Garen", "role": "TOP"},
                {"champion": "Elise", "role": "JUNGLE"},
                {"champion": "Zed", "role": "MIDDLE"},
                {"champion": "Ezreal", "role": "BOTTOM"},
                {"champion": "Lulu", "role": "UTILITY"},
            ],
            "game_mode": "CLASSIC",
        },
        "labels_json": {
            "winner": winner,
            "team_b_kill_lead": winner,
            "first_blood": 0, "first_baron": 2, "first_inhibitor": 2, "first_tower": 0,
            "kills_odd": 0, "both_baron": 0, "both_inhibitor": 0, "both_dragon": 1,
            "elder_dragon": 0, "kill_handicap": 0, "total_kills": 20,
            "team_a_kills": 10, "team_b_kills": 10, "total_barons": 0,
            "total_dragons": 1, "total_towers": 2,
        },
        "winner_label": winner,
    }


def test_build_draft_dense_model_architecture():
    model = build_draft_dense_model(input_dim=DENSE_FEATURE_DIM)
    assert model.input_shape == (None, DENSE_FEATURE_DIM)
    assert list(model.output_names) == ["winner"]
    assert tuple(model.outputs[0].shape) == (None, 2)


def test_train_model_invalid_feature_version():
    records = [make_sample_record(i, winner=i % 2) for i in range(10)]
    with pytest.raises(ValueError, match="feature_version"):
        train_model(records, epochs=1, feature_version="invalid_version")


def test_train_model_with_v2_draft_features(tmp_path):
    records = [make_sample_record(i, winner=i % 2) for i in range(40)]
    ckpt = str(tmp_path / "candidate_model.keras")
    model, report = train_model(
        records,
        epochs=1,
        batch_size=16,
        checkpoint_path=ckpt,
        feature_version="v2_draft",
    )
    assert model is not None
    assert (tmp_path / "candidate_model.keras").exists()
    assert (tmp_path / "candidate_model.metrics.json").exists()
    assert report["parameters"]["feature_version"] == "v2_draft"
    assert model.input_shape[-1] == DENSE_FEATURE_DIM


def test_model_with_report_proxy():
    from scripts.train_pre_match_model import ModelWithReport
    class DummyModel:
        def __init__(self):
            self.input_shape = (None, 1881)
            self.custom_attr = "val"
        def get_weights(self):
            return [1, 2, 3]

    m = DummyModel()
    r = {"metric": 0.5}
    wrapped = ModelWithReport(m, r)

    unpacked_m, unpacked_r = wrapped
    assert unpacked_m is m
    assert unpacked_r is r
    assert wrapped.input_shape == (None, 1881)
    assert wrapped.custom_attr == "val"
    assert wrapped.get_weights() == [1, 2, 3]
    assert wrapped.model is m
    assert wrapped.report is r

    wrapped.new_attr = 42
    assert m.new_attr == 42

    with pytest.raises(AttributeError):
        wrapped.model = "override"
    with pytest.raises(AttributeError):
        wrapped.report = "override"
    with pytest.raises(AttributeError):
        wrapped.count = 99


def test_get_feat_unparsed_json():
    import json
    from scripts.train_pre_match_model import _get_feat

    direct_dict = {
        "team_a": [{"champion": "Ahri", "role": "MIDDLE"}],
        "team_b": [{"champion": "Zed", "role": "MIDDLE"}],
    }
    assert _get_feat({"feature_json": direct_dict}) == direct_dict

    json_str = json.dumps(direct_dict)
    assert _get_feat({"feature_json": json_str}) == direct_dict

    legacy_match = {
        "info": {
            "participants": [
                {"championName": "Ahri", "teamPosition": "MIDDLE", "teamId": 100},
                {"championName": "Zed", "teamPosition": "MIDDLE", "teamId": 200},
            ]
        }
    }
    legacy_res = _get_feat({"feature_json": json.dumps(legacy_match)})
    assert isinstance(legacy_res, dict)
    assert "team_a" in legacy_res and "team_b" in legacy_res
    assert legacy_res["team_a"][0]["champion"] == "Ahri"
    assert legacy_res["team_b"][0]["champion"] == "Zed"

    malformed_res = _get_feat({"feature_json": "not_valid_json{"})
    assert isinstance(malformed_res, dict)
    assert "team_a" in malformed_res and "team_b" in malformed_res

    none_res = _get_feat({"feature_json": None})
    assert isinstance(none_res, dict)
    assert "team_a" in none_res and "team_b" in none_res

