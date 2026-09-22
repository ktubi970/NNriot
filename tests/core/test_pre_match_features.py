import pytest

pytestmark = pytest.mark.unit


def test_extract_pre_match_features_excludes_post_game_leakage():
    from core.json_utils import extract_pre_match_features, extract_match_features

    raw_match = {
        "info": {
            "gameMode": "CLASSIC",
            "gameVersion": "14.10.1.2",
            "participants": [
                {
                    "teamId": 100,
                    "championName": "Aatrox",
                    "teamPosition": "TOP",
                    "kills": 12,
                    "deaths": 2,
                    "assists": 8,
                    "goldEarned": 18500,
                },
                {
                    "teamId": 200,
                    "championName": "Fiora",
                    "teamPosition": "TOP",
                    "kills": 1,
                    "deaths": 10,
                    "assists": 2,
                    "goldEarned": 9200,
                },
            ],
        }
    }
    pre_features = extract_pre_match_features(raw_match, region="EUW")
    
    assert pre_features["game_mode"] == "CLASSIC"
    assert pre_features["region"] == "EUW"
    assert len(pre_features["team_a"]) == 1
    assert len(pre_features["team_b"]) == 1
    
    p_a = pre_features["team_a"][0]
    assert p_a["champion"] == "Aatrox"
    assert p_a["role"] == "TOP"
    
    assert "gold" not in p_a
    assert "kda" not in p_a
    assert "kills" not in p_a
    assert "goldEarned" not in p_a


def test_extract_pre_match_features_accepts_player_career_stats():
    from core.json_utils import extract_pre_match_features, extract_match_features

    raw_match = {
        "info": {
            "gameMode": "CLASSIC",
            "participants": [
                {
                    "teamId": 100,
                    "championName": "Ahri",
                    "teamPosition": "MIDDLE",
                    "career_win_rate": 0.58,
                    "career_matches": 120,
                },
                {
                    "teamId": 200,
                    "championName": "Zed",
                    "teamPosition": "MIDDLE",
                    "career_win_rate": 0.49,
                    "career_matches": 45,
                },
            ],
        }
    }
    feats = extract_pre_match_features(raw_match)
    p_a = feats["team_a"][0]
    assert p_a["champion"] == "Ahri"
    assert p_a["career_win_rate"] == 0.58
    assert p_a["career_matches"] == 120
