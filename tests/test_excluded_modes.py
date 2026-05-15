"""Test that EXCLUDED_GAME_MODES filter rejects malformed-mode matches."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collector import _build_training_record, EXCLUDED_GAME_MODES


def _classic_match(team_a_win=True):
    return {
        "info": {
            "gameMode": "CLASSIC",
            "gameVersion": "14.10.400",
            "participants": [
                {"championName": "Aatrox", "teamId": 100, "kills": 1, "deaths": 1, "assists": 1, "goldEarned": 1000, "teamPosition": "TOP"},
                {"championName": "Lux", "teamId": 200, "kills": 1, "deaths": 1, "assists": 1, "goldEarned": 1000, "teamPosition": "MID"},
            ],
            "teams": [
                {"teamId": 100, "win": team_a_win, "objectives": {}},
                {"teamId": 200, "win": not team_a_win, "objectives": {}},
            ],
        }
    }


def test_classic_match_accepted():
    """CLASSIC mode produces a valid training record."""
    rec = _build_training_record("M1", _classic_match())
    assert rec is not None
    assert rec[0] == "M1"


def test_cherry_mode_rejected():
    """CHERRY mode (Arena) is rejected at _build_training_record."""
    match = _classic_match()
    match["info"]["gameMode"] = "CHERRY"
    rec = _build_training_record("M_arena", match)
    assert rec is None


def test_strawberry_mode_rejected():
    """STRAWBERRY mode (Swarm) is rejected."""
    match = _classic_match()
    match["info"]["gameMode"] = "STRAWBERRY"
    rec = _build_training_record("M_swarm", match)
    assert rec is None


def test_constant_includes_expected_modes():
    """EXCLUDED_GAME_MODES contains CHERRY and STRAWBERRY at minimum."""
    assert "CHERRY" in EXCLUDED_GAME_MODES
    assert "STRAWBERRY" in EXCLUDED_GAME_MODES
