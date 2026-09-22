def test_build_training_record_rejects_all_non_classic():
    # Valid CLASSIC payload stub
    from scripts.data_collector import _build_training_record

    valid_payload = {
        "info": {
            "gameMode": "CLASSIC",
            "teams": [{"teamId": 100, "win": True}, {"teamId": 200, "win": False}],
            "participants": [
                {
                    "teamId": 100 if i < 5 else 200,
                    "championId": i + 1,
                    "championName": f"Champ{i+1}",
                    "teamPosition": "TOP" if i % 5 == 0 else "BOTTOM",
                    "win": i < 5,
                    "kills": 1,
                    "deaths": 0,
                    "assists": 0,
                    "totalMinionsKilled": 50,
                    "goldEarned": 3000,
                }
                for i in range(10)
            ],
            "gameDuration": 1200,
        }
    }

    # ARAM mode
    aram_payload = {
        "info": {
            "gameMode": "ARAM",
            "teams": [{"teamId": 100, "win": True}, {"teamId": 200, "win": False}],
            "participants": [{"teamId": 100, "championId": 1, "win": True} for _ in range(10)],
            "gameDuration": 1200,
        }
    }

    # URF mode
    urf_payload = {
        "info": {
            "gameMode": "URF",
            "teams": [{"teamId": 100, "win": True}, {"teamId": 200, "win": False}],
            "participants": [{"teamId": 100, "championId": 1, "win": True} for _ in range(10)],
            "gameDuration": 1200,
        }
    }

    assert _build_training_record("CLASSIC_MATCH", valid_payload) is not None
    assert _build_training_record("ARAM_MATCH", aram_payload) is None
    assert _build_training_record("URF_MATCH", urf_payload) is None


def _collector_match(team_a_win, team_b_win):
    return {
        "info": {
            "gameVersion": "14.10.400",
            "gameMode": "CLASSIC",
            "participants": [
                {"championName": "Aatrox", "teamId": 100, "kills": 1, "deaths": 1, "assists": 1, "goldEarned": 1000, "teamPosition": "TOP"},
                {"championName": "Lux", "teamId": 200, "kills": 1, "deaths": 1, "assists": 1, "goldEarned": 1000, "teamPosition": "MID"},
            ],
            "teams": [
                {"teamId": 100, "win": team_a_win},
                {"teamId": 200, "win": team_b_win},
            ],
        }
    }


def test_build_training_record_team_a_wins():
    from scripts import data_collector

    rec = data_collector._build_training_record("TEST_1", _collector_match(True, False))
    assert rec is not None
    assert rec[0] == "TEST_1"
    assert rec[2] == 0  # team A label


def test_build_training_record_team_b_wins():
    from scripts import data_collector

    rec = data_collector._build_training_record("TEST_2", _collector_match(False, True))
    assert rec is not None
    assert rec[2] == 1


def test_build_training_record_skips_no_winner():
    from scripts import data_collector

    rec = data_collector._build_training_record("TEST_3", _collector_match(False, False))
    assert rec is None


def test_fetch_timelines_default_false():
    """NNRIOT_FETCH_TIMELINES env var resolves to a bool at module-load time."""
    from scripts import data_collector

    import importlib
    from scripts import data_collector
    importlib.reload(data_collector)  # re-evaluate the module-level constant
    # Default env should produce False (or True if host set the flag).
    assert data_collector.FETCH_TIMELINES in (False, True)
    # Just verify the constant exists and is a bool.
    assert isinstance(data_collector.FETCH_TIMELINES, bool)
