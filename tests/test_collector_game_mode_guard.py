from scripts.data_collector import _build_training_record


def test_build_training_record_rejects_all_non_classic():
    # Valid CLASSIC payload stub
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
