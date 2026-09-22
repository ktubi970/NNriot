import pytest

pytestmark = pytest.mark.integration
from core import database
from scripts.backfill_participants import run_backfill

def test_backfill_participants_script(tmp_path):
    db_path = str(tmp_path / "test_backfill.db")
    database.init_db(db_path)

    # Insert sample match
    match_data = {
        "info": {
            "gameMode": "CLASSIC",
            "gameCreation": 1700000000000,
            "gameDuration": 1500,
            "participants": [
                {
                    "puuid": "p-123",
                    "championName": "Jinx",
                    "teamPosition": "BOTTOM",
                    "win": True,
                    "kills": 10, "deaths": 1, "assists": 5,
                    "totalMinionsKilled": 180, "goldEarned": 12000,
                }
            ]
        }
    }
    database.save_match("MATCH_TEST_1", match_data, db_path=db_path)

    with database.get_connection(db_path) as conn:
        conn.execute("DELETE FROM match_participants")

    total = run_backfill(batch_size=50, max_matches=100, db_path=db_path)
    assert total == 1

    history = database.get_player_match_history("p-123", limit=5, db_path=db_path)
    assert len(history) == 1
    assert history[0]["champion"] == "Jinx"
