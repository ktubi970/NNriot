import sqlite3
import json

import pytest

from analytics.extract import ExtractionError, extract_match
from analytics.source import (
    connect_source,
    decode_raw_json,
    iter_source_matches,
    source_fingerprint,
    source_snapshot,
)


@pytest.mark.unit
def test_extracts_one_match_two_teams_ten_players(classic_match):
    classic_match["info"]["participants"][0]["visionScore"] = 15.5
    result = extract_match("EUW1_1", classic_match)

    assert result.match["region"] == "EUW"
    assert result.match["patch"] == "16.14"
    assert len(result.teams) == 2
    assert len(result.players) == 10
    assert sum(row["win"] for row in result.teams) == 1
    assert result.players[0]["vision_score"] == 15.5
    assert result.teams[0]["gold"] == sum(
        player["gold"] for player in result.players if player["team_id"] == 100
    )


@pytest.mark.unit
def test_rejects_non_classic_without_partial_rows(classic_match):
    classic_match["info"]["gameMode"] = "ARAM"

    with pytest.raises(ExtractionError, match="UNSUPPORTED_GAME_MODE") as raised:
        extract_match("EUW1_2", classic_match)

    assert raised.value.code == "UNSUPPORTED_GAME_MODE"
    assert raised.value.field == "info.gameMode"


@pytest.mark.unit
def test_rejects_duplicate_puuid(classic_match):
    classic_match["info"]["participants"][1]["puuid"] = classic_match["info"][
        "participants"
    ][0]["puuid"]

    with pytest.raises(ExtractionError) as raised:
        extract_match("EUW1_3", classic_match)

    assert raised.value.code == "DUPLICATE_PARTICIPANT"


@pytest.mark.integration
def test_source_access_is_query_only_and_decodes_matches(source_db):
    rows = list(iter_source_matches(str(source_db)))
    assert len(rows) == 1
    assert rows[0][0] == "EUW1_1"
    assert rows[0][1]["info"]["gameMode"] == "CLASSIC"

    with connect_source(str(source_db)) as conn:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM matches")


@pytest.mark.integration
def test_source_tokens_are_deterministic_and_snapshot_is_cheap(source_db):
    first = source_fingerprint(str(source_db), "extractor-v1")
    second = source_fingerprint(str(source_db), "extractor-v1")

    assert first == second
    assert first["source_schema_version"] == "11"
    assert first["source_match_count"] == "1"
    assert first["source_max_creation_ms"] == "1"
    assert source_snapshot(str(source_db)) == {
        key: first[key]
        for key in (
            "source_schema_version",
            "source_match_count",
            "source_max_creation_ms",
        )
    }


@pytest.mark.unit
def test_decoder_accepts_legacy_plain_json():
    payload = {"info": {"gameMode": "CLASSIC"}}
    assert decode_raw_json(json.dumps(payload)) == payload
