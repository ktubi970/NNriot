"""Read-only dataset and temporal-holdout contracts, using temporary SQLite files."""

import base64
import copy
import gzip
import importlib
import json
import logging
import sqlite3

import pytest


def _api():
    return importlib.import_module("core.pre_match_dataset")


def _labels():
    return {
        "winner": 0, "team_b_kill_lead": 0, "kill_handicap": 5,
        "total_kills": 25, "team_a_kills": 15, "team_b_kills": 10,
        "kills_odd": 1, "first_blood": 0, "first_baron": 2,
        "first_inhibitor": 0, "first_tower": 1, "total_barons": 0,
        "total_dragons": 4, "total_towers": 10, "both_baron": 0,
        "both_inhibitor": 0, "both_dragon": 1, "elder_dragon": 0,
    }


def _record(index, timestamp=None):
    return {
        "id": index, "match_id": f"EUW1_{index:04d}",
        "game_creation_time": timestamp if timestamp is not None else index * 1000,
        "winner_label": 0, "labels_json": _labels(),
        "feature_json": {
            "game_mode": "CLASSIC", "game_version": "16.18.1", "season_id": 16,
            "team_a": [{"champion": "Ahri", "role": "MIDDLE"} for _ in range(5)],
            "team_b": [{"champion": "Zed", "role": "MIDDLE"} for _ in range(5)],
        },
    }


def _encode(value, compressed):
    if isinstance(value, str):
        return value
    text = json.dumps(value)
    if compressed:
        return base64.b64encode(gzip.compress(text.encode())).decode()
    return text


def _database(path, records, compressed=True):
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE matches (
                match_id TEXT PRIMARY KEY, game_mode TEXT, game_creation_time INTEGER
            );
            CREATE TABLE training_dataset (
                id INTEGER PRIMARY KEY, match_id TEXT UNIQUE, feature_json TEXT,
                winner_label INTEGER, labels_json TEXT
            );
        """)
        for r in records:
            conn.execute("INSERT INTO matches VALUES (?,?,?)", (
                r["match_id"], r.get("game_mode", "CLASSIC"), r["game_creation_time"],
            ))
            conn.execute("INSERT INTO training_dataset VALUES (?,?,?,?,?)", (
                r["id"], r["match_id"], _encode(r["feature_json"], compressed),
                r["winner_label"], _encode(r["labels_json"], compressed),
            ))


@pytest.mark.parametrize("compressed", [False, True])
def test_load_selects_recent_match_times_not_ingestion_order_without_writing(tmp_path, compressed):
    records = [_record(i, (31 - i) * 1000) for i in range(1, 31)]
    path = tmp_path / "source #1.db"
    _database(path, records, compressed)
    before = path.read_bytes()

    result = _api().load_records(str(path), limit=20)

    assert [r["id"] for r in result] == list(range(20, 0, -1))
    assert result[0] == records[19]
    assert result[-1] == records[0]
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["source #1.db"]


def test_missing_database_is_not_created(tmp_path):
    path = tmp_path / "missing.db"
    with pytest.raises((FileNotFoundError, sqlite3.OperationalError)):
        _api().load_records(str(path))
    assert not path.exists()


def test_load_honors_candidate_limit_when_latest_record_is_invalid(tmp_path):
    records = [_record(1), _record(2)]
    records[-1]["labels_json"].pop("winner")
    path = tmp_path / "bounded.db"
    _database(path, records)
    assert _api().load_records(str(path), limit=1) == []


def test_load_breaks_selection_ties_by_match_id(tmp_path):
    records = [_record(i, 1000) for i in range(1, 5)]
    path = tmp_path / "ties.db"
    _database(path, records)
    assert [r["match_id"] for r in _api().load_records(str(path), limit=2)] == [
        "EUW1_0003", "EUW1_0004",
    ]


def test_load_skips_corrupt_gzip_payload_without_crashing(tmp_path):
    record = _record(1)
    # Valid gzip header followed by an invalid deflate block (reserved type 3).
    damaged = bytes.fromhex("1f8b0800000000000000ff0000000000000000")
    record["feature_json"] = base64.b64encode(damaged).decode()
    path = tmp_path / "corrupt.db"
    _database(path, [record])
    assert _api().load_records(str(path)) == []


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_load_rejects_invalid_limit_before_opening_database(tmp_path, limit):
    with pytest.raises(ValueError, match="limit"):
        _api().load_records(str(tmp_path / "missing.db"), limit=limit)


def test_load_excludes_invalid_mode_time_features_and_labels(tmp_path, caplog):
    records = [_record(i) for i in range(1, 16)]
    records[1]["game_mode"] = "ARAM"
    records[2]["game_creation_time"] = 0
    records[3]["game_creation_time"] = -100
    records[4]["labels_json"].pop("total_kills")
    records[5]["feature_json"]["team_a"].pop()
    records[6]["feature_json"] = "corrupt gzip"
    records[7]["labels_json"] = "corrupt json"
    records[8]["winner_label"] = 1  # Conflicts with the winner target.
    records[9]["feature_json"]["team_b"][0]["champion"] = ""
    records[10]["labels_json"] = []
    records[11]["feature_json"]["team_a"][0] = None
    records[12]["game_creation_time"] = "not a date"
    records[13]["feature_json"]["team_b"] = {"champion": "Ahri"}
    records[14]["labels_json"]["winner"] = 2
    path = tmp_path / "invalid.db"
    _database(path, records)

    with caplog.at_level(logging.INFO):
        result = _api().load_records(str(path), limit=50)

    assert [r["match_id"] for r in result] == ["EUW1_0001"]
    assert any("valid" in message.lower() for message in caplog.messages)


@pytest.mark.parametrize("key,value", [
    ("winner", -1), ("team_b_kill_lead", 2), ("first_blood", 3),
    ("first_baron", 1.5), ("first_tower", "1"), ("kills_odd", 2),
    ("both_dragon", -1), ("elder_dragon", None),
    ("total_kills", float("nan")), ("kill_handicap", float("inf")),
    ("team_a_kills", -1), ("total_dragons", "4"),
    ("total_towers", 1e100), ("first_to_5_kills", 9),
])
def test_load_rejects_unsafe_label_values(tmp_path, key, value):
    record = _record(1)
    record["labels_json"][key] = value
    path = tmp_path / "invalid-label.db"
    _database(path, [record])
    assert _api().load_records(str(path)) == []


def test_split_uses_time_and_preserves_every_record_unchanged():
    records = [_record(i, (101 - i) * 1000) for i in range(1, 101)]
    original = copy.deepcopy(records)

    splits = _api().chronological_split(records)

    assert list(splits) == ["train", "validation", "test"]
    assert [len(v) for v in splits.values()] == [70, 15, 15]
    assert [r["id"] for r in splits["train"]] == list(range(100, 30, -1))
    assert [r["id"] for r in splits["validation"]] == list(range(30, 15, -1))
    assert [r["id"] for r in splits["test"]] == list(range(15, 0, -1))
    combined = [r for partition in splits.values() for r in partition]
    assert len({r["match_id"] for r in combined}) == 100
    assert {id(r) for r in combined} == {id(r) for r in records}
    assert records == original


def test_split_keeps_equal_timestamps_together_and_orders_ids_deterministically():
    records = [_record(i, ((i - 1) // 10 + 1) * 1000) for i in range(1, 101)]
    splits = _api().chronological_split(list(reversed(records)))
    assert [len(v) for v in splits.values()] == [70, 10, 20]
    combined = [r for partition in splits.values() for r in partition]
    assert [r["id"] for r in combined] == list(range(1, 101))
    assert splits["train"][-1]["game_creation_time"] < splits["validation"][0]["game_creation_time"]
    assert splits["validation"][-1]["game_creation_time"] < splits["test"][0]["game_creation_time"]


@pytest.mark.parametrize("timestamp", [0, -1, None, "1000", float("nan"), True])
def test_split_rejects_invalid_timestamps(timestamp):
    records = [_record(i) for i in range(1, 21)]
    records[4]["game_creation_time"] = timestamp
    with pytest.raises(ValueError, match="timestamp"):
        _api().chronological_split(records)


def test_split_rejects_duplicate_match_ids():
    records = [_record(i) for i in range(1, 21)]
    records[-1]["match_id"] = records[0]["match_id"]
    with pytest.raises(ValueError, match="duplicate"):
        _api().chronological_split(records)


@pytest.mark.parametrize("count", [0, 1, 19])
def test_split_rejects_too_few_records(count):
    with pytest.raises(ValueError, match="20"):
        _api().chronological_split([_record(i) for i in range(1, count + 1)])


def test_split_rejects_fewer_than_three_timestamp_groups():
    records = [_record(i, 1000 if i < 15 else 2000) for i in range(1, 21)]
    with pytest.raises(ValueError, match="timestamp"):
        _api().chronological_split(records)


def test_load_records_by_ids(tmp_path):
    records = [_record(i, i * 1000) for i in range(1, 41)]
    path = tmp_path / "source_ids.db"
    _database(path, records)
    before = path.read_bytes()

    oldest_ids = [r["match_id"] for r in records[:4]]
    loaded = _api().load_records_by_ids(str(path), oldest_ids)
    assert [r["match_id"] for r in loaded] == oldest_ids
    assert path.read_bytes() == before

    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM matches WHERE match_id = ?", (oldest_ids[0],))
    with pytest.raises(ValueError):
        _api().load_records_by_ids(str(path), oldest_ids)

    path_corrupt = tmp_path / "corrupt.db"
    corrupt_records = copy.deepcopy(records[:4])
    corrupt_records[0]["feature_json"]["team_a"] = []
    _database(path_corrupt, corrupt_records)
    with pytest.raises(ValueError):
        _api().load_records_by_ids(str(path_corrupt), [r["match_id"] for r in corrupt_records])

    with pytest.raises(ValueError):
        _api().load_records_by_ids(str(path), [oldest_ids[1], oldest_ids[1]])
