"""Integration contracts for the offline, canonical-preserving SQLite merger."""
import base64
import copy
import gzip
import importlib
import json
import sqlite3

import pytest


def api():
    return importlib.import_module("scripts.merge_databases")


MERGE_SCHEMA_VERSION = 14


def database(path, schema_version=MERGE_SCHEMA_VERSION):
    # Migrations are inert SQL; no app/config/network initialization is needed.
    from core.database import _MIGRATIONS
    with sqlite3.connect(path) as conn:
        for from_version, statements in _MIGRATIONS:
            if from_version >= schema_version:
                break
            for sql in statements:
                conn.execute(sql)
    return path


def enqueue(path, *entries):
    with sqlite3.connect(path) as conn:
        for entry in entries:
            row = {"region": "EUW", "priority": 0, "is_processed": 0, **entry}
            conn.execute(f'INSERT INTO crawl_queue ({",".join(row)}) VALUES ({",".join("?" for _ in row)})',
                         tuple(row.values()))


def embed(path, match_id, puuid, blob=b"\x01" * 128, created=1000):
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO player_spatial_embeddings VALUES (?,?,?,?)", (match_id, puuid, created, blob))


def raw(match_id="EUW1_1", date=1000):
    participants = []
    for i in range(10):
        participants.append({
            "puuid": "real-puuid-" + str(i), "participantId": i + 1,
            "teamId": 100 if i < 5 else 200, "win": i < 5,
            "championName": "Ahri", "teamPosition": "MIDDLE",
            "kills": 2, "deaths": 2, "assists": 4, "goldEarned": 10000,
            "totalMinionsKilled": 100, "neutralMinionsKilled": 10,
            "riotIdGameName": "Latest" + str(i), "riotIdTagline": "EUW",
        })
    return {"metadata": {"matchId": match_id}, "info": {
        "gameMode": "CLASSIC", "gameCreation": date, "gameDuration": 1500,
        "gameVersion": "16.18.1", "participants": participants,
        "teams": [{"teamId": 100, "win": True}, {"teamId": 200, "win": False}],
    }}


def add_match(path, match_id="EUW1_1", payload=None, compressed=False, training_id=None):
    payload = raw(match_id) if payload is None else payload
    encoded = json.dumps(payload) if not isinstance(payload, str) else payload
    if compressed:
        encoded = base64.b64encode(gzip.compress(encoded.encode())).decode()
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO matches VALUES (?,?,?,?,?,?)",
                     (match_id, encoded, "CLASSIC", 1000, 1500, 16))
        if training_id is not None:
            conn.execute("""INSERT INTO training_dataset
                (id,match_id,feature_json,winner_label,is_trained,last_trained_at,
                 llm_summary,llm_tags,llm_embedding,labels_json)
                VALUES (?,?,?,1,1,'yesterday','useful','["tag"]',?,?)""",
                         (training_id, match_id, '{"stale":true}', b"embedding", '{}'))


def run(tmp_path, source, target, suffix="", **kwargs):
    return api().merge_databases(source=source, target=target,
        output=tmp_path / ("merged" + suffix + ".db"),
        report=tmp_path / ("report" + suffix + ".json"), **kwargs)


def rows(path, table):
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]


def archived(report):
    with gzip.open(report["archive"]["path"], "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_merge_contract_tracks_the_application_schema():
    # A new migration must come with an explicit merge policy, not silent drift.
    from core.database import SCHEMA_VERSION
    assert api().SCHEMA_VERSION == SCHEMA_VERSION == MERGE_SCHEMA_VERSION


@pytest.mark.parametrize("stale", ["source", "target"])
def test_older_schema_inputs_are_refused_with_migration_guidance(tmp_path, stale):
    paths = {name: database(tmp_path / f"{name}.db", 12 if name == stale else MERGE_SCHEMA_VERSION)
             for name in ("source", "target")}
    with pytest.raises(ValueError, match="only schema version 14 is supported.*migrate a copy"):
        run(tmp_path, paths["source"], paths["target"])
    assert not (tmp_path / "merged.db").exists()


def test_report_declares_v14_and_counts_spatial_embeddings(tmp_path):
    source, target = database(tmp_path / "source.db"), database(tmp_path / "target.db")
    add_match(target)
    embed(target, "EUW1_1", "real-puuid-0")
    report = run(tmp_path, source, target)
    assert report["schema_version"] == 14
    assert report["before"]["target"]["player_spatial_embeddings"] == 1
    assert report["after"]["player_spatial_embeddings"] == 1
    assert "player_spatial_embeddings" in report["inputs"]["target"]["tables"]
    assert "player_spatial_embeddings" in report["target_preservation"]["changed_or_missing_rows"]


def test_source_spatial_embeddings_are_archived_and_never_imported(tmp_path):
    # The trajectory encoder is randomly initialised per run, so two databases'
    # embeddings live in unrelated spaces: only the canonical target's are kept.
    source, target = database(tmp_path / "source.db"), database(tmp_path / "target.db")
    add_match(target, "EUW1_1")
    add_match(source, "EUW1_1")
    add_match(source, "EUW1_2")
    canonical_blob, source_blob = b"\x01" * 128, b"\x02" * 128
    embed(target, "EUW1_1", "real-puuid-0", canonical_blob)
    embed(source, "EUW1_1", "real-puuid-0", canonical_blob)  # identical: nothing to preserve
    embed(source, "EUW1_1", "real-puuid-1", source_blob)     # target lacks it: still not imported
    embed(source, "EUW1_2", "real-puuid-0", source_blob)     # imported match
    embed(source, "EUW1_9", "real-puuid-0", source_blob)     # orphan
    report = run(tmp_path, source, target)
    assert rows(tmp_path / "merged.db", "player_spatial_embeddings") == rows(target, "player_spatial_embeddings")
    assert report["classification"] == {"existing": 1, "imported": 1, "archived": 0}
    assert report["target_preservation"]["ok"] is True
    reasons = {(e["row"]["match_id"], e["row"]["puuid"]): e["reason"]
               for e in archived(report) if e.get("table") == "player_spatial_embeddings"}
    assert reasons == {
        ("EUW1_1", "real-puuid-1"): "derived_embedding_not_merged",
        ("EUW1_2", "real-puuid-0"): "derived_embedding_not_merged",
        ("EUW1_9", "real-puuid-0"): "orphan_relationship",
    }


def test_queue_target_wins_and_new_entries_get_fresh_ids(tmp_path):
    source, target = database(tmp_path / "source.db"), database(tmp_path / "target.db")
    enqueue(target, {"puuid": "puuid-shared", "priority": 10, "created_at": "canonical"},
            {"puuid": "puuid-target-only", "region": "KR"})
    enqueue(source, {"puuid": "puuid-new", "priority": 5, "is_processed": 1,
                     "created_at": "then", "processed_at": "later"},
            {"puuid": "puuid-shared", "priority": 2, "is_processed": 1, "region": "NA"})
    canonical = rows(target, "crawl_queue")
    source_rows = {r["puuid"]: r for r in rows(source, "crawl_queue")}
    report = run(tmp_path, source, target)
    merged = {r["puuid"]: r for r in rows(tmp_path / "merged.db", "crawl_queue")}
    assert [merged[r["puuid"]] for r in canonical] == canonical
    new = merged["puuid-new"]
    assert new["id"] > max(r["id"] for r in canonical)
    assert {k: v for k, v in new.items() if k != "id"} == {
        k: v for k, v in source_rows["puuid-new"].items() if k != "id"}
    assert report["target_preservation"]["ok"] is True
    reasons = {e["row"]["puuid"]: e["reason"] for e in archived(report) if e.get("table") == "crawl_queue"}
    assert reasons == {"puuid-new": "regenerated_queue_id", "puuid-shared": "canonical_queue_precedence"}


def test_identical_queue_row_is_skipped_and_merge_is_repeatable(tmp_path):
    source, target = database(tmp_path / "source.db"), database(tmp_path / "target.db")
    enqueue(target, {"puuid": "puuid-same", "created_at": "t"}, {"puuid": "puuid-other"})
    enqueue(source, {"puuid": "puuid-same", "created_at": "t"}, {"puuid": "puuid-new"})
    first = run(tmp_path, source, target)
    merged = tmp_path / "merged.db"
    assert [e["row"]["puuid"] for e in archived(first) if e.get("table") == "crawl_queue"] == ["puuid-new"]
    queue = rows(merged, "crawl_queue")
    second = run(tmp_path, source, merged, "2")
    assert rows(tmp_path / "merged2.db", "crawl_queue") == queue
    assert second["target_preservation"]["ok"] is True


def test_legacy_queue_entries_without_puuid_match_on_riot_id_and_region(tmp_path):
    source, target = database(tmp_path / "source.db"), database(tmp_path / "target.db")
    enqueue(target, {"game_name": "Legacy", "tag_line": "EUW", "priority": 9})
    enqueue(source, {"game_name": "Legacy", "tag_line": "EUW", "priority": 1},
            {"game_name": "Legacy", "tag_line": "EUW", "region": "KR"},
            {"game_name": "Fresh", "tag_line": "EUW"})
    report = run(tmp_path, source, target)
    merged = {(r["game_name"], r["region"]): r for r in rows(tmp_path / "merged.db", "crawl_queue")}
    assert set(merged) == {("Legacy", "EUW"), ("Legacy", "KR"), ("Fresh", "EUW")}
    assert merged[("Legacy", "EUW")]["priority"] == 9
    assert all(r["puuid"] is None for r in merged.values())
    conflicts = [(e["row"]["game_name"], e["row"]["region"], e["reason"]) for e in archived(report)
                 if e.get("table") == "crawl_queue" and e["reason"] != "regenerated_queue_id"]
    assert conflicts == [("Legacy", "EUW", "canonical_queue_precedence")]


@pytest.mark.parametrize("entry", [
    {"puuid": "UNKNOWN"},
    {"puuid": "__TOMBSTONE__"},
    {"puuid": "puuid-x", "region": ""},
    {"puuid": "puuid-x", "region": "MARS"},
    {"puuid": "puuid-x", "priority": "high"},
    {"puuid": "puuid-x", "is_processed": 2},
    {"game_name": "OnlyName"},
    {},
])
def test_invalid_queue_entries_are_archived(tmp_path, entry):
    source, target = database(tmp_path / "source.db"), database(tmp_path / "target.db")
    enqueue(source, entry)
    report = run(tmp_path, source, target)
    assert rows(tmp_path / "merged.db", "crawl_queue") == []
    assert [e["reason"] for e in archived(report) if e.get("table") == "crawl_queue"] == [
        "invalid_queue_entry"]


def test_merge_preserves_canonical_ids_regenerates_source_and_is_repeatable(tmp_path):
    source, target = database(tmp_path / "source.db"), database(tmp_path / "target.db")
    add_match(target, "EUW1_1", training_id=8)
    add_match(source, "EUW1_1", raw("EUW1_1", 2000), training_id=7)
    add_match(source, "EUW1_2", compressed=True, training_id=8)
    before = {p: p.read_bytes() for p in (source, target)}
    canonical = rows(target, "training_dataset")
    report = run(tmp_path, source, target)
    merged = tmp_path / "merged.db"
    training = rows(merged, "training_dataset")
    assert training[0] == canonical[0]
    assert training[1]["id"] > 8
    assert training[1]["is_trained"] == 0
    assert training[1]["last_trained_at"] is None
    assert training[1]["llm_summary"] == "useful"
    assert training[1]["llm_embedding"] == b"embedding"
    assert json.loads(training[1]["labels_json"])["winner"] == 0
    assert len(json.loads(training[1]["feature_json"])["team_a"]) == 5
    assert len(rows(merged, "match_participants")) == 10
    assert report["classification"] == {"existing": 1, "imported": 1, "archived": 0}
    assert report["target_preservation"]["ok"] is True
    assert report["verification"]["integrity_check"] == ["ok"]
    assert report["verification"]["foreign_key_check"] == []
    assert all(p.read_bytes() == contents for p, contents in before.items())
    entries = archived(report)
    assert any(e.get("table") == "matches" and e["row"]["match_id"] == "EUW1_1" for e in entries)
    report2 = run(tmp_path, source, merged, "2")
    assert report2["classification"] == {"existing": 2, "imported": 0, "archived": 0}
    assert rows(tmp_path / "merged2.db", "training_dataset") == training


@pytest.mark.parametrize("defect", ["json", "mode", "match_id", "date", "duplicate", "synthetic", "teams", "winner"])
def test_invalid_matches_and_all_their_relationships_are_archived(tmp_path, defect):
    source, target = database(tmp_path / "source.db"), database(tmp_path / "target.db")
    payload = raw()
    if defect == "json": payload = "not-json"
    elif defect == "mode": payload["info"]["gameMode"] = "ARAM"
    elif defect == "match_id": payload["metadata"]["matchId"] = "EUW1_wrong"
    elif defect == "date": payload["info"]["gameCreation"] = True
    elif defect == "duplicate": payload["info"]["participants"][1]["puuid"] = "real-puuid-0"
    elif defect == "synthetic": payload["info"]["participants"][0]["puuid"] = "UNKNOWN"
    elif defect == "teams": payload["info"]["participants"][0]["teamId"] = 200
    elif defect == "winner": payload["info"]["teams"][1]["win"] = True
    add_match(source, payload=payload, training_id=1)
    with sqlite3.connect(source) as conn:
        conn.execute("INSERT INTO match_timelines VALUES ('EUW1_1','{}')")
        conn.execute("INSERT INTO match_participants (match_id,puuid) VALUES ('EUW1_1','x')")
        conn.execute("INSERT INTO training_dataset (match_id) VALUES ('orphan')")
    report = run(tmp_path, source, target)
    assert report["classification"] == {"existing": 0, "imported": 0, "archived": 1}
    entries = archived(report)
    for table in ("matches", "training_dataset", "match_timelines", "match_participants"):
        assert any(e.get("table") == table and e["row"]["match_id"] == "EUW1_1" for e in entries)
    assert any(e.get("table") == "training_dataset" and e["row"]["match_id"] == "orphan" for e in entries)
    assert rows(tmp_path / "merged.db", "matches") == []


def test_identity_precedence_alias_conflicts_and_aggregate_stats(tmp_path):
    source, target = database(tmp_path / "source.db"), database(tmp_path / "target.db")
    add_match(source)
    payload = raw("EUW1_2", 2000)
    payload["info"]["participants"][0].update(kills=8, deaths=0, goldEarned=12000,
                                               riotIdGameName="Newest")
    add_match(source, "EUW1_2", payload)
    with sqlite3.connect(target) as conn:
        conn.execute("INSERT INTO players (puuid,game_name) VALUES ('real-puuid-1','Canonical')")
    with sqlite3.connect(source) as conn:
        conn.executemany("INSERT INTO players (puuid,game_name,tag_line) VALUES (?,?,?)", [
            ("real-puuid-0", "Old", "OLD"), ("real-puuid-1", "Wrong", "OLD"),
            ("alias-player", "Alias", "EUW"), ("UNKNOWN", "Fake", "EUW")])
        conn.executemany("INSERT INTO player_aliases VALUES (?,?)", [
            ("alias-player", "real-puuid-0"), ("missing", "real-puuid-0"),
            ("real-puuid-0", "alias-player")])
        conn.executemany("INSERT INTO champions VALUES (?)", [("Zed",), ("UNKNOWN",)])
    report = run(tmp_path, source, target)
    players = {r["puuid"]: r for r in rows(tmp_path / "merged.db", "players")}
    assert players["real-puuid-0"]["game_name"] == "Newest"
    assert players["real-puuid-1"]["game_name"] == "Canonical"
    assert players["real-puuid-0"]["total_matches"] == 2
    assert players["real-puuid-0"]["avg_kda"] == 9
    assert players["real-puuid-0"]["avg_gold"] == 11000
    assert "UNKNOWN" not in players
    aliases = rows(tmp_path / "merged.db", "player_aliases")
    assert len(aliases) <= 1
    assert all(r["alias_puuid"] in players and r["canonical_puuid"] in players for r in aliases)
    assert {r["name"] for r in rows(tmp_path / "merged.db", "champions")} == {"Ahri", "Zed"}
    assert any(e.get("table") == "player_aliases" for e in archived(report))


def test_dry_run_and_invalid_paths_never_create_artifacts(tmp_path):
    source, target = database(tmp_path / "source.db"), database(tmp_path / "target.db")
    add_match(source)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    report = run(tmp_path, source, target, dry_run=True)
    assert report["classification"]["imported"] == 1
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    with pytest.raises(ValueError):
        api().merge_databases(source=source,target=target,output=source,report=tmp_path / "r.json")
    with pytest.raises(ValueError):
        api().merge_databases(source=source,target=target,output=tmp_path / "new.db",report=source)
    with sqlite3.connect(source) as conn:
        conn.execute("UPDATE schema_version SET version=11")
    with pytest.raises(ValueError, match="schema"):
        run(tmp_path, source, target)
    assert not (tmp_path / "merged.db").exists()


def test_existing_archive_or_report_is_never_overwritten(tmp_path):
    source, target = database(tmp_path / "source.db"), database(tmp_path / "target.db")
    path = tmp_path / "report.json.archive.jsonl.gz"
    path.write_bytes(b"valuable")
    with pytest.raises((ValueError, FileExistsError)):
        run(tmp_path, source, target)
    assert path.read_bytes() == b"valuable"
    assert not (tmp_path / "merged.db").exists()


def test_latest_nonempty_names_are_used_even_when_newest_match_lacks_them(tmp_path):
    source, target = database(tmp_path / "source.db"), database(tmp_path / "target.db")
    newer = raw("EUW1_1", 3000)
    newer["info"]["participants"][0].pop("riotIdGameName")
    newer["info"]["participants"][0].pop("riotIdTagline")
    add_match(source, "EUW1_1", newer)
    add_match(source, "EUW1_2", raw("EUW1_2", 2000))
    run(tmp_path, source, target)
    player = next(r for r in rows(tmp_path / "merged.db", "players") if r["puuid"] == "real-puuid-0")
    assert player["game_name"] == "Latest0"
    assert player["tag_line"] == "EUW"


@pytest.mark.parametrize("field", ["teamId", "championName", "gameDuration"])
def test_nested_malformed_values_are_archived_without_aborting(tmp_path, field):
    source, target = database(tmp_path / "source.db"), database(tmp_path / "target.db")
    payload = raw()
    if field == "gameDuration":
        payload["info"][field] = {}
    else:
        payload["info"]["participants"][0][field] = {}
    add_match(source, payload=payload)
    result = run(tmp_path, source, target)
    assert result["classification"]["archived"] == 1
    assert len([e for e in archived(result) if e.get("table") == "matches"]) == 1


def test_compatible_timeline_is_retained_and_adds_training_labels(tmp_path):
    source, target = database(tmp_path / "source.db"), database(tmp_path / "target.db")
    add_match(source)
    timeline = {"metadata": {"matchId": "EUW1_1"}, "info": {
        "participants": [{"puuid": "real-puuid-" + str(i), "participantId": i + 1} for i in range(10)],
        "frames": [{"events": [{"type": "CHAMPION_KILL", "killerId": 1} for _ in range(5)]}]}}
    with sqlite3.connect(source) as conn:
        conn.execute("INSERT INTO match_timelines VALUES (?,?)", ("EUW1_1", json.dumps(timeline)))
    report = run(tmp_path, source, target)
    assert len(rows(tmp_path / "merged.db", "match_timelines")) == 1
    labels = json.loads(rows(tmp_path / "merged.db", "training_dataset")[0]["labels_json"])
    assert labels["first_to_5_kills"] == 0
    assert labels["first_to_10_kills"] == 2
    assert not any(e.get("table") == "match_timelines" for e in archived(report))


def test_committed_wal_is_included_without_checkpointing_inputs(tmp_path):
    source, target = database(tmp_path / "source.db"), database(tmp_path / "target.db")
    live = sqlite3.connect(source)
    try:
        live.execute("PRAGMA journal_mode=WAL")
        live.execute("PRAGMA wal_autocheckpoint=0")
        add_match(source)
        source_bytes = source.read_bytes()
        wal = tmp_path / "source.db-wal"
        wal_bytes = wal.read_bytes()
        report = run(tmp_path, source, target)
        assert report["classification"]["imported"] == 1
        assert report["inputs"]["source"]["wal_present"] is True
        assert source.read_bytes() == source_bytes
        assert wal.read_bytes() == wal_bytes
    finally:
        live.close()


def test_input_sqlite_sidecars_cannot_be_selected_as_outputs(tmp_path):
    source, target = database(tmp_path / "source.db"), database(tmp_path / "target.db")
    with pytest.raises(ValueError):
        api().merge_databases(source=source, target=target, output=str(source) + "-wal",
                             report=tmp_path / "r.json")
    assert not (tmp_path / "source.db-wal").exists()


def test_alias_target_wins_queue_conflicts_and_full_archive_manifest(tmp_path):
    source, target = database(tmp_path / "source.db"), database(tmp_path / "target.db")
    for path in (source, target):
        with sqlite3.connect(path) as conn:
            conn.executemany("INSERT INTO players (puuid) VALUES (?)", [("player-a",), ("player-b",), ("player-c",)])
    with sqlite3.connect(target) as conn:
        conn.execute("INSERT INTO player_aliases VALUES ('player-a','player-b')")
    enqueue(target, {'puuid': 'queue-puuid', 'priority': 10})
    with sqlite3.connect(source) as conn:
        conn.execute("INSERT INTO player_aliases VALUES ('player-a','player-c')")
    enqueue(source, {'puuid': 'queue-puuid', 'priority': 2, 'is_processed': 1},
            {'puuid': 'new-puuid', 'priority': 3}, {'puuid': 'UNKNOWN', 'priority': 1})
    add_match(source, 'EUW1_1', payload='bad')
    add_match(source, 'EUW1_2')
    result = run(tmp_path, source, target)
    assert rows(tmp_path / 'merged.db', 'player_aliases') == [{'alias_puuid': 'player-a', 'canonical_puuid': 'player-b'}]
    queue = {r['puuid']: r for r in rows(tmp_path / 'merged.db', 'crawl_queue')}
    assert set(queue) == {'queue-puuid', 'new-puuid'}
    assert queue['queue-puuid']['priority'] == 10
    entries = archived(result)
    classifications = {e['match_id']: e['classification'] for e in entries if e['kind'] == 'classification'}
    assert classifications == {'EUW1_1': 'archived', 'EUW1_2': 'imported'}
    assert result['archive']['records'] == len(entries)
    import hashlib
    assert result['archive']['sha256'] == hashlib.sha256(__import__('pathlib').Path(result['archive']['path']).read_bytes()).hexdigest()
    assert len([e for e in entries if e.get('table') == 'crawl_queue']) == 2


def test_orphan_relationship_is_archived_even_if_canonical_has_identical_row(tmp_path):
    source, target = database(tmp_path / "source.db"), database(tmp_path / "target.db")
    add_match(target)
    for path in (source, target):
        with sqlite3.connect(path) as conn:
            conn.execute("INSERT INTO match_timelines VALUES ('EUW1_1','{}')")
    report = run(tmp_path, source, target)
    assert any(e.get('table') == 'match_timelines' and e['reason'] == 'orphan_relationship' for e in archived(report))


@pytest.mark.parametrize('marker', ['__TOMBSTONE__', '__UNKNOWN_EUW1_1_1'])
def test_repository_synthetic_puuids_are_rejected_and_not_created_as_players(tmp_path, marker):
    source, target = database(tmp_path / 'source.db'), database(tmp_path / 'target.db')
    payload = raw()
    payload['info']['participants'][0]['puuid'] = marker
    add_match(source, payload=payload)
    with sqlite3.connect(source) as conn:
        conn.execute('INSERT INTO players (puuid) VALUES (?)', (marker,))
    report = run(tmp_path, source, target)
    assert report['classification']['archived'] == 1
    assert rows(tmp_path / 'merged.db', 'players') == []
    assert any(e.get('table') == 'players' for e in archived(report))


def test_canonical_unknown_identity_is_filled_without_preservation_failure(tmp_path):
    source, target = database(tmp_path / 'source.db'), database(tmp_path / 'target.db')
    add_match(source)
    with sqlite3.connect(target) as conn:
        conn.execute("INSERT INTO players (puuid,game_name,tag_line) VALUES ('real-puuid-0','UNKNOWN','N/A')")
    report = run(tmp_path, source, target)
    player = next(r for r in rows(tmp_path / 'merged.db', 'players') if r['puuid'] == 'real-puuid-0')
    assert player['game_name'] == 'Latest0'
    assert player['tag_line'] == 'EUW'
    assert report['target_preservation']['ok'] is True


def test_concurrent_input_write_refuses_readiness_report_and_cleans_staged_artifacts(tmp_path, monkeypatch):
    source, target = database(tmp_path / 'source.db'), database(tmp_path / 'target.db')
    add_match(source)
    writer = sqlite3.connect(source)
    writer.execute('PRAGMA journal_mode=WAL')
    original = api()._merge_registries
    def concurrent_write(*args):
        original(*args)
        writer.execute("INSERT INTO champions VALUES ('Concurrent')")
        writer.commit()
    monkeypatch.setattr(api(), '_merge_registries', concurrent_write)
    try:
        with pytest.raises(ValueError, match='changed'):
            run(tmp_path, source, target)
        assert not (tmp_path / 'merged.db').exists()
        assert not (tmp_path / 'report.json').exists()
        assert not (tmp_path / 'report.json.archive.jsonl.gz').exists()
    finally:
        writer.close()


def test_null_canonical_alias_is_archived_instead_of_disappearing(tmp_path):
    source, target = database(tmp_path / 'source.db'), database(tmp_path / 'target.db')
    with sqlite3.connect(source) as conn:
        conn.execute("INSERT INTO players (puuid) VALUES ('player-a')")
        conn.execute("INSERT INTO player_aliases VALUES ('player-a',NULL)")
    report = run(tmp_path, source, target)
    assert rows(tmp_path / 'merged.db', 'player_aliases') == []
    assert any(e.get('table') == 'player_aliases' and e['row']['canonical_puuid'] is None for e in archived(report))


def test_logical_fingerprint_is_independent_of_sqlite_insertion_order(tmp_path):
    first, second = database(tmp_path / 'first.db'), database(tmp_path / 'second.db')
    for match_id in ('EUW1_1', 'EUW1_2', 'EUW1_3'):
        add_match(first, match_id)
    for match_id in ('EUW1_3', 'EUW1_1', 'EUW1_2'):
        add_match(second, match_id)
    from contextlib import closing
    with closing(api()._read_only(first)) as a, closing(api()._read_only(second)) as b:
        left, right = api()._fingerprint(a, first), api()._fingerprint(b, second)
    assert left['logical_sha256'] == right['logical_sha256']
    assert left['algorithm'] == 'sha256-sorted-row-sha256-v1'


def test_identity_stats_use_table_scan_and_preserve_integer_null_arithmetic(tmp_path, monkeypatch):
    source, target = database(tmp_path / "source.db"), database(tmp_path / "target.db")
    samples = [
        ("EUW1_1", "player-b", 8, 0, 3, 12000, 3000),
        ("EUW1_2", "player-a", None, None, 4, 10000, 1000),
        ("EUW1_3", "player-b", 2, 2, 1, None, 1000),
        ("EUW1_4", "player-a", 6, 1, None, 18000, 4000),
        ("EUW1_5", "player-zero", None, None, None, None, 2000),
        ("EUW1_6", "__UNKNOWN_M6_1", 100, 1, 100, 99999, 5000),
    ]
    for match_id, *_ in samples:
        add_match(target, match_id)
    with sqlite3.connect(target) as conn:
        conn.executemany("INSERT INTO match_participants "
            "(match_id,puuid,kills,deaths,assists,gold_earned,game_creation_time) "
            "VALUES (?,?,?,?,?,?,?)", samples)
        conn.execute("INSERT INTO players(puuid,total_matches,avg_kda,avg_gold) "
                     "VALUES ('__UNKNOWN_M6_1',99,99,99)")
    original = api()._merge_identities
    observed_plan = []

    def capture_stats_plan(src, out, archive):
        statements = []
        out.set_trace_callback(statements.append)
        try:
            original(src, out, archive)
        finally:
            out.set_trace_callback(None)
        statement = next(sql for sql in statements if sql.startswith("CREATE TEMP TABLE merge_stats AS SELECT"))
        select = statement.split(" AS ", 1)[1]
        observed_plan.extend(row[3] for row in out.execute("EXPLAIN QUERY PLAN " + select))

    monkeypatch.setattr(api(), "_merge_identities", capture_stats_plan)
    report = run(tmp_path, source, target)
    players = {row["puuid"]: row for row in rows(tmp_path / "merged.db", "players")}
    for puuid, expected in {
        "player-a": (2, 10.0, 14000.0),
        "player-b": (2, 7.0, 6000.0),
        "player-zero": (1, 0.0, 0.0),
        "__UNKNOWN_M6_1": (0, 0.0, 0.0),
    }.items():
        assert tuple(players[puuid][key] for key in ("total_matches", "avg_kda", "avg_gold")) == expected
    assert report["target_preservation"]["ok"]
    assert observed_plan == ["SCAN match_participants", "USE TEMP B-TREE FOR GROUP BY"]
