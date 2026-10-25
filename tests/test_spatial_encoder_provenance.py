"""Provenance contract for spatial trajectory embeddings.

The encoder is a fixed (untrained) projection whose weights are committed to the
repository. Every stored embedding records the encoder that produced it, and no
writer may mix encoders inside one database.
"""
import base64
import copy
import gzip
import json
import logging
import re
import sqlite3

import numpy as np
import pytest

from core import database
from core.spatial_rnn import (
    SPATIAL_ENCODER_WEIGHTS_PATH,
    MixedSpatialEncoderError,
    build_dual_phase_trajectory_encoder,
    extract_and_store_timeline_embeddings,
    extract_dual_phase_trajectory,
    generate_spatial_encoder_weights,
    load_pinned_spatial_encoder,
    load_spatial_encoder_weights,
    spatial_encoder_id,
)


def timeline(match_id, players):
    participants = [{"participantId": i + 1, "puuid": f"{match_id}-p{i}"} for i in range(players)]
    frames = [{"participantFrames": {str(i + 1): {"position": {"x": 400 * (i + 1) + 60 * f, "y": 300 + 45 * f + 20 * i}}
                                     for i in range(players)}, "events": []} for f in range(25)]
    return {"metadata": {"matchId": match_id}, "info": {"participants": participants, "frames": frames}}


def source_db(tmp_path, matches):
    path = str(tmp_path / "source.db")
    database.init_db(path)
    with sqlite3.connect(path) as conn:
        for match_id, players in matches:
            encoded = base64.b64encode(gzip.compress(json.dumps(timeline(match_id, players)).encode())).decode()
            conn.execute("INSERT INTO matches (match_id, game_creation_time) VALUES (?, ?)", (match_id, 1000))
            conn.execute("INSERT INTO match_timelines VALUES (?, ?)", (match_id, encoded))
    return path


def live_db(tmp_path, *match_ids):
    path = source_db(tmp_path, [])
    with sqlite3.connect(path) as conn:
        conn.executemany("INSERT INTO matches (match_id, game_creation_time) VALUES (?, 1234)",
                         [(m,) for m in match_ids])
    return path


def stored(path):
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        return {(r["match_id"], r["puuid"]): dict(r) for r in conn.execute("SELECT * FROM player_spatial_embeddings")}


def assert_embeddings_follow_players(rows, model, match_id):
    # Batch composition changes float round-off (oneDNN), so compare within tolerance and
    # require each stored vector to be nearest to its own player's trajectory.
    data = timeline(match_id, 10)
    trajectories = [extract_dual_phase_trajectory(data, puuid=f"{match_id}-p{i}", is_red_side=i >= 5)
                    for i in range(10)]
    expected = model([np.stack([e for e, _ in trajectories]), np.stack([m for _, m in trajectories])]).numpy()
    for player in range(10):
        vector = np.frombuffer(rows[(match_id, f"{match_id}-p{player}")]["embedding_blob"], dtype=np.float32)
        assert np.allclose(vector, expected[player], atol=1e-5)
        assert int(np.argmin(np.linalg.norm(expected - vector, axis=1))) == player


def legacy_row(path, match_id="LEGACY_1", encoder_id=None):
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT OR IGNORE INTO matches (match_id, game_creation_time) VALUES (?, 5)", (match_id,))
        conn.execute("INSERT INTO player_spatial_embeddings VALUES (?,?,?,?,?)",
                     (match_id, "old-player", 5, b"\x07" * 128, encoder_id))


def test_committed_weights_match_the_encoder_architecture():
    payload = load_spatial_encoder_weights(SPATIAL_ENCODER_WEIGHTS_PATH)
    fresh = build_dual_phase_trajectory_encoder()
    assert [w["path"] for w in payload["weights"]] == [w.path for w in fresh.weights]
    assert [tuple(w["shape"]) for w in payload["weights"]] == [tuple(w.shape) for w in fresh.weights]


def test_pinned_encoder_is_reproducible_across_loads():
    first, first_id = load_pinned_spatial_encoder()
    second, second_id = load_pinned_spatial_encoder()
    rng = np.random.default_rng(7)
    early = rng.uniform(size=(6, 15, 4)).astype(np.float32)
    mid = rng.uniform(size=(6, 10, 4)).astype(np.float32)
    assert first_id == second_id
    assert re.fullmatch(r"dual_phase_gru_v1-[0-9a-f]{16}", first_id)
    assert np.array_equal(first([early, mid]).numpy(), second([early, mid]).numpy())


def test_encoder_id_identifies_weight_values_not_file_bytes(tmp_path):
    payload = load_spatial_encoder_weights(SPATIAL_ENCODER_WEIGHTS_PATH)
    crlf = tmp_path / "crlf.json"
    crlf.write_bytes(SPATIAL_ENCODER_WEIGHTS_PATH.read_bytes().replace(b"\n", b"\r\n"))
    assert spatial_encoder_id(load_spatial_encoder_weights(crlf)) == spatial_encoder_id(payload)
    changed = copy.deepcopy(payload)
    changed["weights"][0]["values"][0] += 1e-3
    assert spatial_encoder_id(changed) != spatial_encoder_id(payload)


def test_weight_generation_is_deterministic_for_a_seed():
    assert generate_spatial_encoder_weights(11) == generate_spatial_encoder_weights(11)
    assert generate_spatial_encoder_weights(11) != generate_spatial_encoder_weights(12)


def test_schema_v15_records_the_encoder_of_each_embedding(tmp_path):
    path = str(tmp_path / "schema.db")
    database.init_db(path)
    assert database.SCHEMA_VERSION >= 15
    with sqlite3.connect(path) as conn:
        columns = [r[1] for r in conn.execute("PRAGMA table_info(player_spatial_embeddings)")]
    assert columns == ["match_id", "puuid", "game_creation_time", "embedding_blob", "encoder_id"]


def test_migrating_v14_keeps_existing_embeddings_with_unknown_encoder(tmp_path):
    path = str(tmp_path / "v14.db")
    with sqlite3.connect(path) as conn:
        for from_version, statements in database._MIGRATIONS:
            if from_version >= 14:
                break
            for sql in statements:
                conn.execute(sql)
        conn.execute("INSERT INTO matches (match_id) VALUES ('M1')")
        conn.execute("INSERT INTO player_spatial_embeddings VALUES ('M1','p',1,?)", (b"\x01" * 128,))
    database.init_db(path)
    assert [r["encoder_id"] for r in stored(path).values()] == [None]


def test_batch_embeddings_follow_their_own_player_after_an_empty_match(tmp_path):
    from scripts.compute_spatial_embeddings import compute_and_save_spatial_embeddings
    path = source_db(tmp_path, [("EMPTY_1", 0), ("FULL_2", 10)])
    compute_and_save_spatial_embeddings(db_path=path, batch_size=10)
    rows = stored(path)
    model, encoder_id = load_pinned_spatial_encoder()
    assert_embeddings_follow_players(rows, model, "FULL_2")
    assert rows[("EMPTY_1", "__EMPTY_TRAJECTORY__")]["embedding_blob"] == b"\x00" * 128
    assert {r["encoder_id"] for r in rows.values()} == {encoder_id}
    assert len(rows) == 11


@pytest.mark.parametrize("foreign", [None, "some_other_encoder"])
def test_batch_refuses_to_mix_encoders(tmp_path, foreign):
    from scripts.compute_spatial_embeddings import compute_and_save_spatial_embeddings
    path = source_db(tmp_path, [("FULL_1", 10)])
    legacy_row(path, encoder_id=foreign)
    before = stored(path)
    with pytest.raises(MixedSpatialEncoderError, match="--recompute-all"):
        compute_and_save_spatial_embeddings(db_path=path)
    assert stored(path) == before


def test_recompute_all_replaces_every_embedding_with_the_pinned_encoder(tmp_path):
    from scripts.compute_spatial_embeddings import compute_and_save_spatial_embeddings
    path = source_db(tmp_path, [("FULL_1", 10)])
    legacy_row(path)
    compute_and_save_spatial_embeddings(db_path=path, recompute_all=True)
    rows = stored(path)
    _, encoder_id = load_pinned_spatial_encoder()
    assert ("LEGACY_1", "old-player") not in rows
    assert len(rows) == 10
    assert {r["encoder_id"] for r in rows.values()} == {encoder_id}


def test_live_extraction_records_the_pinned_encoder(tmp_path):
    path = live_db(tmp_path, "LIVE_1")
    assert extract_and_store_timeline_embeddings("LIVE_1", timeline("LIVE_1", 10), 1234, db_path=path) == 10
    model, encoder_id = load_pinned_spatial_encoder()
    rows = stored(path)
    assert {r["encoder_id"] for r in rows.values()} == {encoder_id}
    assert_embeddings_follow_players(rows, model, "LIVE_1")


def test_live_extraction_skips_instead_of_mixing_encoders(tmp_path, caplog):
    path = live_db(tmp_path, "LIVE_1", "LIVE_2")
    legacy_row(path)
    with caplog.at_level(logging.WARNING, logger="core.spatial_rnn"):
        assert extract_and_store_timeline_embeddings("LIVE_1", timeline("LIVE_1", 10), 1234, db_path=path) == 0
        assert extract_and_store_timeline_embeddings("LIVE_2", timeline("LIVE_2", 10), 1234, db_path=path) == 0
    assert set(stored(path)) == {("LEGACY_1", "old-player")}
    assert len([r for r in caplog.records if "recompute-all" in r.getMessage()]) == 1


def test_spatial_training_loads_only_pinned_encoder_embeddings(tmp_path):
    from scripts.train_spatial_deepsets import load_spatial_embeddings_by_puuid
    path = live_db(tmp_path, "LIVE_1")
    extract_and_store_timeline_embeddings("LIVE_1", timeline("LIVE_1", 10), 1234, db_path=path)
    legacy_row(path, match_id="LEGACY_1")
    legacy_row(path, match_id="LEGACY_2", encoder_id="some_other_encoder")
    _, encoder_id = load_pinned_spatial_encoder()
    loaded = load_spatial_embeddings_by_puuid(path, encoder_id)
    assert set(loaded) == {f"LIVE_1-p{i}" for i in range(10)}
    assert [time for time, _ in loaded["LIVE_1-p0"]] == [1234]
    assert loaded["LIVE_1-p0"][0][1].shape == (32,)


def test_spatial_training_refuses_a_database_without_pinned_embeddings(tmp_path):
    from scripts.train_spatial_deepsets import load_spatial_embeddings_by_puuid
    path = source_db(tmp_path, [])
    legacy_row(path)
    _, encoder_id = load_pinned_spatial_encoder()
    with pytest.raises(ValueError, match="--recompute-all"):
        load_spatial_embeddings_by_puuid(path, encoder_id)
