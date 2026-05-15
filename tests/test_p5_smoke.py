"""
End-to-end smoke test for P5: synthetic match -> save -> backfill -> train one batch.

Verifies:
- DB migration to v10 works on a fresh DB
- save_training_record + save_match round-trips correctly
- backfill_labels populates labels_json from raw_json
- ContinuousTrainer can build the model, fetch records, build dict targets,
  and run one train_on_batch without crashing
- model_v2.keras is written

This test does NOT verify model accuracy -- only that the pipeline executes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import numpy as np
import pytest


def _make_real_match(match_id: str, team_a_win: bool, seed: int = 0) -> dict:
    """Build a realistic Match-V5-ish dict that extract_labels can process."""
    rng = np.random.default_rng(seed)
    a_kills = int(rng.integers(5, 25))
    b_kills = int(rng.integers(5, 25))
    participants = []
    for tid, kills_total in ((100, a_kills), (200, b_kills)):
        # Split kills across 5 fake participants per team
        per = [kills_total // 5] * 5
        per[0] += kills_total - sum(per)
        for i, k in enumerate(per):
            participants.append({
                "puuid": f"puuid_{tid}_{i}",
                "teamId": tid,
                "championName": f"Champ{tid}_{i}",
                "kills": k,
                "deaths": int(rng.integers(0, 10)),
                "assists": int(rng.integers(0, 15)),
                "goldEarned": int(rng.integers(8000, 18000)),
                "teamPosition": ["TOP", "JUNGLE", "MID", "BOTTOM", "UTILITY"][i],
            })
    return {
        "info": {
            "gameVersion": "14.10.400",
            "gameMode": "CLASSIC",
            "gameCreation": 1700000000000 + seed * 1000,
            "gameDuration": int(rng.integers(1200, 2400)),
            "participants": participants,
            "teams": [
                {"teamId": 100, "win": team_a_win, "objectives": {
                    "baron":     {"first": team_a_win, "kills": 1 if team_a_win else 0},
                    "dragon":    {"first": True,       "kills": int(rng.integers(0, 5))},
                    "inhibitor": {"first": team_a_win, "kills": 1 if team_a_win else 0},
                    "tower":     {"first": True,       "kills": int(rng.integers(3, 12))},
                    "champion":  {"first": team_a_win, "kills": a_kills},
                }},
                {"teamId": 200, "win": not team_a_win, "objectives": {
                    "baron":     {"first": not team_a_win, "kills": 1 if not team_a_win else 0},
                    "dragon":    {"first": False,           "kills": int(rng.integers(0, 5))},
                    "inhibitor": {"first": not team_a_win, "kills": 1 if not team_a_win else 0},
                    "tower":     {"first": False,           "kills": int(rng.integers(3, 12))},
                    "champion":  {"first": not team_a_win, "kills": b_kills},
                }},
            ],
        }
    }


def test_p5_end_to_end(tmp_path, monkeypatch):
    """
    Smoke test: synthetic DB -> backfill -> one training step -> model_v2 saved.

    Loads TF and trains one micro-batch; ~30-60s.
    """
    # Force the trainer to use the temp DB and a small batch
    db_path = str(tmp_path / "test_training.db")
    monkeypatch.setenv("NNRIOT_DB_PATH", db_path)
    monkeypatch.setenv("NNRIOT_BATCH_SIZE", "8")
    monkeypatch.setenv("NNRIOT_EPOCHS", "1")
    monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "3")
    # Use a small CHECKPOINT_PATH in tmp so we don't touch the real model file
    checkpoint_path = str(tmp_path / "model_v2_test.keras")

    import database
    import feature_labels
    from data_collector import _build_training_record

    # 1. Init fresh DB at v10
    database.init_db(db_path)

    # Verify schema reached v10
    with database.get_connection(db_path) as conn:
        v = database._get_schema_version(conn)
    assert v == 10, f"Expected schema v10, got v{v}"

    # 2. Insert 12 synthetic matches with labels
    matches_batch = []
    records_with_labels = []
    for i in range(12):
        m_id = f"TEST_{i:03d}"
        match = _make_real_match(m_id, team_a_win=(i % 2 == 0), seed=i)
        matches_batch.append((m_id, match))
        # Build training record using the existing _build_training_record + labels
        rec = _build_training_record(m_id, match, region="TEST")
        assert rec is not None, f"Match {m_id} should produce a valid training record"
        labels = feature_labels.extract_labels(match)
        assert labels is not None, f"Match {m_id} should have extractable labels"
        records_with_labels.append((rec[0], rec[1], rec[2], labels))

    database.save_matches_batch(matches_batch, db_path=db_path)
    database.save_training_records_batch(records_with_labels, db_path=db_path)

    # 3. Verify labels_json was saved
    with database.get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT labels_json FROM training_dataset WHERE labels_json IS NOT NULL LIMIT 1"
        ).fetchone()
    assert row is not None, "At least one training record should have labels_json"

    # 4. Run backfill (should be no-op since labels_json is already populated)
    from backfill_labels import backfill_labels
    stats = backfill_labels(db_path)
    assert stats["total"] == 0, f"No records should need backfill: {stats}"

    # 5. Insert a 13th record WITHOUT labels (3-tuple) to test backfill
    extra_match = _make_real_match("TEST_extra", team_a_win=True, seed=99)
    extra_rec = _build_training_record("TEST_extra", extra_match, region="TEST")
    database.save_matches_batch([("TEST_extra", extra_match)], db_path=db_path)
    # 3-tuple, no labels
    database.save_training_records_batch(
        [(extra_rec[0], extra_rec[1], extra_rec[2])],
        db_path=db_path,
    )

    stats = backfill_labels(db_path)
    assert stats["total"] == 1, f"Exactly 1 record should need backfill: {stats}"
    assert stats["updated"] == 1, f"Backfill should update 1 record: {stats}"

    # 6. Build and run trainer for one step using mocked checkpoint
    import continuous_trainer
    # Force the env-var-derived module globals in case the module was imported
    # earlier by another test (env vars are read at import time).
    monkeypatch.setattr(continuous_trainer, "CHECKPOINT_PATH", checkpoint_path)
    monkeypatch.setattr(continuous_trainer, "BATCH_SIZE", 8)
    monkeypatch.setattr(continuous_trainer, "EPOCHS_PER_BATCH", 1)
    # Use small input dim to keep test fast -- trainer reads VECTOR_DIM at runtime
    monkeypatch.setattr(continuous_trainer, "VECTOR_DIM", 1000)

    # Make the trainer use the temp DB path even though it imports database at top.
    # database.DB_PATH was already read at import; patch the default explicitly.
    monkeypatch.setattr(database, "DB_PATH", db_path)

    trainer = continuous_trainer.ContinuousTrainer()
    # Confirm model has 18 heads
    assert len(trainer.model.output_names) == 18

    # 7. Run one training step
    processed = trainer.run_training_step()
    assert processed > 0, "Should have processed at least one record"

    # 8. Verify checkpoint was saved
    assert Path(checkpoint_path).exists(), f"Checkpoint should be saved at {checkpoint_path}"
