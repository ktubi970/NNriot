"""Reusable 140-match fixture for real ordered neural and ensemble smoke runs."""
from __future__ import annotations

import json
from pathlib import Path
import pickle
import sqlite3

from core.experiment import build_manifest, resolve_result_end_times, write_result_end_times_artifact
from core.pre_match_dataset import split_records


def make_ordered_fixture(base_dir: str | Path) -> dict:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=False)
    roles = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
    left = ("Aatrox", "LeeSin", "Ahri", "Jinx", "Thresh")
    right = ("Darius", "Elise", "Zed", "Ezreal", "Lulu")
    records = []
    for i in range(140):
        creation = 1_700_000_000_000 + i * 60_000
        winner = i % 2
        records.append({
            "match_id": f"ORDERED_SMOKE_{i:03d}",
            "game_creation_time": creation,
            "feature_json": {
                "team_a": [{"champion": c, "role": role, "priors": [0.5] * 10}
                           for c, role in zip(left, roles)],
                "team_b": [{"champion": c, "role": role, "priors": [0.5] * 10}
                           for c, role in zip(right, roles)],
                "game_mode": "CLASSIC", "game_version": "14.1.1",
            },
            "labels_json": {
                "winner": winner, "team_b_kill_lead": winner,
                "first_blood": winner, "first_baron": 2,
                "first_inhibitor": 2, "first_tower": winner,
                "kills_odd": 0, "both_baron": 0, "both_inhibitor": 0,
                "both_dragon": 1, "elder_dragon": 0,
                "kill_handicap": 2 if winner else -2,
                "total_kills": 30, "team_a_kills": 14 if winner else 16,
                "team_b_kills": 16 if winner else 14,
                "total_barons": 1, "total_dragons": 3, "total_towers": 11,
            },
            "winner_label": winner,
        })
    splits, split_meta = split_records(records, strategy="chronological")
    db = base / "results.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE matches (match_id TEXT PRIMARY KEY, game_creation_time INTEGER, raw_json TEXT)")
        conn.executemany(
            "INSERT INTO matches VALUES (?, ?, ?)",
            [(r["match_id"], r["game_creation_time"], json.dumps({
                "info": {"gameStartTimestamp": r["game_creation_time"] + 1_000,
                         "gameEndTimestamp": r["game_creation_time"] + 30_000},
            })) for r in records],
        )
    cache = base / "splits.pkl"
    with cache.open("xb") as f:
        pickle.dump(splits, f, protocol=pickle.HIGHEST_PROTOCOL)
    manifest_path = base / "splits.split.json"
    manifest = build_manifest(splits, source_db=str(db), split_metadata=split_meta)
    with manifest_path.open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    ends, provenance = resolve_result_end_times(splits, db_path=str(db))
    end_path = base / "ends.json"
    write_result_end_times_artifact(end_path, splits, ends, provenance)
    return {"base": base, "records": records, "splits": splits,
            "db": db, "cache": cache, "manifest": manifest_path, "ends": end_path}
