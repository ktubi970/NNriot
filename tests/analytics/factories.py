"""Realistic Match-V5 fixtures for analytics tests."""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path
import sqlite3
from typing import Iterable


ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")


def classic_match(
    match_id: str = "EUW1_1",
    queue_id: int = 420,
    patch: str = "16.14",
    created_ms: int = 1,
) -> dict:
    """Return a complete ten-player CLASSIC Match-V5 payload."""

    participants = []
    for index in range(10):
        team_id = 100 if index < 5 else 200
        team_index = index % 5
        participants.append(
            {
                "puuid": f"puuid-{match_id}-{index}",
                "teamId": team_id,
                "win": team_id == 100,
                "championName": f"Champion{index}",
                "teamPosition": ROLES[team_index],
                "kills": 10 - team_index if team_id == 100 else 2 + team_index,
                "deaths": 2 + team_index if team_id == 100 else 10 - team_index,
                "assists": 5 + index,
                "goldEarned": 10_000 + index * 100,
                "totalDamageDealtToChampions": 20_000 + index * 250,
                "visionScore": 15.0 + index,
                "totalMinionsKilled": 150 + index,
                "neutralMinionsKilled": 20 + index,
            }
        )

    def objectives(winner: bool) -> dict:
        return {
            "champion": {"first": winner, "kills": 35 if winner else 15},
            "baron": {"first": winner, "kills": 2 if winner else 0},
            "dragon": {"first": winner, "kills": 4 if winner else 1},
            "tower": {"first": winner, "kills": 10 if winner else 3},
            "inhibitor": {"first": winner, "kills": 2 if winner else 0},
            "riftHerald": {"first": winner, "kills": 1 if winner else 0},
            "atakhan": {"first": winner, "kills": 1 if winner else 0},
        }

    return {
        "metadata": {"matchId": match_id},
        "info": {
            "gameMode": "CLASSIC",
            "queueId": queue_id,
            "gameVersion": f"{patch}.123456",
            "gameCreation": created_ms,
            "gameDuration": 1_800,
            "participants": participants,
            "teams": [
                {"teamId": 100, "win": True, "objectives": objectives(True)},
                {"teamId": 200, "win": False, "objectives": objectives(False)},
            ],
        },
    }


def _encoded(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(gzip.compress(raw)).decode("ascii")


def write_source_db(path: Path, matches: Iterable[dict]) -> Path:
    """Create or append to the operational subset consumed by analytics."""

    path = Path(path)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL)")
        if conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 0:
            conn.execute("INSERT INTO schema_version(version) VALUES (11)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS matches(
                match_id TEXT PRIMARY KEY,
                raw_json TEXT NOT NULL,
                game_mode TEXT,
                game_creation_time INTEGER,
                game_duration INTEGER,
                season_id INTEGER
            )
            """
        )
        for payload in matches:
            info = payload["info"]
            match_id = payload["metadata"]["matchId"]
            conn.execute(
                "INSERT OR REPLACE INTO matches VALUES (?,?,?,?,?,?)",
                (
                    match_id,
                    _encoded(payload),
                    info.get("gameMode"),
                    info.get("gameCreation"),
                    info.get("gameDuration"),
                    int(str(info["gameVersion"]).split(".", 1)[0]),
                ),
            )
    return path
