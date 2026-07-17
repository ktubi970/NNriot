"""Realistic Match-V5 fixtures for analytics tests."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Iterable

from analytics.db import connect_analytics, initialize_schema, set_meta


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


def _patch_key(patch: str) -> tuple[int, int]:
    major, minor = patch.split(".", 1)
    return int(major), int(minor)


def seed_fact_rows(
    path: Path,
    patch_counts: dict[str, int],
    region_shifts: dict[str, float] | None = None,
) -> Path:
    """Seed deterministic, internally consistent analytical fact rows."""

    path = Path(path)
    initialize_schema(str(path))
    shifts = region_shifts or {"EUW": 10.0}
    regions = tuple(sorted(shifts))
    prefixes = {"EUW": "EUW1", "EUN": "EUN1", "NA": "NA1", "KR": "KR"}
    roles = ROLES
    global_index = 0

    with connect_analytics(str(path)) as conn:
        for patch in sorted(patch_counts, key=_patch_key):
            for _ in range(patch_counts[patch]):
                region = regions[global_index % len(regions)]
                shift = float(shifts[region])
                effective_shift = shift + ((global_index % 7) - 3) * 0.4
                match_id = f"{prefixes.get(region, 'OTHER')}_{global_index + 1}"
                winner_team = 100 if global_index % 2 == 0 else 200
                conn.execute(
                    "INSERT INTO match_facts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        match_id,
                        region,
                        prefixes.get(region, "OTHER"),
                        f"{patch}.123456",
                        patch,
                        420,
                        global_index + 1,
                        1_800,
                        winner_team,
                        10,
                        int(patch.split(".", 1)[0]),
                    ),
                )

                for team_id in (100, 200):
                    won = team_id == winner_team
                    signed_shift = effective_shift if won else 0.0
                    values = {
                        "kills": int(20 + signed_shift),
                        "deaths": int(20 - signed_shift if won else 20),
                        "assists": int(30 + signed_shift),
                        "gold": int(50_000 + signed_shift * 100),
                        "champion_damage": int(80_000 + signed_shift * 200),
                        "vision_score": 100.0 + signed_shift,
                        "lane_cs": int(700 + signed_shift),
                        "jungle_cs": int(150 + signed_shift),
                    }
                    objectives = {
                        "barons": 1 if won else 0,
                        "dragons": 3 if won else 1,
                        "towers": 8 if won else 3,
                        "inhibitors": 1 if won else 0,
                        "rift_heralds": 1 if won else 0,
                        "atakhan": global_index % 2,
                    }
                    firsts = {key: int(won) for key in (
                        "first_blood", "first_baron", "first_dragon",
                        "first_tower", "first_inhibitor"
                    )}
                    conn.execute(
                        """
                        INSERT INTO team_facts VALUES (
                            :match_id,:team_id,:side,:win,:kills,:deaths,:assists,
                            :gold,:champion_damage,:vision_score,:lane_cs,:jungle_cs,
                            :barons,:dragons,:towers,:inhibitors,:rift_heralds,:atakhan,
                            :first_blood,:first_baron,:first_dragon,:first_tower,
                            :first_inhibitor
                        )
                        """,
                        {
                            "match_id": match_id,
                            "team_id": team_id,
                            "side": "BLUE" if team_id == 100 else "RED",
                            "win": int(won),
                            **values,
                            **objectives,
                            **firsts,
                        },
                    )

                for player_index in range(10):
                    team_id = 100 if player_index < 5 else 200
                    won = team_id == winner_team
                    conn.execute(
                        "INSERT INTO player_facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            match_id,
                            f"player-{(global_index * 10 + player_index) % 300}",
                            team_id,
                            "BLUE" if team_id == 100 else "RED",
                            int(won),
                            f"Champion{player_index}",
                            roles[player_index % 5],
                            4 + int(won),
                            4 - int(won),
                            6 + int(won),
                            10_000 + int(won) * 200,
                            16_000 + int(won) * 300,
                            20.0 + int(won),
                            140 + int(won) * 2,
                            30 + int(won),
                        ),
                    )
                global_index += 1
        descriptor = json.dumps(
            {"patch_counts": patch_counts, "region_shifts": shifts},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        metadata = {
            "dataset_version": hashlib.sha256(descriptor).hexdigest(),
            "extractor_version": "1.0.0",
            "method_version": "1.0.0",
            "source_schema_version": "11",
            "source_match_count": str(sum(patch_counts.values())),
            "source_max_creation_ms": str(global_index),
            "quality_gate_passed": "true",
        }
        for key, value in metadata.items():
            set_meta(conn, key, value)
    return path
