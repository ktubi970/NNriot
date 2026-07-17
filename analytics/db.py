"""SQLite analytical mart schema, metadata, and evidence cache."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Iterator


ANALYTICS_SCHEMA_VERSION = 1


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS analytics_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS match_facts (
    match_id TEXT PRIMARY KEY,
    region TEXT NOT NULL,
    platform_prefix TEXT NOT NULL,
    game_version TEXT NOT NULL,
    patch TEXT NOT NULL,
    queue_id INTEGER,
    game_creation_ms INTEGER NOT NULL,
    game_duration_s INTEGER NOT NULL,
    winner_team_id INTEGER NOT NULL CHECK (winner_team_id IN (100, 200)),
    participant_count INTEGER NOT NULL CHECK (participant_count = 10),
    source_season_id INTEGER
);

CREATE TABLE IF NOT EXISTS team_facts (
    match_id TEXT NOT NULL REFERENCES match_facts(match_id) ON DELETE CASCADE,
    team_id INTEGER NOT NULL CHECK (team_id IN (100, 200)),
    side TEXT NOT NULL CHECK (side IN ('BLUE', 'RED')),
    win INTEGER NOT NULL CHECK (win IN (0, 1)),
    kills INTEGER NOT NULL,
    deaths INTEGER NOT NULL,
    assists INTEGER NOT NULL,
    gold INTEGER NOT NULL,
    champion_damage INTEGER NOT NULL,
    vision_score REAL NOT NULL,
    lane_cs INTEGER NOT NULL,
    jungle_cs INTEGER NOT NULL,
    barons INTEGER NOT NULL,
    dragons INTEGER NOT NULL,
    towers INTEGER NOT NULL,
    inhibitors INTEGER NOT NULL,
    rift_heralds INTEGER NOT NULL,
    atakhan INTEGER NOT NULL,
    first_blood INTEGER NOT NULL CHECK (first_blood IN (0, 1)),
    first_baron INTEGER NOT NULL CHECK (first_baron IN (0, 1)),
    first_dragon INTEGER NOT NULL CHECK (first_dragon IN (0, 1)),
    first_tower INTEGER NOT NULL CHECK (first_tower IN (0, 1)),
    first_inhibitor INTEGER NOT NULL CHECK (first_inhibitor IN (0, 1)),
    PRIMARY KEY (match_id, team_id)
);

CREATE TABLE IF NOT EXISTS player_facts (
    match_id TEXT NOT NULL REFERENCES match_facts(match_id) ON DELETE CASCADE,
    puuid TEXT NOT NULL,
    team_id INTEGER NOT NULL CHECK (team_id IN (100, 200)),
    side TEXT NOT NULL CHECK (side IN ('BLUE', 'RED')),
    win INTEGER NOT NULL CHECK (win IN (0, 1)),
    champion TEXT NOT NULL,
    role TEXT NOT NULL,
    kills INTEGER NOT NULL,
    deaths INTEGER NOT NULL,
    assists INTEGER NOT NULL,
    gold INTEGER NOT NULL,
    champion_damage INTEGER NOT NULL,
    vision_score REAL NOT NULL,
    lane_cs INTEGER NOT NULL,
    jungle_cs INTEGER NOT NULL,
    PRIMARY KEY (match_id, puuid)
);

CREATE TABLE IF NOT EXISTS extraction_issues (
    match_id TEXT NOT NULL,
    issue_code TEXT NOT NULL,
    field TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    PRIMARY KEY (match_id, issue_code, field, extractor_version)
);

CREATE TABLE IF NOT EXISTS evidence_cache (
    evidence_id TEXT PRIMARY KEY,
    dataset_version TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    method_version TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (dataset_version, request_hash, method_version)
);

CREATE INDEX IF NOT EXISTS idx_match_patch_region_queue
    ON match_facts(patch, region, queue_id);
CREATE INDEX IF NOT EXISTS idx_match_creation ON match_facts(game_creation_ms);
CREATE INDEX IF NOT EXISTS idx_match_region ON match_facts(region);
CREATE INDEX IF NOT EXISTS idx_team_match_win ON team_facts(match_id, win);
CREATE INDEX IF NOT EXISTS idx_player_puuid ON player_facts(puuid);
CREATE INDEX IF NOT EXISTS idx_player_champion_role
    ON player_facts(champion, role);
CREATE INDEX IF NOT EXISTS idx_evidence_lookup
    ON evidence_cache(dataset_version, request_hash, method_version);
"""


@contextmanager
def connect_analytics(
    path: str, readonly: bool = False
) -> Iterator[sqlite3.Connection]:
    """Open the mart with transactions for writes and query-only reads."""

    if readonly:
        uri = Path(path).resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.execute("PRAGMA query_only=ON")
    else:
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        if not readonly:
            conn.commit()
    except Exception:
        if not readonly:
            conn.rollback()
        raise
    finally:
        conn.close()


def initialize_schema(path: str) -> None:
    """Create or validate the version-one analytical schema."""

    with connect_analytics(path) as conn:
        conn.executescript(SCHEMA_SQL)
        set_meta(conn, "schema_version", str(ANALYTICS_SCHEMA_VERSION))


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO analytics_meta(key,value) VALUES (?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )


def get_meta(path: str, key: str) -> str | None:
    with connect_analytics(path, readonly=True) as conn:
        row = conn.execute(
            "SELECT value FROM analytics_meta WHERE key=?", (key,)
        ).fetchone()
    return str(row[0]) if row else None


def all_meta(path: str) -> dict[str, str]:
    with connect_analytics(path, readonly=True) as conn:
        rows = conn.execute(
            "SELECT key,value FROM analytics_meta ORDER BY key"
        ).fetchall()
    return {str(row["key"]): str(row["value"]) for row in rows}


class EvidenceRepository:
    """Version-scoped immutable evidence JSON cache."""

    def __init__(self, path: str):
        self.path = path

    def get(
        self, dataset_version: str, request_hash: str, method_version: str
    ) -> dict | None:
        with connect_analytics(self.path, readonly=True) as conn:
            row = conn.execute(
                "SELECT evidence_json FROM evidence_cache "
                "WHERE dataset_version=? AND request_hash=? AND method_version=?",
                (dataset_version, request_hash, method_version),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def get_by_id(self, evidence_id: str) -> dict | None:
        with connect_analytics(self.path, readonly=True) as conn:
            row = conn.execute(
                "SELECT evidence_json FROM evidence_cache WHERE evidence_id=?",
                (evidence_id,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(
        self,
        evidence_id: str,
        dataset_version: str,
        request_hash: str,
        method_version: str,
        packet: dict,
    ) -> None:
        payload = json.dumps(
            packet,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with connect_analytics(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO evidence_cache "
                "(evidence_id,dataset_version,request_hash,method_version,"
                "evidence_json,created_at) "
                "VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)",
                (
                    evidence_id,
                    dataset_version,
                    request_hash,
                    method_version,
                    payload,
                ),
            )
