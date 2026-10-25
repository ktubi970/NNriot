"""
database.py - SQLite database management for NNriot

Provides connection management, schema versioning/migrations, batch inserts,
indexed queries, and comprehensive error handling for the NNriot training pipeline.
"""

import sqlite3
import json
import gzip
import base64
import os
import time
import logging
import datetime as _dt
from contextlib import closing, contextmanager
from pathlib import Path

from core import config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Use environment variable or fall back to a relative path
DB_PATH = config.DB_PATH

# Current schema version – bump this whenever you add a migration below
SCHEMA_VERSION = 16

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
# NOTE: basicConfig is intentionally NOT called here.
# Root logger configuration is the responsibility of the application entry-point.

# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------


@contextmanager
def get_connection(db_path: str = DB_PATH):
    """
    Context-manager that opens a SQLite connection, configures it with
    sensible pragmas, and guarantees commit-on-success / rollback-on-error
    and unconditional close.

    Usage::

        with get_connection() as conn:
            conn.execute("SELECT 1")
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # dict-like row access
    conn.execute("PRAGMA journal_mode=WAL")  # better concurrency
    conn.execute("PRAGMA foreign_keys=ON")  # enforce FK constraints
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema initialisation & migrations
# ---------------------------------------------------------------------------

# Each entry is (from_version, sql_statements_list).
# Migrations are applied in order until the DB reaches SCHEMA_VERSION.
_MIGRATIONS = [
    # v0 → v1 : initial schema
    (
        0,
        [
            """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL
        )
        """,
            "INSERT INTO schema_version (version) VALUES (1)",
            """
        CREATE TABLE IF NOT EXISTS players (
            puuid       TEXT PRIMARY KEY,
            game_name   TEXT,
            tag_line    TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
            """
        CREATE TABLE IF NOT EXISTS matches (
            match_id           TEXT PRIMARY KEY,
            raw_json           TEXT,
            game_mode          TEXT,
            game_creation_time INTEGER,
            game_duration      INTEGER
        )
        """,
            """
        CREATE TABLE IF NOT EXISTS training_dataset (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id      TEXT UNIQUE,
            feature_json  TEXT,   -- Aggregated matchup JSON
            winner_label  INTEGER, -- 0 for Team A, 1 for Team B
            FOREIGN KEY(match_id) REFERENCES matches(match_id)
        )
        """,
        ],
    ),
    # v1 → v2 : add performance indexes
    (
        1,
        [
            "CREATE INDEX IF NOT EXISTS idx_matches_game_mode         ON matches(game_mode)",
            "CREATE INDEX IF NOT EXISTS idx_matches_game_creation     ON matches(game_creation_time)",
            "CREATE INDEX IF NOT EXISTS idx_training_winner           ON training_dataset(winner_label)",
            "UPDATE schema_version SET version = 2",
        ],
    ),
    # v2 → v3 : add training status tagging
    (
        2,
        [
            "ALTER TABLE training_dataset ADD COLUMN is_trained INTEGER DEFAULT 0",
            "ALTER TABLE training_dataset ADD COLUMN last_trained_at TIMESTAMP",
            "CREATE INDEX IF NOT EXISTS idx_training_is_trained ON training_dataset(is_trained)",
            "UPDATE schema_version SET version = 3",
        ],
    ),
    # v3 -> v4 : add player performance metrics
    (
        3,
        [
            "ALTER TABLE players ADD COLUMN avg_kda REAL DEFAULT 0.0",
            "ALTER TABLE players ADD COLUMN avg_gold INTEGER DEFAULT 0",
            "ALTER TABLE players ADD COLUMN total_matches INTEGER DEFAULT 0",
            "UPDATE schema_version SET version = 4",
        ],
    ),
    # v4 -> v5 : include LLM-generated annotations
    (
        4,
        [
            "ALTER TABLE training_dataset ADD COLUMN llm_summary TEXT",
            "ALTER TABLE training_dataset ADD COLUMN llm_tags TEXT",
            "ALTER TABLE training_dataset ADD COLUMN llm_embedding BLOB",
            "UPDATE schema_version SET version = 5",
        ],
    ),
    # v5 -> v6 : add champions registry
    (
        5,
        [
            "CREATE TABLE IF NOT EXISTS champions (name TEXT PRIMARY KEY)",
            "CREATE INDEX IF NOT EXISTS idx_players_total_matches ON players(total_matches)",
            "UPDATE schema_version SET version = 6",
        ],
    ),
    # v6 -> v7 : add persistent crawl queue
    (
        6,
        [
            """
        CREATE TABLE IF NOT EXISTS crawl_queue (
            game_name TEXT,
            tag_line  TEXT,
            priority  INTEGER DEFAULT 0,
            is_processed INTEGER DEFAULT 0,
            added_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(game_name, tag_line)
        )
        """,
            "CREATE INDEX IF NOT EXISTS idx_crawl_processed ON crawl_queue(is_processed)",
            "UPDATE schema_version SET version = 7",
        ],
    ),
    # v7 -> v8 : add player aliases
    (
        7,
        [
            """
        CREATE TABLE IF NOT EXISTS player_aliases (
            alias_puuid TEXT PRIMARY KEY,
            canonical_puuid TEXT,
            FOREIGN KEY(alias_puuid) REFERENCES players(puuid),
            FOREIGN KEY(canonical_puuid) REFERENCES players(puuid)
        )
        """,
            "UPDATE schema_version SET version = 8",
        ],
    ),
    # v8 -> v9 : add season_id to matches
    (
        8,
        [
            "ALTER TABLE matches ADD COLUMN season_id INTEGER",
            "CREATE INDEX IF NOT EXISTS idx_matches_season_id ON matches(season_id)",
            "UPDATE schema_version SET version = 9",
        ],
    ),
    # v9 -> v10 : add multi-output labels JSON to training_dataset
    (
        9,
        [
            "ALTER TABLE training_dataset ADD COLUMN labels_json TEXT",
            "UPDATE schema_version SET version = 10",
        ],
    ),
    # v10 -> v11 : add match_timelines table for Riot Match-V5 timeline data
    (
        10,
        [
            """
            CREATE TABLE IF NOT EXISTS match_timelines (
                match_id      TEXT PRIMARY KEY,
                timeline_json TEXT,
                FOREIGN KEY(match_id) REFERENCES matches(match_id)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_timelines_match ON match_timelines(match_id)",
            "UPDATE schema_version SET version = 11",
        ],
    ),
    # v11 -> v12 : add match_participants table for fast indexed player history
    (
        11,
        [
            """
            CREATE TABLE IF NOT EXISTS match_participants (
                match_id           TEXT NOT NULL,
                puuid              TEXT NOT NULL,
                champion_name      TEXT,
                team_position      TEXT,
                win                INTEGER,
                kills              INTEGER,
                deaths             INTEGER,
                assists            INTEGER,
                total_cs           INTEGER,
                gold_earned        INTEGER,
                game_creation_time INTEGER,
                game_duration      INTEGER,
                game_mode          TEXT,
                items_json         TEXT,
                PRIMARY KEY (match_id, puuid),
                FOREIGN KEY(match_id) REFERENCES matches(match_id)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_match_participants_puuid ON match_participants(puuid, game_creation_time DESC)",
            "CREATE INDEX IF NOT EXISTS idx_match_participants_match ON match_participants(match_id)",
            "UPDATE schema_version SET version = 12",
        ],
    ),
    # v12 -> v13 : add player_spatial_embeddings for trajectory GRU vectors
    (
        12,
        [
            """
            CREATE TABLE IF NOT EXISTS player_spatial_embeddings (
                match_id           TEXT NOT NULL,
                puuid              TEXT NOT NULL,
                game_creation_time INTEGER NOT NULL,
                embedding_blob     BLOB NOT NULL,
                PRIMARY KEY (match_id, puuid),
                FOREIGN KEY(match_id) REFERENCES matches(match_id)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_spatial_embeddings_lookup ON player_spatial_embeddings(puuid, game_creation_time DESC)",
            "UPDATE schema_version SET version = 13",
        ],
    ),
    # v13 -> v14 : migrate crawl_queue to support PUUIDs, region, and auto-increment ID
    (
        13,
        [
            """
            CREATE TABLE IF NOT EXISTS crawl_queue_v2 (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                puuid        TEXT UNIQUE,
                region       TEXT NOT NULL,
                game_name    TEXT,
                tag_line     TEXT,
                priority     INTEGER DEFAULT 0,
                is_processed INTEGER DEFAULT 0,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP
            )
            """,
            """
            INSERT INTO crawl_queue_v2 (game_name, tag_line, priority, is_processed, created_at, region)
            SELECT game_name, tag_line, priority, is_processed, added_at, 'EUW'
            FROM crawl_queue
            """,
            "DROP TABLE IF EXISTS crawl_queue",
            "ALTER TABLE crawl_queue_v2 RENAME TO crawl_queue",
            "CREATE INDEX IF NOT EXISTS idx_crawl_queue_lookup ON crawl_queue(is_processed, priority DESC, id ASC)",
            "CREATE INDEX IF NOT EXISTS idx_crawl_queue_puuid ON crawl_queue(puuid)",
            "UPDATE schema_version SET version = 14",
        ],
    ),
    # v14 -> v15 : record which pinned encoder produced each spatial embedding
    # (NULL = unknown provenance, written before the encoder weights were committed)
    (
        14,
        [
            "ALTER TABLE player_spatial_embeddings ADD COLUMN encoder_id TEXT",
            "UPDATE schema_version SET version = 15",
        ],
    ),
    # v15 -> v16 : record when each match became available (UTC epoch ms) and
    # whose match list returned it (NULL = ingested before v16, unknown)
    (
        15,
        [
            "ALTER TABLE matches ADD COLUMN ingested_at INTEGER",
            "ALTER TABLE matches ADD COLUMN fetched_by_puuid TEXT",
            "UPDATE schema_version SET version = 16",
        ],
    ),
]


def _get_schema_version(conn) -> int:
    """Return the current schema version stored in the DB (0 if not set)."""
    try:
        # Use MAX to be safe against accidental duplicate rows
        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    except sqlite3.OperationalError:
        # Table doesn't exist yet
        return 0


def migrate_db(db_path: str = DB_PATH):
    """
    Apply any pending schema migrations in order.
    Safe to call multiple times – only missing migrations are applied.
    """
    with get_connection(db_path) as conn:
        current = _get_schema_version(conn)
        if current >= SCHEMA_VERSION:
            logger.debug("Schema is up-to-date (version %d).", current)
            return

        for from_version, statements in _MIGRATIONS:
            if current == from_version:
                logger.info(
                    "Applying migration v%d → v%d …",
                    from_version,
                    from_version + 1,
                )
                for sql in statements:
                    conn.execute(sql)
                current += 1

        logger.info("Database schema migrated to version %d.", current)


def init_db(db_path: str = DB_PATH):
    """
    Initialise the database and run all pending migrations.
    This is the single entry-point that callers should use.
    """
    migrate_db(db_path)
    logger.info("Database ready at '%s'.", db_path)


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def save_player(puuid: str, name: str, tag: str, db_path: str = DB_PATH):
    """Insert or replace a player record."""
    if not puuid:
        raise ValueError("puuid must not be empty.")
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO players (puuid, game_name, tag_line, last_updated)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(puuid) DO UPDATE SET
                game_name    = excluded.game_name,
                tag_line     = excluded.tag_line,
                last_updated = excluded.last_updated
            """,
            (puuid, name, tag),
        )
    logger.debug("Saved player %s#%s (%s).", name, tag, puuid)


def increment_player_matches(puuid: str, count: int, db_path: str = DB_PATH):
    """Increment the total_matches counter for a specific player."""
    if count <= 0:
        return
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE players SET total_matches = total_matches + ? WHERE puuid = ?",
            (count, puuid),
        )


def _now_ms() -> int:
    return int(time.time() * 1000)


def _fetching_participant(info: dict, fetched_by_puuid: str | None) -> str | None:
    """The anchor is recorded only if it played the match, as Riot's match lists guarantee."""
    if not fetched_by_puuid:
        return None
    participants = info.get("participants", [])
    if any(isinstance(p, dict) and p.get("puuid") == fetched_by_puuid for p in participants):
        return fetched_by_puuid
    return None


def save_match(
    match_id: str,
    raw_data: dict,
    db_path: str = DB_PATH,
    *,
    fetched_by_puuid: str | None = None,
    ingested_at: int | None = None,
):
    """Insert a match record (silently ignored if it already exists).

    ``ingested_at`` (UTC epoch ms, default: now) is when the row became available;
    the first insert wins. ``fetched_by_puuid`` is the anchor whose match list returned it.
    """
    if not match_id:
        raise ValueError("match_id must not be empty.")
    info = raw_data.get("info", {})
    gv = info.get("gameVersion")
    season_id = None
    if gv:
        try:
            season_id = int(gv.split(".")[0])
        except (ValueError, IndexError):
            pass

    compressed_json = base64.b64encode(
        gzip.compress(json.dumps(raw_data).encode("utf-8"))
    ).decode("ascii")
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO matches
                (match_id, raw_json, game_mode, game_creation_time, game_duration, season_id,
                 ingested_at, fetched_by_puuid)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                match_id,
                compressed_json,
                info.get("gameMode"),
                info.get("gameCreation"),
                info.get("gameDuration"),
                season_id,
                _now_ms() if ingested_at is None else ingested_at,
                _fetching_participant(info, fetched_by_puuid),
            ),
        )

        # Update champion registry and participant records
        participants = info.get("participants", [])
        p_rows = []
        for p in participants:
            c_name = p.get("championName")
            if c_name:
                conn.execute(
                    "INSERT OR IGNORE INTO champions (name) VALUES (?)", (c_name,)
                )
            p_puuid = p.get("puuid")
            if p_puuid:
                items = [p.get(f"item{i}", 0) for i in range(7)]
                p_rows.append(
                    (
                        match_id,
                        p_puuid,
                        c_name or "Unknown",
                        p.get("teamPosition") or p.get("individualPosition", "UNKNOWN"),
                        1 if p.get("win") else 0,
                        p.get("kills", 0),
                        p.get("deaths", 0),
                        p.get("assists", 0),
                        p.get("totalMinionsKilled", 0) + p.get("neutralMinionsKilled", 0),
                        p.get("goldEarned", 0),
                        info.get("gameCreation"),
                        info.get("gameDuration"),
                        info.get("gameMode", "CLASSIC"),
                        json.dumps(items),
                    )
                )

        if p_rows:
            try:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO match_participants
                        (match_id, puuid, champion_name, team_position, win, kills, deaths, assists,
                         total_cs, gold_earned, game_creation_time, game_duration, game_mode, items_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    p_rows,
                )
            except sqlite3.OperationalError:
                pass

    logger.debug("Saved match %s.", match_id)


def save_matches_batch(
    matches: list[tuple],
    db_path: str = DB_PATH,
    *,
    fetched_by_puuid: str | None = None,
    ingested_at: int | None = None,
):
    """
    Bulk-insert multiple matches and their champions in a single transaction.
    Each element of *matches* should be a (match_id, raw_data_dict).
    ``ingested_at`` and ``fetched_by_puuid`` apply to every new row, as in save_match.
    """
    ingested_at = _now_ms() if ingested_at is None else ingested_at
    rows = []
    unique_champions = set()
    all_p_rows = []
    for match_id, raw_data in matches:
        if not match_id:
            continue
        info = raw_data.get("info", {})
        gv = info.get("gameVersion")
        season_id = None
        if gv:
            try:
                season_id = int(gv.split(".")[0])
            except (ValueError, IndexError):
                pass

        compressed_json = base64.b64encode(
            gzip.compress(json.dumps(raw_data).encode("utf-8"))
        ).decode("ascii")
        rows.append(
            (
                match_id,
                compressed_json,
                info.get("gameMode"),
                info.get("gameCreation"),
                info.get("gameDuration"),
                season_id,
                ingested_at,
                _fetching_participant(info, fetched_by_puuid),
            )
        )

        # Collect champions and participants for batch insert
        for p in info.get("participants", []):
            c_name = p.get("championName")
            if c_name:
                unique_champions.add(c_name)
            p_puuid = p.get("puuid")
            if p_puuid:
                items = [p.get(f"item{i}", 0) for i in range(7)]
                all_p_rows.append(
                    (
                        match_id,
                        p_puuid,
                        c_name or "Unknown",
                        p.get("teamPosition") or p.get("individualPosition", "UNKNOWN"),
                        1 if p.get("win") else 0,
                        p.get("kills", 0),
                        p.get("deaths", 0),
                        p.get("assists", 0),
                        p.get("totalMinionsKilled", 0) + p.get("neutralMinionsKilled", 0),
                        p.get("goldEarned", 0),
                        info.get("gameCreation"),
                        info.get("gameDuration"),
                        info.get("gameMode", "CLASSIC"),
                        json.dumps(items),
                    )
                )

    if not rows:
        return

    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO matches
                (match_id, raw_json, game_mode, game_creation_time, game_duration, season_id,
                 ingested_at, fetched_by_puuid)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        if unique_champions:
            conn.executemany(
                "INSERT OR IGNORE INTO champions (name) VALUES (?)",
                [(c,) for c in unique_champions],
            )
        if all_p_rows:
            try:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO match_participants
                        (match_id, puuid, champion_name, team_position, win, kills, deaths, assists,
                         total_cs, gold_earned, game_creation_time, game_duration, game_mode, items_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    all_p_rows,
                )
            except sqlite3.OperationalError:
                pass

    logger.debug("Batch-saved %d matches and updated champions.", len(rows))


def save_match_timeline(match_id: str, timeline_data: dict, db_path: str = DB_PATH):
    """Insert a timeline record (silently ignored if already exists)."""
    if not match_id:
        raise ValueError("match_id must not be empty.")
    compressed = base64.b64encode(
        gzip.compress(json.dumps(timeline_data).encode("utf-8"))
    ).decode("ascii")
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO match_timelines (match_id, timeline_json) VALUES (?, ?)",
            (match_id, compressed),
        )
    logger.debug("Saved timeline %s.", match_id)


def save_match_timelines_batch(timelines: list[tuple], db_path: str = DB_PATH):
    """Bulk-insert multiple (match_id, timeline_data) tuples."""
    rows = []
    for match_id, timeline_data in timelines:
        if not match_id:
            continue
        compressed = base64.b64encode(
            gzip.compress(json.dumps(timeline_data).encode("utf-8"))
        ).decode("ascii")
        rows.append((match_id, compressed))
    if not rows:
        return
    with get_connection(db_path) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO match_timelines (match_id, timeline_json) VALUES (?, ?)",
            rows,
        )
    logger.debug("Batch-saved %d timelines.", len(rows))


def get_match_timeline(match_id: str, db_path: str = DB_PATH) -> dict | None:
    """Return the parsed timeline dict for a match, or None if not present."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT timeline_json FROM match_timelines WHERE match_id = ? LIMIT 1",
            (match_id,),
        ).fetchone()
    if not row:
        return None
    tj = row["timeline_json"]
    if not tj:
        return None
    return json.loads(gzip.decompress(base64.b64decode(tj)).decode("utf-8"))


def match_timeline_exists(match_id: str, db_path: str = DB_PATH) -> bool:
    """True if a timeline is stored for this match."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM match_timelines WHERE match_id = ? LIMIT 1", (match_id,)
        ).fetchone()
    return row is not None


def save_player_spatial_embedding(
    match_id: str,
    puuid: str,
    game_creation_time: int,
    embedding: bytes,
    encoder_id: str,
    db_path: str = DB_PATH,
) -> None:
    """Save a player's spatial trajectory embedding blob and the encoder that produced it."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO player_spatial_embeddings
            (match_id, puuid, game_creation_time, embedding_blob, encoder_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (match_id, puuid, int(game_creation_time), sqlite3.Binary(embedding), encoder_id),
        )


def get_player_recent_spatial_embeddings(
    puuid: str,
    before_timestamp: int | None = None,
    limit: int = 50,
    db_path: str = DB_PATH,
) -> list[bytes]:
    """Retrieve up to `limit` spatial embeddings for a player strictly before `before_timestamp`."""
    with get_connection(db_path) as conn:
        if before_timestamp is not None:
            rows = conn.execute(
                """
                SELECT embedding_blob FROM player_spatial_embeddings
                WHERE puuid = ? AND game_creation_time < ?
                ORDER BY game_creation_time DESC
                LIMIT ?
                """,
                (puuid, int(before_timestamp), int(limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT embedding_blob FROM player_spatial_embeddings
                WHERE puuid = ?
                ORDER BY game_creation_time DESC
                LIMIT ?
                """,
                (puuid, int(limit)),
            ).fetchall()
    return [bytes(row["embedding_blob"]) for row in rows]


def _compress_labels(labels: dict | None) -> str | None:
    """Serialise+gzip+b64 encode a labels dict, or return None if labels is None."""
    if labels is None:
        return None
    return base64.b64encode(
        gzip.compress(json.dumps(labels).encode("utf-8"))
    ).decode("ascii")


def save_training_record(
    match_id: str,
    feature_json: dict,
    winner_label: int,
    db_path: str = DB_PATH,
    labels: dict | None = None,
):
    """
    Insert a training record (silently ignored if it already exists).

    Parameters
    ----------
    labels:
        Optional dict of multi-output labels (see ``feature_labels.extract_labels``).
        Stored gzipped+b64 in the ``labels_json`` column.  ``None`` stores SQL NULL.
    """
    if not match_id:
        raise ValueError("match_id must not be empty.")
    if winner_label not in (0, 1):
        raise ValueError("winner_label must be 0 or 1.")
    compressed_feature = base64.b64encode(
        gzip.compress(json.dumps(feature_json).encode("utf-8"))
    ).decode("ascii")
    compressed_labels = _compress_labels(labels)
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO training_dataset
                (match_id, feature_json, winner_label, labels_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                match_id,
                compressed_feature,
                winner_label,
                compressed_labels,
            ),
        )
    logger.debug("Saved training record for match %s.", match_id)


def save_training_records_batch(
    records: list[tuple],
    db_path: str = DB_PATH,
):
    """
    Bulk-insert multiple training records in a single transaction.

    Each element of *records* may be either:
    - a 3-tuple ``(match_id, feature_json_dict, winner_label)`` -- labels_json NULL.
    - a 4-tuple ``(match_id, feature_json_dict, winner_label, labels_dict)``
      where ``labels_dict`` is the output of ``feature_labels.extract_labels``
      (``None`` allowed).
    """
    rows = []
    for rec in records:
        if len(rec) == 4:
            match_id, feature_json, winner_label, labels = rec
        elif len(rec) == 3:
            match_id, feature_json, winner_label = rec
            labels = None
        else:
            logger.warning(
                "Skipping malformed training record tuple of length %d.", len(rec)
            )
            continue
        if not match_id:
            logger.warning("Skipping training record with empty match_id.")
            continue
        if winner_label not in (0, 1):
            logger.warning(
                "Skipping record %s: invalid winner_label %s.", match_id, winner_label
            )
            continue
        compressed_feature = base64.b64encode(
            gzip.compress(json.dumps(feature_json).encode("utf-8"))
        ).decode("ascii")
        compressed_labels = _compress_labels(labels)
        rows.append(
            (
                match_id,
                compressed_feature,
                winner_label,
                compressed_labels,
            )
        )

    if not rows:
        return

    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO training_dataset
                (match_id, feature_json, winner_label, labels_json)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
    logger.info("Batch-inserted %d training records.", len(rows))


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def match_exists(match_id: str, db_path: str = DB_PATH) -> bool:
    """Return True if *match_id* is already stored in the matches table."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM matches WHERE match_id = ? LIMIT 1", (match_id,)
        ).fetchone()
    return row is not None


def get_training_batch(
    limit: int | None = None,
    offset: int = 0,
    db_path: str = DB_PATH,
) -> tuple[list[dict], list[list[float]]]:
    """
    Return ``(features, labels)`` ready for model training.

    Parameters
    ----------
    limit:
        Maximum number of rows to return.  ``None`` returns all rows.
    offset:
        Number of rows to skip (useful for pagination).

    Returns
    -------
    features : list[dict]
        Deserialized feature dicts.
    labels : list[list[float]]
        One-hot encoded labels, e.g. ``[1.0, 0.0]`` for Team A win.
    """
    sql = "SELECT feature_json, winner_label FROM training_dataset ORDER BY id"
    params: list = []
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params = [limit, offset]
    elif offset > 0:
        # SQLite supports LIMIT -1 to mean "no limit" while still honouring OFFSET
        sql += " LIMIT -1 OFFSET ?"
        params = [offset]

    with get_connection(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()

    features = []
    for r in rows:
        rj = r["feature_json"]
        if rj.startswith("{") or rj.startswith("["):
            features.append(json.loads(rj))
        else:
            features.append(
                json.loads(gzip.decompress(base64.b64decode(rj)).decode("utf-8"))
            )

    labels: list[list[float]] = []
    for r in rows:
        label = [0.0, 0.0]
        label[r["winner_label"]] = 1.0  # one-hot encode
        labels.append(label)

    return features, labels


def get_recent_training_records(
    limit: int = 50,
    offset: int = 0,
    db_path: str = DB_PATH,
) -> list[dict]:
    """
    Return recent training records with their features and winner labels.
    Used for the Match Explorer UI.
    """
    sql = (
        "SELECT id, match_id, feature_json, winner_label "
        "FROM training_dataset ORDER BY id DESC LIMIT ? OFFSET ?"
    )

    with get_connection(db_path) as conn:
        rows = conn.execute(sql, [limit, offset]).fetchall()

    records = []
    for r in rows:
        rj = r["feature_json"]
        if rj.startswith("{") or rj.startswith("["):
            feat = json.loads(rj)
        else:
            feat = json.loads(gzip.decompress(base64.b64decode(rj)).decode("utf-8"))

        rec = {
            "id": r["id"],
            "match_id": r["match_id"],
            "features": feat,
            "winner_label": r["winner_label"],
        }
        records.append(rec)

    return records


def get_matches_by_mode(
    game_mode: str,
    limit: int = 100,
    db_path: str = DB_PATH,
) -> list[dict]:
    """
    Return raw match dicts filtered by *game_mode* (e.g. ``"CLASSIC"``).
    Results are ordered by creation time descending (newest first).
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT match_id, raw_json, game_mode, game_creation_time, game_duration
            FROM   matches
            WHERE  game_mode = ?
            ORDER  BY game_creation_time DESC
            LIMIT  ?
            """,
            (game_mode, limit),
        ).fetchall()

    res = []
    for r in rows:
        d = dict(r)
        rj = d["raw_json"]
        if rj.startswith("{") or rj.startswith("["):
            pass  # Keep as is, though usually returning dict means returning raw_json parsed
        else:
            d["raw_json"] = gzip.decompress(base64.b64decode(rj)).decode("utf-8")
        res.append(d)
    return res


def search_players(query: str, limit: int = 10, db_path: str = DB_PATH) -> list[dict]:
    """
    Search for players by game_name or tag_line.
    """
    # Escape SQLite LIKE special characters (% and _) to prevent injection
    safe = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT puuid, game_name, tag_line, total_matches, avg_kda, avg_gold
            FROM players
            WHERE game_name LIKE ? ESCAPE '\\' OR tag_line LIKE ? ESCAPE '\\'
            ORDER BY total_matches DESC
            LIMIT ?
            """,
            (f"%{safe}%", f"%{safe}%", limit),
        ).fetchall()

    results = []
    for r in rows:
        results.append(
            {
                "puuid": r["puuid"],
                "game_name": r["game_name"],
                "tag_line": r["tag_line"],
                "total_matches": r["total_matches"] or 0,
                "avg_kda": r["avg_kda"] or 0.0,
                "avg_gold": r["avg_gold"] or 0,
            }
        )
    return results


def get_player_stats(puuid: str, db_path: str = DB_PATH) -> dict | None:
    """
    Get historical stats for a specific player, automatically aggregating smurfs/aliases.
    """
    with get_connection(db_path) as conn:
        cannon_row = conn.execute(
            "SELECT canonical_puuid FROM player_aliases WHERE alias_puuid = ?", (puuid,)
        ).fetchone()

        target_puuid = cannon_row["canonical_puuid"] if cannon_row else puuid

        puuids_to_check = [target_puuid]
        alias_rows = conn.execute(
            "SELECT alias_puuid FROM player_aliases WHERE canonical_puuid = ?",
            (target_puuid,),
        ).fetchall()
        for ar in alias_rows:
            puuids_to_check.append(ar["alias_puuid"])

        placeholders = ",".join(["?"] * len(puuids_to_check))
        rows = conn.execute(
            f"SELECT puuid, game_name, tag_line, avg_kda, avg_gold, total_matches FROM players WHERE puuid IN ({placeholders})",
            tuple(puuids_to_check),
        ).fetchall()

    if not rows:
        return None

    total_matches = 0
    sum_kda = 0.0
    sum_gold = 0.0

    main_row = next((r for r in rows if r["puuid"] == target_puuid), rows[0])

    for r in rows:
        m = r["total_matches"] or 0
        total_matches += m
        sum_kda += (r["avg_kda"] or 0.0) * m
        sum_gold += (r["avg_gold"] or 0) * m

    avg_kda = sum_kda / total_matches if total_matches > 0 else 0.0
    avg_gold = int(sum_gold / total_matches) if total_matches > 0 else 0

    return {
        "puuid": main_row["puuid"],
        "game_name": main_row["game_name"],
        "tag_line": main_row["tag_line"],
        "avg_kda": avg_kda,
        "avg_gold": avg_gold,
        "total_matches": total_matches,
    }


def get_player_by_name_tag(
    game_name: str, tag_line: str, db_path: str = DB_PATH
) -> dict | None:
    """Retrieve player by exact (case-insensitive) game_name and tag_line."""
    if not game_name or not tag_line:
        return None
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT puuid, game_name, tag_line, total_matches, avg_kda, avg_gold
            FROM players
            WHERE LOWER(game_name) = LOWER(?) AND LOWER(tag_line) = LOWER(?)
            LIMIT 1
            """,
            (game_name.strip(), tag_line.strip()),
        ).fetchone()
    if row:
        return dict(row)
    return None


def get_player_match_history(
    puuid: str, limit: int = 10, db_path: str = DB_PATH
) -> list[dict]:
    """Read indexed player history without changing or repairing the database.

    Legacy raw-only matches require explicit backfill_match_participants first.
    Missing schema is a setup error and is deliberately not hidden.
    """
    if not puuid:
        return []
    matches = []
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        p_rows = conn.execute(
            """
            SELECT match_id, champion_name, team_position, win, kills, deaths, assists,
                   total_cs, gold_earned, game_creation_time, game_duration, game_mode, items_json
            FROM match_participants
            WHERE puuid = ?
            ORDER BY game_creation_time DESC
            LIMIT ?
            """,
            (puuid, limit),
        ).fetchall()

        for r in p_rows:
            dur_secs = r["game_duration"] or 0
            dur_mins = max(1, dur_secs // 60)
            kills = r["kills"] or 0
            deaths = r["deaths"] or 0
            assists = r["assists"] or 0
            kda = round((kills + assists) / max(1, deaths), 2)
            cs = r["total_cs"] or 0
            items = json.loads(r["items_json"]) if r["items_json"] else [0] * 7

            matches.append(
                {
                    "match_id": r["match_id"],
                    "champion": r["champion_name"] or "Unknown",
                    "role": r["team_position"] or "UNKNOWN",
                    "win": bool(r["win"]),
                    "kills": kills,
                    "deaths": deaths,
                    "assists": assists,
                    "kda": str(kda),
                    "cs": cs,
                    "cs_per_min": f"{(cs / dur_mins):.1f}",
                    "gold": r["gold_earned"] or 0,
                    "duration": f"{dur_mins}:{dur_secs % 60:02d}",
                    "game_mode": r["game_mode"] or "CLASSIC",
                    "game_creation_time": r["game_creation_time"] or 0,
                    "items": items,
                }
            )

    return matches


def backfill_match_participants(limit: int = 500, db_path: str = DB_PATH, order_by_recency: bool = False) -> int:
    """
    Backfill match_participants from compressed raw_json in matches table.
    Uses WHERE NOT EXISTS for optimal query plan.
    Handles corrupt/empty records gracefully to avoid infinite loops.
    """
    order_clause = "ORDER BY m.game_creation_time DESC" if order_by_recency else ""
    query = f"""
        SELECT m.match_id, m.raw_json, m.game_mode, m.game_creation_time, m.game_duration
        FROM matches m
        WHERE NOT EXISTS (
            SELECT 1 FROM match_participants mp WHERE mp.match_id = m.match_id
        )
        {order_clause}
        LIMIT ?
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(query, (limit,)).fetchall()
        if not rows:
            return 0

        p_rows = []
        for r in rows:
            try:
                raw = _decode_compressed_json(r["raw_json"])
            except Exception:
                raw = None
            info = (raw.get("info") or {}) if isinstance(raw, dict) else {}
            participants = (info.get("participants") or []) if isinstance(info, dict) else []
            valid_participants = [p for p in participants if isinstance(p, dict)]

            if not valid_participants:
                # Insert a sentinel tombstone row so this match is marked as processed and never re-queried
                p_rows.append(
                    (
                        r["match_id"],
                        "__TOMBSTONE__",
                        "Unknown",
                        "UNKNOWN",
                        0, 0, 0, 0, 0, 0,
                        r["game_creation_time"] or 0,
                        r["game_duration"] or 0,
                        r["game_mode"] or "UNKNOWN",
                        "[]",
                    )
                )
                continue

            for p in valid_participants:
                if not isinstance(p, dict):
                    continue
                p_puuid = p.get("puuid") or f"__UNKNOWN_{r['match_id']}_{p.get('participantId', 0)}"
                items = [p.get(f"item{i}", 0) for i in range(7)]
                p_rows.append(
                    (
                        r["match_id"],
                        p_puuid,
                        p.get("championName", "Unknown"),
                        p.get("teamPosition") or p.get("individualPosition", "UNKNOWN"),
                        1 if p.get("win") else 0,
                        p.get("kills", 0),
                        p.get("deaths", 0),
                        p.get("assists", 0),
                        p.get("totalMinionsKilled", 0) + p.get("neutralMinionsKilled", 0),
                        p.get("goldEarned", 0),
                        r["game_creation_time"] or info.get("gameCreation"),
                        r["game_duration"] or info.get("gameDuration"),
                        r["game_mode"] or info.get("gameMode", "CLASSIC"),
                        json.dumps(items),
                    )
                )

        if p_rows:
            try:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO match_participants
                        (match_id, puuid, champion_name, team_position, win, kills, deaths, assists,
                         total_cs, gold_earned, game_creation_time, game_duration, game_mode, items_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    p_rows,
                )
            except sqlite3.OperationalError as exc:
                logger.error("Failed to insert match_participants batch: %s", exc)
                raise
        return len(rows)


def get_stale_puuids(puuids: list[str], stale_after_days: int = 7,
                    db_path: str = DB_PATH) -> list[str]:
    """
    Given a list of puuids, return those whose `players.last_updated`
    is older than `stale_after_days` (or NULL / missing from the table).

    Used by the Custom Predictor warm-up: stale players get enqueued
    for background refresh without blocking the prediction.

    Args:
        puuids: List of puuid strings to check.
        stale_after_days: Threshold in days. Default 7.

    Returns:
        Subset of `puuids` considered stale.
    """
    if not puuids:
        return []

    cutoff = _dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None) - _dt.timedelta(days=stale_after_days)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

    # Use SQL placeholders for the IN clause
    placeholders = ",".join(["?"] * len(puuids))
    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT puuid FROM players
            WHERE puuid IN ({placeholders})
              AND (last_updated IS NULL OR last_updated < ?)
            """,
            tuple(puuids) + (cutoff_str,),
        ).fetchall()

    fresh_in_db = {p for p in puuids}
    found_stale = {r["puuid"] for r in rows}

    # Also flag puuids not in players table at all
    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"SELECT puuid FROM players WHERE puuid IN ({placeholders})",
            tuple(puuids),
        ).fetchall()
    in_db = {r["puuid"] for r in rows}
    not_in_db = fresh_in_db - in_db

    return list(found_stale | not_in_db)


def link_accounts(main_puuid: str, smurf_puuid: str, db_path: str = DB_PATH):
    """Link a smurf account to a canonical main account."""
    if not main_puuid or not smurf_puuid or main_puuid == smurf_puuid:
        return
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO player_aliases (alias_puuid, canonical_puuid) VALUES (?, ?)",
            (smurf_puuid, main_puuid),
        )


def unlink_accounts(alias_puuid: str, db_path: str = DB_PATH):
    """Remove a smurf account link from the canonical main account."""
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM player_aliases WHERE alias_puuid = ?", (alias_puuid,))


def get_all_aliases(db_path: str = DB_PATH) -> list[dict]:
    """Retrieve all linked player aliases with their names."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                pa.alias_puuid,
                pa.canonical_puuid,
                p_alias.game_name as alias_name,
                p_alias.tag_line as alias_tag,
                p_canon.game_name as canon_name,
                p_canon.tag_line as canon_tag
            FROM player_aliases pa
            JOIN players p_alias ON pa.alias_puuid = p_alias.puuid
            JOIN players p_canon ON pa.canonical_puuid = p_canon.puuid
        """
        ).fetchall()
    return [dict(r) for r in rows]


def get_top_players(limit: int = 10, db_path: str = DB_PATH) -> list[dict]:
    """
    Get the top players by match count.
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT puuid, game_name, tag_line, total_matches 
            FROM players 
            ORDER BY total_matches DESC 
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_unique_champions(db_path: str = DB_PATH) -> list[str]:
    """
    Retrieve alphabetical list of unique champion names from the registry.
    """
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT name FROM champions ORDER BY name").fetchall()
    return [r["name"] for r in rows]


def _decode_compressed_json(blob: str | None):
    """Decode a possibly-gzip+b64 JSON blob; pass through plain JSON / None."""
    if not blob:
        return None
    try:
        if blob.startswith("{") or blob.startswith("["):
            return json.loads(blob)
        return json.loads(gzip.decompress(base64.b64decode(blob)).decode("utf-8"))
    except Exception:
        return None


def get_untrained_records(limit: int = 500, db_path: str = DB_PATH) -> list[dict]:
    """
    Retrieve training records that have not yet been used for model training.

    Returns dicts with keys: ``id``, ``match_id``, ``feature_json`` (parsed),
    ``winner_label``, ``labels_json`` (parsed dict or None).
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, match_id, feature_json, winner_label, labels_json
            FROM training_dataset
            WHERE is_trained = 0
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    res = []
    for r in rows:
        d = dict(r)
        d["feature_json"] = _decode_compressed_json(d["feature_json"])
        d["labels_json"] = _decode_compressed_json(d["labels_json"])
        res.append(d)
    return res


def get_random_trained_records(limit: int = 500, db_path: str = DB_PATH) -> list[dict]:
    """
    Retrieve a random sample of training records that HAVE been trained on.
    Used for experience replay to prevent catastrophic forgetting.

    Returns dicts with keys: ``id``, ``match_id``, ``feature_json`` (parsed),
    ``winner_label``, ``labels_json`` (parsed dict or None).
    """
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, match_id, feature_json, winner_label, labels_json
            FROM training_dataset
            WHERE is_trained = 1
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    res = []
    for r in rows:
        d = dict(r)
        d["feature_json"] = _decode_compressed_json(d["feature_json"])
        d["labels_json"] = _decode_compressed_json(d["labels_json"])
        res.append(d)
    return res


_MARK_TRAINED_CHUNK = (
    900  # safely below SQLite's default SQLITE_MAX_VARIABLE_NUMBER (999)
)


def mark_as_trained(record_ids: list[int], db_path: str = DB_PATH):
    """
    Mark a list of record IDs as 'trained' to prevent redundant training.

    IDs are processed in chunks of 900 to stay within SQLite's
    SQLITE_MAX_VARIABLE_NUMBER limit (default 999).
    """
    if not record_ids:
        return

    with get_connection(db_path) as conn:
        for i in range(0, len(record_ids), _MARK_TRAINED_CHUNK):
            chunk = record_ids[i : i + _MARK_TRAINED_CHUNK]
            placeholders = ",".join(["?"] * len(chunk))
            conn.execute(
                f"""
                UPDATE training_dataset
                SET is_trained = 1, last_trained_at = CURRENT_TIMESTAMP
                WHERE id IN ({placeholders})
                """,
                tuple(chunk),
            )


def get_raw_match(match_id: str, db_path: str = DB_PATH) -> dict | None:
    """
    Return the fully parsed raw_json for a specific match_id.
    Useful for displaying deep match details.
    """
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT raw_json FROM matches WHERE match_id = ? LIMIT 1", (match_id,)
        ).fetchone()

    if not row:
        return None
    rj = row["raw_json"]
    if rj.startswith("{") or rj.startswith("["):
        return json.loads(rj)
    else:
        return json.loads(gzip.decompress(base64.b64decode(rj)).decode("utf-8"))


def get_match_count(db_path: str = DB_PATH) -> int:
    """Return the total number of stored matches."""
    with get_connection(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]


def get_training_count(db_path: str = DB_PATH) -> int:
    """Return the total number of stored training records."""
    with get_connection(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM training_dataset").fetchone()[0]


def get_db_stats(db_path: str = DB_PATH) -> dict:
    """
    Return a summary dictionary with row counts for all tables and the
    current schema version.

    Example output::

        {
            "schema_version": 2,
            "players": 5,
            "matches": 120,
            "training_records": 118,
        }
    """
    with get_connection(db_path) as conn:
        version = _get_schema_version(conn)
        players = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
        matches = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        training = conn.execute("SELECT COUNT(*) FROM training_dataset").fetchone()[0]
        champs = conn.execute("SELECT COUNT(*) FROM champions").fetchone()[0]

        # New: count trained vs untrained for the dashboard
        trained = conn.execute(
            "SELECT COUNT(*) FROM training_dataset WHERE is_trained = 1"
        ).fetchone()[0]

    untrained = training - trained
    ratio = trained / training if training > 0 else 0

    return {
        "schema_version": version,
        "players": players,
        "matches": matches,
        "training_records": training,
        "champions": champs,
        "trained_records": trained,
        "untrained_records": untrained,
        "trained_ratio": round(ratio, 4),
    }


# ---------------------------------------------------------------------------
# Crawl Queue helpers
# ---------------------------------------------------------------------------


def add_to_crawl_queue(
    game_name: str, tag_line: str, priority: int = 0, region: str = "EUW", db_path: str = DB_PATH
):
    """Add a player to the crawl queue if they don't already exist."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO crawl_queue (game_name, tag_line, priority, region)
            VALUES (?, ?, ?, ?)
            """,
            (game_name, tag_line, priority, region),
        )


def get_next_from_crawl_queue(limit: int = 1, db_path: str = DB_PATH) -> list[dict]:
    """Get the next unprocessed players from the crawl queue."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT game_name, tag_line, region FROM crawl_queue 
            WHERE is_processed = 0 
            ORDER BY priority DESC, id ASC 
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def mark_as_processed(game_name: str, tag_line: str, db_path: str = DB_PATH):
    """Mark a player as processed in the crawl queue."""
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE crawl_queue SET is_processed = 1, processed_at = CURRENT_TIMESTAMP WHERE game_name = ? AND tag_line = ?",
            (game_name, tag_line),
        )


def get_crawl_queue_stats(db_path: str = DB_PATH) -> dict:
    """Get stats about the crawl queue."""
    with get_connection(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM crawl_queue").fetchone()[0]
        processed = conn.execute(
            "SELECT COUNT(*) FROM crawl_queue WHERE is_processed = 1"
        ).fetchone()[0]
    return {"total": total, "processed": processed, "pending": total - processed}


def backfill_seasons(db_path: str = DB_PATH):
    """
    Populate the season_id column for all matches that currently have NULL.
    This reads and parses the raw matches, so it can be slow for large DBs.
    """
    with get_connection(db_path) as conn:
        # Get all matches without a season_id
        rows = conn.execute(
            "SELECT match_id, raw_json FROM matches WHERE season_id IS NULL"
        ).fetchall()

    if not rows:
        logger.info("No matches found requiring season backfill.")
        return

    logger.info("Backfilling season_id for %d matches...", len(rows))
    updates = []
    for r in rows:
        match_id = r["match_id"]
        rj = r["raw_json"]
        try:
            # Parse raw_json (handling potential compression)
            if rj.startswith("{") or rj.startswith("["):
                raw_data = json.loads(rj)
            else:
                raw_data = json.loads(
                    gzip.decompress(base64.b64decode(rj)).decode("utf-8")
                )

            info = raw_data.get("info", {})
            gv = info.get("gameVersion")
            if gv:
                season_id = int(gv.split(".")[0])
                updates.append((season_id, match_id))
        except Exception as e:
            logger.error("Failed to parse match %s during backfill: %s", match_id, e, exc_info=True)

    if updates:
        with get_connection(db_path) as conn:
            conn.executemany(
                "UPDATE matches SET season_id = ? WHERE match_id = ?", updates
            )
        logger.info("Successfully backfilled %d matches.", len(updates))
    else:
        logger.info("No valid season data found in matches.")


def purge_polluted_training_data(db_path: str = DB_PATH) -> dict:
    """
    Remove non-CLASSIC matches and dummy test fixtures from training_dataset
    and delete TEST_MATCH from matches.
    """
    with get_connection(db_path) as conn:
        # Delete training records for TEST_MATCH or non-CLASSIC game modes
        cur = conn.execute(
            """
            DELETE FROM training_dataset
            WHERE match_id = 'TEST_MATCH'
               OR match_id IN (
                   SELECT match_id FROM matches WHERE game_mode != 'CLASSIC' OR game_mode IS NULL OR game_mode = ''
               )
            """
        )
        deleted_training = cur.rowcount

        # Delete TEST_MATCH from matches table
        cur2 = conn.execute("DELETE FROM matches WHERE match_id = 'TEST_MATCH'")
        deleted_test = cur2.rowcount

    logger.info(
        "Purged %d polluted training records and %d test matches.",
        deleted_training,
        deleted_test,
    )
    return {
        "deleted_polluted_training_records": deleted_training,
        "deleted_test_matches": deleted_test,
    }


if __name__ == "__main__":
    init_db()
    backfill_seasons()
    stats = get_db_stats()
    print(f"DB stats: {stats}")
