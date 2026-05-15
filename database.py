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
import logging
from contextlib import contextmanager
import functools

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Use environment variable or fall back to a relative path
DB_PATH = os.environ.get("NNRIOT_DB_PATH", "training_data.db")

# Current schema version – bump this whenever you add a migration below
SCHEMA_VERSION = 10

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


def save_match(match_id: str, raw_data: dict, db_path: str = DB_PATH):
    """Insert a match record (silently ignored if it already exists)."""
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
                (match_id, raw_json, game_mode, game_creation_time, game_duration, season_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                match_id,
                compressed_json,
                info.get("gameMode"),
                info.get("gameCreation"),
                info.get("gameDuration"),
                season_id,
            ),
        )

        # Update champion registry
        participants = info.get("participants", [])
        for p in participants:
            c_name = p.get("championName")
            if c_name:
                conn.execute(
                    "INSERT OR IGNORE INTO champions (name) VALUES (?)", (c_name,)
                )

    logger.debug("Saved match %s.", match_id)


def save_matches_batch(matches: list[tuple], db_path: str = DB_PATH):
    """
    Bulk-insert multiple matches and their champions in a single transaction.
    Each element of *matches* should be a (match_id, raw_data_dict).
    """
    rows = []
    unique_champions = set()
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
            )
        )

        # Collect champions for batch insert
        for p in info.get("participants", []):
            c_name = p.get("championName")
            if c_name:
                unique_champions.add(c_name)

    if not rows:
        return

    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO matches
                (match_id, raw_json, game_mode, game_creation_time, game_duration, season_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        if unique_champions:
            conn.executemany(
                "INSERT OR IGNORE INTO champions (name) VALUES (?)",
                [(c,) for c in unique_champions],
            )

    logger.debug("Batch-saved %d matches and updated champions.", len(rows))


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


def link_accounts(main_puuid: str, smurf_puuid: str, db_path: str = DB_PATH):
    """Link a smurf account to a canonical main account."""
    if not main_puuid or not smurf_puuid or main_puuid == smurf_puuid:
        return
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO player_aliases (alias_puuid, canonical_puuid) VALUES (?, ?)",
            (smurf_puuid, main_puuid),
        )


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


@functools.lru_cache(maxsize=4)
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
    if blob.startswith("{") or blob.startswith("["):
        return json.loads(blob)
    return json.loads(gzip.decompress(base64.b64decode(blob)).decode("utf-8"))


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
    game_name: str, tag_line: str, priority: int = 0, db_path: str = DB_PATH
):
    """Add a player to the crawl queue if they don't already exist."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO crawl_queue (game_name, tag_line, priority)
            VALUES (?, ?, ?)
            """,
            (game_name, tag_line, priority),
        )


def get_next_from_crawl_queue(limit: int = 1, db_path: str = DB_PATH) -> list[dict]:
    """Get the next unprocessed players from the crawl queue."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT game_name, tag_line FROM crawl_queue 
            WHERE is_processed = 0 
            ORDER BY priority DESC, added_at ASC 
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def mark_as_processed(game_name: str, tag_line: str, db_path: str = DB_PATH):
    """Mark a player as processed in the crawl queue."""
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE crawl_queue SET is_processed = 1 WHERE game_name = ? AND tag_line = ?",
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


if __name__ == "__main__":
    init_db()
    backfill_seasons()
    stats = get_db_stats()
    print(f"DB stats: {stats}")


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
