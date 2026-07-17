"""Read-only access to the operational match store."""

from __future__ import annotations

import base64
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Iterator


@contextmanager
def connect_source(path: str) -> Iterator[sqlite3.Connection]:
    """Open the operational database in SQLite query-only mode."""

    uri = Path(path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    try:
        yield conn
    finally:
        conn.close()


def decode_raw_json(value: str | bytes | None) -> dict:
    """Decode either legacy plain JSON or the operational gzip/base64 format."""

    try:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not isinstance(value, str):
            raise ValueError("raw match JSON must be text or bytes")
        stripped = value.lstrip()
        if stripped.startswith(("{", "[")):
            decoded = json.loads(stripped)
        else:
            compressed = base64.b64decode(value, validate=True)
            decoded = json.loads(gzip.decompress(compressed).decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("raw match JSON must be an object")
        return decoded
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("invalid raw match JSON") from exc


def iter_source_rows(
    path: str, after_match_id: str | None = None
) -> Iterator[tuple[str, str | bytes | None]]:
    """Yield undecoded CLASSIC source rows in deterministic order."""

    sql = "SELECT match_id,raw_json FROM matches WHERE game_mode='CLASSIC'"
    params: tuple[str, ...] = ()
    if after_match_id is not None:
        sql += " AND match_id > ?"
        params = (after_match_id,)
    sql += " ORDER BY match_id"
    with connect_source(path) as conn:
        for row in conn.execute(sql, params):
            yield str(row["match_id"]), row["raw_json"]


def iter_source_matches(
    path: str, after_match_id: str | None = None
) -> Iterator[tuple[str, dict]]:
    """Yield eligible CLASSIC matches in a deterministic order."""

    for match_id, raw_json in iter_source_rows(path, after_match_id):
        yield match_id, decode_raw_json(raw_json)


def source_fingerprint(path: str, extractor_version: str) -> dict[str, str]:
    """Return a deterministic version for the eligible source dataset."""

    digest = hashlib.sha256(extractor_version.encode("utf-8"))
    with connect_source(path) as conn:
        schema = conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT match_id,game_creation_time FROM matches "
            "WHERE game_mode='CLASSIC' ORDER BY match_id"
        )
        count = 0
        maximum = 0
        for match_id, created in rows:
            created_ms = int(created or 0)
            digest.update(f"{match_id}:{created_ms}\n".encode("utf-8"))
            count += 1
            maximum = max(maximum, created_ms)
    return {
        "dataset_version": digest.hexdigest(),
        "source_schema_version": str(schema),
        "source_match_count": str(count),
        "source_max_creation_ms": str(maximum),
    }


def source_snapshot(path: str) -> dict[str, str]:
    """Return the cheap request-time token for freshness checks."""

    with connect_source(path) as conn:
        schema = conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()[0]
        count, maximum = conn.execute(
            "SELECT COUNT(*),COALESCE(MAX(game_creation_time),0) FROM matches "
            "WHERE game_mode='CLASSIC'"
        ).fetchone()
    return {
        "source_schema_version": str(schema),
        "source_match_count": str(count),
        "source_max_creation_ms": str(maximum),
    }
