"""Backfill match_facts and match_participants.team_id for matches stored before schema v17.

Every legacy match's raw_json is read once: on a large database that is the
whole matches table, so run it with collectors stopped. Each batch commits on
its own and the scan resumes where it stopped, since matches that already have
facts are skipped. Availability is never invented: ``ingested_at`` and
``fetched_by_puuid`` come from the deprecated schema-v16 columns of matches when
they are coherent with the match, and stay NULL otherwise.

    python -m scripts.backfill_match_facts --db path/to/training_data.db
"""
from __future__ import annotations

import argparse
import json
import logging
from typing import Optional

from core import database

logger = logging.getLogger(__name__)


def backfill(db_path: str, *, batch_size: int = 500, max_batches: Optional[int] = None) -> dict:
    database.init_db(db_path)
    report = {"facts_inserted": 0, "participants_updated": 0, "batches": 0}
    last_rowid = 0
    while max_batches is None or report["batches"] < max_batches:
        with database.get_connection(db_path) as conn:
            rows = conn.execute(
                """
                SELECT m.rowid AS rowid, m.match_id, m.raw_json, m.ingested_at, m.fetched_by_puuid
                FROM matches m
                WHERE m.rowid > ?
                  AND NOT EXISTS (SELECT 1 FROM match_facts f WHERE f.match_id = m.match_id)
                ORDER BY m.rowid LIMIT ?
                """,
                (last_rowid, batch_size),
            ).fetchall()
            if not rows:
                break
            facts, teams = [], []
            for row in rows:
                data = database._decode_compressed_json(row["raw_json"])
                info = data.get("info") if isinstance(data, dict) else None
                if not isinstance(info, dict):
                    info = {}  # Still recorded, so an unreadable payload is not scanned again.
                facts.append(database._facts_row(
                    row["match_id"], info,
                    database._coherent_ingested_at(row["ingested_at"], info), row["fetched_by_puuid"],
                ))
                participants = info.get("participants")
                for p in participants if isinstance(participants, list) else []:
                    if isinstance(p, dict) and p.get("puuid") and type(p.get("teamId")) is int:
                        teams.append((p["teamId"], row["match_id"], p["puuid"]))
            conn.executemany(database._INSERT_FACTS, facts)
            cursor = conn.executemany(
                "UPDATE match_participants SET team_id = ? WHERE match_id = ? AND puuid = ? AND team_id IS NULL",
                teams,
            )
            report["facts_inserted"] += len(facts)
            report["participants_updated"] += max(0, cursor.rowcount)
            report["batches"] += 1
            last_rowid = rows[-1]["rowid"]
        logger.info("Backfilled %d matches so far", report["facts_inserted"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", required=True)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(json.dumps(backfill(args.db, batch_size=args.batch_size, max_batches=args.max_batches)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
