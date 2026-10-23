"""
backfill_timelines.py ÔÇö Fetch Riot Match-V5 timelines for matches that
don't have one stored yet, and save them to `match_timelines`.

This script is SLOW (~1-2s per match due to Riot API rate limits,
~17h for 50k matches). Idempotent: re-running only processes matches
without a timeline.

Key design notes:

1. Idempotent via LEFT JOIN: each iteration only fetches matches NOT in
   `match_timelines`. Re-running picks up where it left off.

2. NULL placeholder for permanent failures: if Riot returns no timeline
   (deleted match, region mismatch, etc.), we insert a row with
   `timeline_json = NULL`. This prevents infinite re-fetch loops on the
   next run. The `match_timelines.timeline_json` column is nullable per
   the Sprint 4a schema (v10 -> v11 migration).

3. Region inference from match_id: Match-V5 match IDs are
   "{platformId}_{gameId}". We need this because RiotAPI requires a
   region at construction time. Unknown platform prefix -> skip + NULL.

4. Per-region API instances: cached in a dict to reuse rate-limit
   tracking. A single instance per region is sufficient.

5. Sleep is per-chunk, not per-match: matches in a chunk fire in
   parallel (max 5 by default); we sleep after each chunk to give
   Riot's rate-limit window room to refill.

Usage:
    python backfill_timelines.py            # uses defaults; region detected per match
    python backfill_timelines.py --limit N  # only process N matches (for testing)
    python backfill_timelines.py --chunk 20 # change parallel-fetch batch size (default 5)
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import logging
import time
from collections import defaultdict

from core import config, database
from core import riot_api

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 5  # Riot timeline rate limit is tighter than /matches/{id}
_SLEEP_BETWEEN_CHUNKS = 1.2  # seconds ÔÇö tune based on observed rate-limit headers


def _region_from_match_id(match_id: str) -> str | None:
    """
    Infer the region tag from a Match-V5 match_id.

    Match IDs are formatted "{platformId}_{gameId}", e.g. "EUW1_1234567890".
    Returns the RiotAPI region constructor argument (EUW, NA, KR, etc.)
    or None if the format is unrecognized.
    """
    if not match_id or "_" not in match_id:
        return None
    prefix = match_id.split("_", 1)[0].upper()

    # Map platform IDs to the RiotAPI region constructor using centralized mapping
    from core.collection.regions import TAG_TO_REGION
    return TAG_TO_REGION.get(prefix)


def _wait_for_new_key(
    api: riot_api.RiotAPI,
    api_by_region: dict[str, riot_api.RiotAPI],
    poll_interval: float = 10.0,
) -> None:
    """Pause backfill when RIOT_API_KEY expires and poll .env until a new key is provided."""
    old_key = api.api_key
    banner = (
        "\n"
        + "=" * 72 + "\n"
        + " [PAUSE] La clé Riot API a expiré ou a été rejetée (HTTP 403 / 401).\n"
        + " Le backfill est suspendu en toute sécurité (AUCUN placeholder NULL inséré).\n\n"
        + " PROCÉDURE DE REPRISE EN COURS D'EXÉCUTION :\n"
        + " 1. Ouvrez https://developer.riotgames.com/ et générez une nouvelle clé.\n"
        + " 2. Collez-la dans votre fichier .env (ex: RIOT_API_KEY=RGAPI-...).\n"
        + " 3. Enregistrez le fichier .env.\n\n"
        + " Le script détectera automatiquement la nouvelle clé et reprendra le travail.\n"
        + " (Appuyez sur Ctrl+C pour quitter proprement si vous préférez arrêter).\n"
        + "=" * 72 + "\n"
    )
    print(banner, flush=True)
    logger.warning("Riot API key expired. Waiting for a new key in .env...")

    while True:
        try:
            time.sleep(poll_interval)
        except KeyboardInterrupt:
            logger.info("Backfill interrompu par l'utilisateur.")
            raise

        new_key = api.reload_key()
        if new_key and new_key != old_key:
            logger.info("Nouvelle clé détectée dans .env ! Reprise automatique du backfill...")
            print("\n[OK] Nouvelle clé détectée ! Reprise du backfill...\n", flush=True)
            for r_api in api_by_region.values():
                r_api.api_key = new_key
            break


def backfill_timelines(
    db_path: str = database.DB_PATH,
    chunk_size: int = _CHUNK_SIZE,
    limit: int | None = None,
    duration_hours: float | None = None,
    sleep_between_chunks: float = _SLEEP_BETWEEN_CHUNKS,
) -> dict:
    """
    Stream-fetch timelines for matches without one. Idempotent.

    Args:
        db_path: SQLite DB path.
        chunk_size: How many parallel timeline fetches per Riot API batch.
        limit: Stop after fetching N timelines (None = all missing).
        duration_hours: Stop after running for this many hours (None = no time limit).
        sleep_between_chunks: Seconds to sleep between chunks to respect
            Riot's rate-limit headers.

    Returns:
        Stats dict: {processed, saved, errors, skipped_region_unknown}.
    """
    database.init_db(db_path)

    stats = {"processed": 0, "saved": 0, "errors": 0, "skipped_region_unknown": 0}
    start_time = time.monotonic()
    max_seconds = duration_hours * 3600.0 if duration_hours is not None else None

    # Cache one RiotAPI instance per region (each does its own rate-limit tracking)
    api_by_region: dict[str, riot_api.RiotAPI] = {}

    while True:
        # Check elapsed time limit before querying DB
        if max_seconds is not None and (time.monotonic() - start_time) >= max_seconds:
            logger.info("Reached maximum duration of %.2f hours — stopping.", duration_hours)
            break

        # 1) Find next chunk of matches without a timeline
        with database.get_connection(db_path) as conn:
            rows = conn.execute(
                """
                SELECT m.match_id
                FROM   matches m
                LEFT JOIN match_timelines t ON t.match_id = m.match_id
                WHERE  t.match_id IS NULL
                LIMIT  ?
                """,
                (chunk_size,),
            ).fetchall()

        if not rows:
            logger.info("No more matches without a timeline. Done.")
            break

        # 2) Group by region (so each RiotAPI batch hits the right endpoint)
        match_ids_by_region: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            mid = r["match_id"]
            region = _region_from_match_id(mid)
            if region is None:
                stats["skipped_region_unknown"] += 1
                logger.warning("Cannot infer region for %s ÔÇö skipping.", mid)
                # Mark this row done by inserting a NULL timeline to avoid
                # infinite loop. The LEFT JOIN driver query keys on the row
                # existing, so a NULL timeline_json still removes it from
                # the work-list. Persists progress across restarts.
                with database.get_connection(db_path) as conn:
                    conn.execute(
                        "INSERT OR IGNORE INTO match_timelines (match_id, timeline_json) VALUES (?, NULL)",
                        (mid,),
                    )
                continue
            match_ids_by_region[region].append(mid)

        # 3) Per region, fetch the timelines in parallel
        any_fetched = False
        for region, mids in match_ids_by_region.items():
            if region not in api_by_region:
                api_by_region[region] = riot_api.RiotAPI(region=region)
            api = api_by_region[region]

            results = None
            while True:
                try:
                    results = api.get_match_timelines_batch(
                        mids, max_workers=min(chunk_size, len(mids))
                    )
                    break
                except riot_api.RiotAuthError:
                    pause_t0 = time.monotonic()
                    _wait_for_new_key(api, api_by_region)
                    start_time += (time.monotonic() - pause_t0)
                except Exception:
                    stats["errors"] += len(mids)
                    logger.error(
                        "Batch timeline fetch failed for region=%s mids=%s",
                        region,
                        mids,
                        exc_info=True,
                    )
                    break

            if results is None:
                # Transient failure: do NOT mark matches as NULL; retry on next pass
                continue

            # 4) Save and track stats
            batch = []
            for mid in mids:
                stats["processed"] += 1
                tl = results.get(mid)
                if tl is None:
                    # Riot returned no timeline ÔÇö log and mark as NULL
                    # to avoid infinite re-fetch on the next run.
                    stats["errors"] += 1
                    logger.warning(
                        "Riot returned no timeline for %s ÔÇö storing NULL placeholder.",
                        mid,
                    )
                    batch.append((mid, None))
                else:
                    batch.append((mid, tl))
                    any_fetched = True

            # Save real timelines via the bulk helper, NULL placeholders via raw insert
            real = [(mid, tl) for mid, tl in batch if tl is not None]
            null_marks = [mid for mid, tl in batch if tl is None]

            if real:
                database.save_match_timelines_batch(real, db_path=db_path)
                stats["saved"] += len(real)
            if null_marks:
                with database.get_connection(db_path) as conn:
                    conn.executemany(
                        "INSERT OR IGNORE INTO match_timelines (match_id, timeline_json) VALUES (?, NULL)",
                        [(m,) for m in null_marks],
                    )

        elapsed_sec = time.monotonic() - start_time
        logger.info(
            "Progress: processed=%d saved=%d errors=%d skipped=%d (elapsed: %.1fm / %.2fh)",
            stats["processed"],
            stats["saved"],
            stats["errors"],
            stats["skipped_region_unknown"],
            elapsed_sec / 60.0,
            elapsed_sec / 3600.0,
        )

        # 5) Early termination on --limit or --duration
        if limit is not None and stats["processed"] >= limit:
            logger.info("Reached --limit %d — stopping.", limit)
            break
        if max_seconds is not None and elapsed_sec >= max_seconds:
            logger.info("Reached maximum duration of %.2f hours — stopping.", duration_hours)
            break

        # 6) Sleep between chunks to respect rate limits
        if any_fetched:
            time.sleep(sleep_between_chunks)

    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Backfill Riot timelines for matches without one."
    )
    parser.add_argument("--db", help="Existing source SQLite database")
    parser.add_argument(
        "--chunk", type=int, default=_CHUNK_SIZE,
        help="Parallel fetch batch size (default 5).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Stop after this many matches processed.",
    )
    parser.add_argument(
        "--duration", "--duration-hours",
        dest="duration_hours",
        type=float,
        default=None,
        help="Stop after running for this many hours (e.g. 10 or 0.5).",
    )
    parser.add_argument(
        "--sleep", type=float, default=_SLEEP_BETWEEN_CHUNKS,
        help="Seconds to sleep between chunks (default 1.2).",
    )
    args = parser.parse_args(argv)
    try:
        db_path = config.get_db_path(args.db, require_exists=True)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(f"Source database: {db_path}")
    result = backfill_timelines(
        db_path=db_path,
        chunk_size=args.chunk,
        limit=args.limit,
        duration_hours=args.duration_hours,
        sleep_between_chunks=args.sleep,
    )
    print(f"Result: {result}")
    return result


if __name__ == "__main__":
    main()
