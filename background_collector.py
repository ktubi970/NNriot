import logging
import os
import riot_api
import database
import time
from data_collector import (
    resolve_region,
    EXCLUDED_GAME_MODES,
    FETCH_TIMELINES,
    _build_training_record,
)

logger = logging.getLogger(__name__)

# Limit in Bytes — defaults to 50 GB, overridable via NNRIOT_MAX_DB_SIZE_GB.
MAX_DB_SIZE = int(os.environ.get("NNRIOT_MAX_DB_SIZE_GB", "50")) * 1024 * 1024 * 1024


def get_db_size():
    if os.path.exists(database.DB_PATH):
        return os.path.getsize(database.DB_PATH)
    return 0


def run_collector():
    # We will initialize API per region inside the loop
    database.init_db()

    # Seeds to ensure queue is never empty
    seeds = [
        ("Hide on bush", "KR1"),
        ("Chovy", "KR1"),
        ("T1 Gumayusi", "KR1"),
        ("ShowMaker", "KR1"),
        ("Canyon", "KR1"),
        ("T1 Faker", "KR1"),
        ("T1 Keira", "KR1"),
        ("T1 Zeus", "KR1"),
        ("T1 Oner", "KR1"),
    ]

    for name, tag in seeds:
        database.add_to_crawl_queue(name, tag, priority=10)

    print(f"Background collector started. Target DB: {database.DB_PATH}")
    print(f"Size limit: {MAX_DB_SIZE / (1024**3):.1f} GB")

    while True:
        # Check size limit
        current_size = get_db_size()
        if current_size >= MAX_DB_SIZE:
            print(
                f"Database reached size limit ({current_size / (1024**3):.2f} GB). Stopping."
            )
            break

        # Get next player from persistent queue
        players = database.get_next_from_crawl_queue(limit=1)
        if not players:
            print("Crawl queue empty. Re-seeding...")
            for name, tag in seeds:
                database.add_to_crawl_queue(name, tag)
            time.sleep(5)
            continue

        player = players[0]
        name, tag = player["game_name"], player["tag_line"]

        try:
            print(f"Processing {name}#{tag}...")

            # Resolve region from tag
            try:
                region = resolve_region(tag)
            except ValueError:
                print(f"  Unsupported tag {tag}, skipping.")
                database.mark_as_processed(name, tag)
                continue

            api = riot_api.RiotAPI(region)
            puuid = api.get_puuid(name, tag)
            if not puuid:
                database.mark_as_processed(name, tag)
                continue

            database.save_player(puuid, name, tag)

            # Fetch last 20 matches
            match_ids = api.get_match_ids(puuid, count=20)

            # Filter matches we already have
            new_mids = [mid for mid in match_ids if not database.match_exists(mid)]

            if not new_mids:
                print(f"  No new matches for {name}")
                database.mark_as_processed(name, tag)
                continue

            # Fetch details in parallel (Batch size 5)
            print(f"  Fetching {len(new_mids)} new matches in parallel...")
            details_map = api.get_match_details_batch(new_mids, max_workers=5)

            matches_batch = []
            training_records_batch = []

            for mid, details in details_map.items():
                # Pre-filter excluded modes so we don't seed the crawl queue
                # with their participants (no useful training pool growth).
                info = details.get("info", {})
                if (info.get("gameMode") or "") in EXCLUDED_GAME_MODES:
                    continue

                for p in info.get("participants", []):
                    # Grow the crawl pool via DB
                    p_name = p.get("riotIdGameName")
                    p_tag = p.get("riotIdTagline")
                    if p_name and p_tag:
                        database.add_to_crawl_queue(p_name, p_tag, priority=0)

                record = _build_training_record(mid, details, region=region)
                if record is None:
                    continue
                matches_batch.append((mid, details))
                training_records_batch.append(record)

            # Flush batches
            if matches_batch:
                database.save_matches_batch(matches_batch)
            if training_records_batch:
                database.save_training_records_batch(training_records_batch)

            if FETCH_TIMELINES and new_mids:
                timelines = api.get_match_timelines_batch(new_mids, max_workers=5)
                if timelines:
                    database.save_match_timelines_batch(list(timelines.items()))

            database.mark_as_processed(name, tag)
            print(
                f"  Finished {name}#{tag}. Queue stats: {database.get_crawl_queue_stats()}"
            )

            # Rate limit buffer
            time.sleep(2)

        except Exception as e:
            logger.error(f"Error processing {name}: {e}", exc_info=True)
            time.sleep(10)


if __name__ == "__main__":
    run_collector()
