import riot_api
import database
import time
import json_utils
import threading

# Maps tag suffixes (as returned by the Riot API or used by collect_training_data)
# to the region code accepted by RiotAPI.__init__.
_TAG_TO_REGION = {
    "BR1": "BR",
    "BR": "BR",
    "EUN1": "EUN",
    "EUN": "EUN",
    "EUW1": "EUW",
    "EUW": "EUW",
    "JP1": "JP",
    "JP": "JP",
    "KR1": "KR",
    "KR": "KR",
    "KR2": "KR",
    "LA1": "LA1",
    "LA2": "LA2",
    "NA1": "NA",
    "NA": "NA",
    "OC1": "OCE",
    "OCE": "OCE",
    "RU": "RU",
    "TR1": "TR",
    "TR": "TR",
}

# Game modes with non-standard teams[] structures (Arena 2v2v2v2, Swarm PvE,
# etc.) that don't fit the binary team_a/team_b model. Matches with these
# modes are skipped at training-record build time.
EXCLUDED_GAME_MODES = frozenset({"CHERRY", "STRAWBERRY"})


def resolve_region(tag: str) -> str:
    """
    Convert a Riot tag/platform suffix to the canonical region code
    used by RiotAPI (e.g. 'KR1' -> 'KR').
    If the tag is numeric or unrecognised, defaults to EUW.
    """
    region = _TAG_TO_REGION.get(tag.upper())
    if region is None:
        # Fallback to EUW for numeric/custom tags (like #3327)
        return "EUW"
    return region


def _build_training_record(
    match_id: str, details: dict, region: str = None
) -> tuple | None:
    """
    Parse a raw Riot match-details dict into a (match_id, feature_dict, winner_label)
    training tuple. Returns None if the data is malformed.
    """
    try:
        # Determine winner label (Team 100 win -> 0, Team 200 win -> 1)
        # Note: Riot Match-V5 has 'win' boolean in each participant and also in 'teams'
        info = details.get("info", details)

        game_mode = info.get("gameMode") or ""
        if game_mode in EXCLUDED_GAME_MODES:
            return None

        # Use centralized feature extraction
        feature = json_utils.extract_match_features(details, region=region)

        teams = info.get("teams", [])
        winner = 0
        for t in teams:
            if t["win"] and t["teamId"] == 200:
                winner = 1
                break

        team_a_won = any(t.get("win") and t.get("teamId") == 100 for t in teams)
        team_b_won = any(t.get("win") and t.get("teamId") == 200 for t in teams)
        if not (team_a_won or team_b_won):
            print(f"  Skipping match {match_id}: no team marked as winner")
            return None

        return (match_id, feature, winner)
    except Exception as e:
        print(f"  Error building training record for match {match_id}: {e}")
        return None


def collect_training_data(seed_players, matches_per_player=5, days_back=None):
    """
    Collects match data for training.
    If days_back is provided, it fetches matches from the last X days.
    seed_players should be a list of (name, tag) tuples, where tag includes region (e.g., "KR1", "NA1")
    """
    database.init_db()

    # Calculate start time if days_back is specified
    start_time = None
    if days_back:
        # 86400 seconds in a day
        start_time = int(time.time()) - (days_back * 24 * 60 * 60)
        print(
            f"Filtering matches from the last {days_back} days (since {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))})"
        )

    print(f"Starting collection for {len(seed_players)} seed players...")

    for name, tag in seed_players:
        region = resolve_region(tag)
        print(f"Processing {name}#{tag} (Region: {region})...")

        api = riot_api.RiotAPI(region)
        puuid = api.get_puuid(name, tag)
        if not puuid:
            continue

        database.save_player(puuid, name, tag)
        match_ids = api.get_match_ids(
            puuid, count=matches_per_player, start_time=start_time
        )

        # Filter and fetch details in batch
        new_mids = [mid for mid in match_ids if not database.match_exists(mid)]

        if new_mids:
            print(f"  Fetching {len(new_mids)} new matches in batch...")
            details_map = api.get_match_details_batch(new_mids, max_workers=5)

            matches_batch = []
            training_records_batch = []

            for mid, details in details_map.items():
                matches_batch.append((mid, details))
                record = _build_training_record(mid, details, region=region)
                if record:
                    training_records_batch.append(record)

            if matches_batch:
                database.save_matches_batch(matches_batch)
            if training_records_batch:
                database.save_training_records_batch(training_records_batch)
                database.increment_player_matches(puuid, len(training_records_batch))
                print(
                    f"  Saved {len(training_records_batch)} training records for {name}#{tag}"
                )


def collect_top_leagues_data(
    regions=["KR"], total_players_per_region=100, matches_per_player=10, days_back=30
):
    """
    Fetches official top league players (Challenger/GM/Master) from multiple regions and collects their matches.
    """
    database.init_db()

    # Calculate start time
    start_time = int(time.time()) - (days_back * 24 * 60 * 60)

    total_regions = len(regions)
    total_players = total_players_per_region * total_regions

    print(
        f"Fetching top {total_players_per_region} players from {total_regions} regions ({regions})..."
    )
    print(
        f"Total target: {total_players} players, {total_players * matches_per_player} matches"
    )

    all_entries = []

    for region_idx, region in enumerate(regions):
        print(f"\n🌍 Region {region_idx+1}/{total_regions}: {region}")
        api = riot_api.RiotAPI(region)

        try:
            region_entries = api.get_top_players(total_count=total_players_per_region)
            print(f"  Found {len(region_entries)} top players in {region}")

            # Add region info to each entry
            for entry in region_entries:
                entry["_region"] = region

            all_entries.extend(region_entries)

        except Exception as e:
            print(f"  ❌ Error fetching {region}: {e}")
            continue

    print(f"\n📊 Total players collected across all regions: {len(all_entries)}")

    for i, entry in enumerate(all_entries):
        region = entry.get("_region", "UNKNOWN")
        puuid = entry.get("puuid")

        if not puuid:
            print(f"[{i+1}/{len(all_entries)}] ⚠️ Skipping entry: no PUUID found")
            continue

        # Use region-specific API for this player
        api = riot_api.RiotAPI(region)

        # Fetch actual account name from PUUID
        account_info = api.get_account_by_puuid(puuid)
        if not account_info or not account_info.get("game_name"):
            print(
                f"[{i+1}/{len(all_entries)}] ⚠️ Skipping entry: could not fetch account name for {puuid}"
            )
            continue

        game_name = account_info.get("game_name", "Unknown")
        tag_line = account_info.get("tag_line", region)
        summoner_name = f"{game_name}#{tag_line}"

        print(
            f"[{i+1}/{len(all_entries)}] [{region}] Processing: {summoner_name} (LP: {entry.get('leaguePoints', 0)})..."
        )

        # 1. Save player (use game_name as display name, tag_line as tag, region in tag_line)
        database.save_player(puuid, game_name, tag_line)

        # 2. Get matches
        match_ids = api.get_match_ids(
            puuid, count=matches_per_player, start_time=start_time
        )

        # Filter and fetch details in batch
        new_mids = [mid for mid in match_ids if not database.match_exists(mid)]

        if new_mids:
            print(f"    Fetching {len(new_mids)} new matches in batch...")
            details_map = api.get_match_details_batch(new_mids, max_workers=5)

            matches_batch = []
            training_records_batch = []

            for mid, details in details_map.items():
                matches_batch.append((mid, details))
                record = _build_training_record(mid, details, region=region)
                if record:
                    training_records_batch.append(record)

            # Save batches
            if matches_batch:
                database.save_matches_batch(matches_batch)
            if training_records_batch:
                database.save_training_records_batch(training_records_batch)
                database.increment_player_matches(puuid, len(training_records_batch))
                print(
                    f"    Saved {len(training_records_batch)} records for {summoner_name}"
                )
        else:
            print(f"    No new matches for {summoner_name}")


def collect_by_puuid(accounts: list[dict], matches_per_player: int = 20) -> dict:
    """
    Collect match history for a list of accounts where the PUUID is already known
    (e.g. imported from lolpros.gg). Skips the name→PUUID lookup step.

    Parameters
    ----------
    accounts : list of dicts with keys:
        - puuid   : str   (encrypted_puuid from Riot / lolpros.gg)
        - server  : str   (e.g. "EUW", "NA", "KR") — used as region
        - gamename: str
        - tagline : str
    matches_per_player : int
        How many recent matches to fetch per account.

    Returns
    -------
    dict with keys 'processed', 'saved', 'skipped', 'errors'
    """
    database.init_db()
    stats = {"processed": 0, "saved": 0, "skipped": 0, "errors": 0}

    for acc in accounts:
        puuid = acc.get("puuid", "").strip()
        server = acc.get("server", "").upper()
        name = acc.get("gamename") or acc.get("summoner_name") or "?"
        tag = acc.get("tagline") or server or "?"

        if not puuid:
            print(f"  ⚠ Skipping account with no PUUID: {name}")
            stats["skipped"] += 1
            continue

        # Resolve region — the lolpros.gg 'server' field maps directly
        try:
            region = resolve_region(server) if server else None
        except ValueError:
            # Try a fallback: keep server as-is if it looks like a known region
            region = (
                server
                if server
                in (
                    "KR",
                    "NA",
                    "EUW",
                    "EUN",
                    "BR",
                    "LA1",
                    "LA2",
                    "OCE",
                    "JP",
                    "RU",
                    "TR",
                )
                else None
            )

        if not region:
            print(f"  ⚠ Cannot resolve region '{server}' for {name} — skipping")
            stats["skipped"] += 1
            continue

        print(f"  [{region}] {name}#{tag} ({puuid[:16]}…)")
        try:
            # Ensure player stub exists in DB
            database.save_player(puuid, name, tag)

            api = riot_api.RiotAPI(region)
            match_ids = api.get_match_ids(puuid, count=matches_per_player)
            new_mids = [mid for mid in match_ids if not database.match_exists(mid)]

            if not new_mids:
                print(f"    No new matches.")
                stats["processed"] += 1
                continue

            print(f"    Fetching {len(new_mids)} new matches…")
            details_map = api.get_match_details_batch(new_mids, max_workers=4)

            matches_batch = []
            training_batch = []
            for mid, details in details_map.items():
                matches_batch.append((mid, details))
                record = _build_training_record(mid, details, region=region)
                if record:
                    training_batch.append(record)

            if matches_batch:
                database.save_matches_batch(matches_batch)
            if training_batch:
                database.save_training_records_batch(training_batch)
                database.increment_player_matches(puuid, len(training_batch))
                print(f"    ✓ Saved {len(training_batch)} records.")
                stats["saved"] += len(training_batch)

            stats["processed"] += 1

        except Exception as e:
            print(f"    ✗ Error for {name}#{tag}: {e}")
            stats["errors"] += 1

    return stats


batch_status = {
    "status": "idle",
    "total": 0,
    "current": 0,
    "player": "",
    "logs": []
}
batch_lock = threading.Lock()

def add_batch_log(msg):
    with batch_lock:
        batch_status["logs"].append(msg)
        print(f"[BATCH] {msg}")

def collect_batch_with_smurfs(player_list, sources, count=50):
    with batch_lock:
        batch_status["status"] = "running"
        batch_status["total"] = len(player_list)
        batch_status["current"] = 0
        batch_status["logs"] = []
    
    database.init_db()
    
    for name, tag in player_list:
        with batch_lock:
            batch_status["current"] += 1
            batch_status["player"] = f"{name}#{tag}"
        
        add_batch_log(f"Processing {name}#{tag}...")
        try:
            region = resolve_region(tag)
            api = riot_api.RiotAPI(region)
            main_puuid = api.get_puuid(name, tag)
            
            if not main_puuid:
                add_batch_log(f"  [!] Could not resolve PUUID for {name}#{tag}")
                continue
                
            database.save_player(main_puuid, name, tag)
            accounts_to_collect = [{"puuid": main_puuid, "server": region, "gamename": name, "tagline": tag}]
            
            # Smurf Discovery
            if "liquipedia" in sources:
                add_batch_log("  Searching Liquipedia for smurfs...")
                add_batch_log("  (Liquipedia specific search placeholder)")
                
            if "lolpros" in sources:
                add_batch_log("  Lolpros discovery is currently disabled.")

            # Collection for all accounts
            for acc in accounts_to_collect:
                p_name = f"{acc['gamename']}#{acc['tagline']}"
                add_batch_log(f"  Fetching history for {p_name}...")
                
                p_api = riot_api.RiotAPI(acc["server"])
                matches = p_api.get_match_ids(acc["puuid"], count=count)
                
                if len(matches) <= 5:
                    add_batch_log(f"    Skipping {p_name} (only {len(matches)} matches played)")
                    continue
                
                new_mids = [mid for mid in matches if not database.match_exists(mid)]
                add_batch_log(f"    Found {len(new_mids)} new matches to download")
                
                if new_mids:
                    details_map = p_api.get_match_details_batch(new_mids, max_workers=3, verbose=True)
                    matches_batch = []
                    training_batch = []
                    for mid, details in details_map.items():
                        matches_batch.append((mid, details))
                        record = _build_training_record(mid, details, region=acc["server"])
                        if record:
                            training_batch.append(record)

                    if matches_batch:
                        database.save_matches_batch(matches_batch)
                    if training_batch:
                        database.save_training_records_batch(training_batch)
                        database.increment_player_matches(acc["puuid"], len(training_batch))
                        add_batch_log(f"    ✓ Saved {len(training_batch)} records.")
                        
        except Exception as e:
            add_batch_log(f"  [ERROR] {e}")

    with batch_lock:
        batch_status["status"] = "completed"
        add_batch_log("Batch collection finished.")


if __name__ == "__main__":

    # Option 1: Collect from specific seeds (single region)
    # seeds = [("Hide on bush", "KR1"), ("Chovy", "KR1")]
    # collect_training_data(seeds, matches_per_player=10, days_back=30)

    # Option 2: Collect from top league players across all supported regions
    # Supported entries map (region codes accepted by RiotAPI constructor):
    #  KR, NA, EUW, EUN, BR, LA1, LA2, OCE, JP, RU, TR
    # regions = ["KR", "NA", "EUW", "EUN", "BR", "LA1", "LA2", "OCE", "JP", "RU", "TR"]
    regions = ["NA", "EUW", "EUN"]
    collect_top_leagues_data(
        regions=regions,
        total_players_per_region=500,  # 500 players per region
        matches_per_player=20,  # 20 matches per player
        days_back=60,  # Last 60 days of matches
    )
