import os
from typing import List, Dict, Optional

import riot_api

# Helper to resolve region for featured games. Riot API requires a regional endpoint (e.g., 'KR', 'NA', etc.)
# We'll default to KR but allow caller to specify.

def fetch_featured_games(region: str = "KR") -> List[Dict]:
    """Retrieve the list of currently featured games for a given region.

    Returns a list of game dictionaries as provided by the Spectator-V4 endpoint.
    Each game dict contains a ``participants`` list with basic summoner info.
    """
    api = riot_api.RiotAPI(region)
    return api.get_featured_games()


def extract_participants_from_featured(game: Dict) -> List[Dict]:
    """Extract a simplified participant view from a featured game dict.

    The returned list contains dictionaries with:
    - ``summonerName`` (string)
    - ``summonerId`` (string, can be used to resolve PUUID)
    - ``championId`` (int)
    - ``teamId`` (int)
    """
    participants = game.get("participants", [])
    simplified = []
    for p in participants:
        simplified.append({
            "summonerName": p.get("summonerName"),
            "summonerId": p.get("summonerId"),
            "championId": p.get("championId"),
            "teamId": p.get("teamId"),
        })
    return simplified


def resolve_puuid_for_participants(region: str, participants: List[Dict]) -> List[Dict]:
    """Given a list of participant dicts (with ``summonerId``), resolve their PUUIDs.

    Returns the same list with an added ``puuid`` key when resolution succeeds.
    """
    api = riot_api.RiotAPI(region)
    for p in participants:
        if p.get("puuid"):
            continue
        summoner_id = p.get("summonerId")
        if summoner_id:
            puuid = api.get_puuid_by_summoner_id(summoner_id)
            p["puuid"] = puuid
    return participants


def fetch_active_game_by_summoner(name: str, tag: str) -> Optional[Dict]:
    """Fetch the live game data for a specific summoner (by Riot ID).

    Returns the full active-game payload or ``None`` if the summoner is not currently in a game.
    """
    # Resolve region from tag using the same mapping as data_collector
    from data_collector import _resolve_region
    region = _resolve_region(tag)
    api = riot_api.RiotAPI(region)
    puuid = api.get_puuid(name, tag)
    if not puuid:
        return None
    return api.get_active_game_by_summoner(puuid)

if __name__ == "__main__":
    # Simple demo: print featured games and participants for the default region
    games = fetch_featured_games()
    print(f"Found {len(games)} featured games in KR region.")
    for idx, game in enumerate(games, 1):
        participants = extract_participants_from_featured(game)
        participants = resolve_puuid_for_participants("KR", participants)
        print(f"Game {idx}: {len(participants)} participants")
        for p in participants:
            print(f"  {p.get('summonerName')} (PUUID: {p.get('puuid')[:8] if p.get('puuid') else 'N/A'})")
