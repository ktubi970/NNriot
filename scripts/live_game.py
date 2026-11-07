"""Backward-compatibility shim for live game spectator functions.

Core implementation moved to core.live_game.
"""

from core.live_game import (
    extract_participants_from_featured,
    fetch_active_game_by_summoner,
    fetch_featured_games,
    resolve_puuid_for_participants,
)

__all__ = [
    "extract_participants_from_featured",
    "fetch_active_game_by_summoner",
    "fetch_featured_games",
    "resolve_puuid_for_participants",
]

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
