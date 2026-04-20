#!/usr/bin/env python3
"""
calculate_stats.py - Utility to populate player historical performance metrics.
Iterates through matches and updates the 'players' table with averages.
"""

import database
import json
import logging

logger = logging.getLogger(__name__)


def update_all_player_stats():
    """
    Recalculate and update stats for all players in the database.

    Uses two separate connections to minimise write-lock contention:
      1. A streaming read (cursor iteration, not fetchall) to aggregate stats
         without loading the entire raw_json column into memory.
      2. A short write transaction that only holds the lock for the UPDATEs.
    """
    database.init_db()

    player_data: dict = {}  # puuid -> {kills, deaths, assists, gold, matches}

    # ── Phase 1: streaming read — never holds a write lock ──────────────────
    logger.info("Streaming all matches to aggregate player stats...")
    with database.get_connection() as conn:
        cursor = conn.execute("SELECT raw_json FROM matches")
        for row in cursor:
            data = json.loads(row["raw_json"])
            participants = data.get("info", {}).get("participants", [])

            for p in participants:
                puuid = p.get("puuid")
                if not puuid:
                    continue

                if puuid not in player_data:
                    player_data[puuid] = {
                        "kills": 0,
                        "deaths": 0,
                        "assists": 0,
                        "gold": 0,
                        "matches": 0,
                    }

                s = player_data[puuid]
                s["kills"] += p.get("kills", 0)
                s["deaths"] += p.get("deaths", 0)
                s["assists"] += p.get("assists", 0)
                s["gold"] += p.get("goldEarned", 0)
                s["matches"] += 1

    logger.info(
        "Aggregated stats for %d players. Writing to database...", len(player_data)
    )

    # ── Phase 2: short write transaction ────────────────────────────────────
    with database.get_connection() as conn:
        for puuid, s in player_data.items():
            kda = (s["kills"] + s["assists"]) / max(1, s["deaths"])
            avg_gold = s["gold"] // s["matches"]

            conn.execute(
                """
                UPDATE players
                SET avg_kda = ?, avg_gold = ?, total_matches = ?
                WHERE puuid = ?
                """,
                (kda, avg_gold, s["matches"], puuid),
            )

    logger.info("Player stats updated successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    update_all_player_stats()
