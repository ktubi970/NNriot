"""
Shared test data for the NNriot project.
Contains common match structures, participant records, and teams.
"""


def get_sample_participant(
    puuid="sample_puuid_1", name="Player1", champion_name="Aatrox", team_id=100
):
    """Returns a sample participant record."""
    return {
        "puuid": puuid,
        "summoner_name": name,
        "champion_id": 266 if champion_name == "Aatrox" else 145,
        "champion_name": champion_name,
        "team_id": team_id,
        "win": team_id == 100,
        "kills": 12 if champion_name == "Aatrox" else 18,
        "deaths": 3 if champion_name == "Aatrox" else 2,
        "assists": 8 if champion_name == "Aatrox" else 12,
        "gold_earned": 15200 if champion_name == "Aatrox" else 18500,
        "total_damage_dealt": 45000 if champion_name == "Aatrox" else 62000,
        "total_damage_taken": 18000 if champion_name == "Aatrox" else 15000,
        "vision_score": 15 if champion_name == "Aatrox" else 22,
        "cs": 180 if champion_name == "Aatrox" else 245,
        "role": "TOP" if champion_name == "Aatrox" else "CARRY",
        "lane": "TOP" if champion_name == "Aatrox" else "BOTTOM",
        "items": (
            [3082, 3071, 3075, 3006, 3053, 1036]
            if champion_name == "Aatrox"
            else [3085, 3005, 3035, 3086, 3006, 3153]
        ),
        "spells": [12, 4] if champion_name == "Aatrox" else [7, 4],
        "level": 18,
    }


def get_sample_match():
    """Returns a complete sample match with 2 participants and 2 teams."""
    return {
        "match_id": "EUW1_1234567890",
        "game_mode": "CLASSIC",
        "game_duration": 1850,
        "game_version": "14.10.400.3000",
        "participants": [
            get_sample_participant("sample_puuid_1", "Player1", "Aatrox", 100),
            get_sample_participant("sample_puuid_2", "Player2", "Kaisa", 100),
        ],
        "teams": [
            {
                "team_id": 100,
                "win": True,
                "dragon_kills": 3,
                "baron_kills": 1,
                "tower_kills": 8,
                "inhibitor_kills": 2,
            },
            {
                "team_id": 200,
                "win": False,
                "dragon_kills": 1,
                "baron_kills": 0,
                "tower_kills": 3,
                "inhibitor_kills": 0,
            },
        ],
        "metadata": {"platform_id": "EUW1", "queue_id": 420, "map_id": 11},
    }


def get_extended_test_data():
    """Returns a list of multiple match structures for testing vectorization."""
    return [
        get_sample_match(),
        {
            "match_id": "KR1_9876543210",
            "game_mode": "CLASSIC",
            "game_duration": 2100,
            "game_version": "14.10.400.3000",
            "participants": [
                {
                    "puuid": "chovy_puuid",
                    "summoner_name": "Chovy",
                    "champion_id": 141,
                    "champion_name": "Kayn",
                    "team_id": 200,
                    "win": False,
                    "kills": 8,
                    "deaths": 6,
                    "assists": 10,
                    "gold_earned": 14800,
                    "cs": 165,
                    "role": "JUNGLE",
                    "lane": "JUNGLE",
                    "items": [3053, 3071, 3075, 3006, 3086, 3111],
                    "spells": [11, 4],
                    "level": 18,
                },
                {
                    "puuid": "faker2_puuid",
                    "summoner_name": "Faker",
                    "champion_id": 34,
                    "champion_name": "Anivia",
                    "team_id": 200,
                    "win": False,
                    "kills": 6,
                    "deaths": 8,
                    "assists": 15,
                    "gold_earned": 12500,
                    "cs": 142,
                    "role": "MID",
                    "lane": "MID",
                    "items": [3089, 3157, 3001, 3020, 3107, 3116],
                    "spells": [14, 4],
                    "level": 18,
                },
            ],
            "teams": [
                {
                    "team_id": 100,
                    "win": True,
                    "dragon_kills": 4,
                    "baron_kills": 2,
                    "tower_kills": 9,
                    "inhibitor_kills": 3,
                },
                {
                    "team_id": 200,
                    "win": False,
                    "dragon_kills": 1,
                    "baron_kills": 0,
                    "tower_kills": 2,
                    "inhibitor_kills": 0,
                },
            ],
            "metadata": {"platform_id": "KR1", "queue_id": 420, "map_id": 11},
        },
        {
            "match_id": "NA1_555666777888",
            "game_mode": "CLASSIC",
            "game_duration": 1620,
            "game_version": "14.10.400.3000",
            "participants": [
                {
                    "puuid": "t1_gumayusi_puuid",
                    "summoner_name": "T1 Gumayusi",
                    "champion_id": 222,
                    "champion_name": "Jinx",
                    "team_id": 100,
                    "win": True,
                    "kills": 22,
                    "deaths": 4,
                    "assists": 6,
                    "gold_earned": 22000,
                    "cs": 320,
                    "role": "CARRY",
                    "lane": "BOTTOM",
                    "items": [3085, 3006, 3035, 3086, 3033, 3008],
                    "spells": [7, 4],
                    "level": 18,
                },
                {
                    "puuid": "t1_keria_puuid",
                    "summoner_name": "T1 Keria",
                    "champion_id": 267,
                    "champion_name": "Nami",
                    "team_id": 100,
                    "win": True,
                    "kills": 2,
                    "deaths": 3,
                    "assists": 18,
                    "gold_earned": 9800,
                    "cs": 45,
                    "role": "SUPPORT",
                    "lane": "BOTTOM",
                    "items": [3012, 3020, 3116, 3009, 3069, 3599],
                    "spells": [3, 4],
                    "level": 18,
                },
            ],
            "teams": [
                {
                    "team_id": 100,
                    "win": True,
                    "dragon_kills": 5,
                    "baron_kills": 2,
                    "tower_kills": 11,
                    "inhibitor_kills": 4,
                },
                {
                    "team_id": 200,
                    "win": False,
                    "dragon_kills": 0,
                    "baron_kills": 0,
                    "tower_kills": 1,
                    "inhibitor_kills": 0,
                },
            ],
            "metadata": {"platform_id": "NA1", "queue_id": 420, "map_id": 11},
        },
    ]
