"""Static meta characteristics for League of Legends champions.

Provides domain properties per champion:
- Damage distribution: physical (AD), magic (AP), true damage ratios (sum to 1.0)
- Crowd control (CC) / Hard engage intensity [0.0, 1.0]
- Scaling curve: early game bully (-1.0) to late game hypercarry (+1.0)
- Range: melee (0.0) vs ranged (1.0)
- Base champion baseline winrate prior (default 0.50)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from core.champion_registry import (
    NUM_CHAMPIONS,
    UNK_CHAMPION_ID,
    CANONICAL_CHAMPIONS,
    champion_to_id,
    normalize_champion_name,
)


class ChampionStaticProfile(NamedTuple):
    """Immutable domain attributes for a champion."""
    ad_ratio: float       # Ratio of physical damage [0.0, 1.0]
    ap_ratio: float       # Ratio of magic damage [0.0, 1.0]
    true_ratio: float     # Ratio of true damage [0.0, 1.0]
    cc_score: float       # Hard CC / engage potency [0.0, 1.0]
    scaling_score: float  # Power curve: early (-1.0) to late (+1.0)
    is_ranged: float      # Melee (0.0) or Ranged (1.0)
    base_winrate: float   # Long-term baseline winrate prior [0.45, 0.55]


# Neutral default profile for UNKNOWN or unrecognized champions
DEFAULT_PROFILE = ChampionStaticProfile(
    ad_ratio=0.50,
    ap_ratio=0.45,
    true_ratio=0.05,
    cc_score=0.40,
    scaling_score=0.00,
    is_ranged=0.50,
    base_winrate=0.50,
)

# Explicit domain profiles for major League of Legends archetypes
_CHAMPION_PROFILES: dict[str, ChampionStaticProfile] = {
    # Tanks / Hard Engage
    "Alistar": ChampionStaticProfile(0.15, 0.80, 0.05, 0.95, -0.10, 0.0, 0.50),
    "Amumu": ChampionStaticProfile(0.10, 0.85, 0.05, 0.95, 0.20, 0.0, 0.51),
    "Blitzcrank": ChampionStaticProfile(0.20, 0.75, 0.05, 0.90, -0.20, 0.0, 0.50),
    "Braum": ChampionStaticProfile(0.25, 0.70, 0.05, 0.80, 0.10, 0.0, 0.50),
    "Leona": ChampionStaticProfile(0.20, 0.75, 0.05, 0.95, -0.15, 0.0, 0.50),
    "Malphite": ChampionStaticProfile(0.15, 0.80, 0.05, 0.90, 0.20, 0.0, 0.51),
    "Maokai": ChampionStaticProfile(0.15, 0.80, 0.05, 0.85, 0.15, 0.0, 0.51),
    "Nautilus": ChampionStaticProfile(0.15, 0.80, 0.05, 0.95, -0.15, 0.0, 0.49),
    "Ornn": ChampionStaticProfile(0.20, 0.75, 0.05, 0.85, 0.50, 0.0, 0.50),
    "Rammus": ChampionStaticProfile(0.15, 0.80, 0.05, 0.80, 0.10, 0.0, 0.51),
    "Rell": ChampionStaticProfile(0.15, 0.80, 0.05, 0.95, 0.10, 0.0, 0.50),
    "Sejuani": ChampionStaticProfile(0.20, 0.75, 0.05, 0.90, 0.15, 0.0, 0.49),
    "Sion": ChampionStaticProfile(0.70, 0.25, 0.05, 0.75, 0.40, 0.0, 0.49),
    "Zac": ChampionStaticProfile(0.10, 0.85, 0.05, 0.90, 0.30, 0.0, 0.51),

    # Late-game Hypercarries / Scaling
    "Kayle": ChampionStaticProfile(0.25, 0.65, 0.10, 0.20, 0.95, 1.0, 0.52),
    "Kassadin": ChampionStaticProfile(0.05, 0.90, 0.05, 0.30, 0.90, 0.0, 0.51),
    "Smolder": ChampionStaticProfile(0.60, 0.25, 0.15, 0.15, 0.90, 1.0, 0.49),
    "AurelionSol": ChampionStaticProfile(0.05, 0.90, 0.05, 0.50, 0.85, 1.0, 0.51),
    "Vayne": ChampionStaticProfile(0.60, 0.05, 0.35, 0.30, 0.85, 1.0, 0.51),
    "Jinx": ChampionStaticProfile(0.90, 0.05, 0.05, 0.30, 0.75, 1.0, 0.51),
    "KogMaw": ChampionStaticProfile(0.40, 0.55, 0.05, 0.20, 0.80, 1.0, 0.51),
    "Vladimir": ChampionStaticProfile(0.05, 0.95, 0.00, 0.10, 0.80, 1.0, 0.50),
    "Veigar": ChampionStaticProfile(0.05, 0.95, 0.00, 0.65, 0.75, 1.0, 0.50),
    "Senna": ChampionStaticProfile(0.85, 0.10, 0.05, 0.40, 0.75, 1.0, 0.50),
    "Twitch": ChampionStaticProfile(0.70, 0.20, 0.10, 0.20, 0.75, 1.0, 0.51),

    # Early Game Bullies / Snowballers
    "Renekton": ChampionStaticProfile(0.90, 0.05, 0.05, 0.60, -0.65, 0.0, 0.49),
    "Pantheon": ChampionStaticProfile(0.85, 0.10, 0.05, 0.65, -0.55, 0.0, 0.50),
    "Draven": ChampionStaticProfile(0.95, 0.00, 0.05, 0.30, -0.50, 1.0, 0.50),
    "Elise": ChampionStaticProfile(0.10, 0.85, 0.05, 0.65, -0.60, 1.0, 0.50),
    "LeeSin": ChampionStaticProfile(0.85, 0.10, 0.05, 0.55, -0.45, 0.0, 0.49),
    "Nidalee": ChampionStaticProfile(0.10, 0.90, 0.00, 0.05, -0.55, 1.0, 0.49),
    "Jayce": ChampionStaticProfile(0.85, 0.10, 0.05, 0.35, -0.35, 1.0, 0.48),
    "Lucian": ChampionStaticProfile(0.85, 0.15, 0.00, 0.10, -0.30, 1.0, 0.49),
    "Kalista": ChampionStaticProfile(0.90, 0.05, 0.05, 0.40, -0.45, 1.0, 0.48),
    "RekSai": ChampionStaticProfile(0.90, 0.00, 0.10, 0.60, -0.50, 0.0, 0.50),
    "Olaf": ChampionStaticProfile(0.75, 0.00, 0.25, 0.30, -0.35, 0.0, 0.50),

    # AD Assassins / Skirmishers
    "Zed": ChampionStaticProfile(0.95, 0.05, 0.00, 0.15, 0.05, 0.0, 0.50),
    "Talon": ChampionStaticProfile(0.95, 0.05, 0.00, 0.20, -0.20, 0.0, 0.50),
    "Qiyana": ChampionStaticProfile(0.90, 0.10, 0.00, 0.65, 0.10, 0.0, 0.49),
    "Khazix": ChampionStaticProfile(0.95, 0.00, 0.05, 0.20, 0.15, 0.0, 0.50),
    "Rengar": ChampionStaticProfile(0.90, 0.05, 0.05, 0.30, 0.00, 0.0, 0.50),
    "Shaco": ChampionStaticProfile(0.60, 0.35, 0.05, 0.45, -0.10, 0.0, 0.50),
    "Naafiri": ChampionStaticProfile(0.95, 0.00, 0.05, 0.20, 0.10, 0.0, 0.51),
    "Yasuo": ChampionStaticProfile(0.90, 0.05, 0.05, 0.60, 0.30, 0.0, 0.49),
    "Yone": ChampionStaticProfile(0.70, 0.20, 0.10, 0.65, 0.35, 0.0, 0.49),
    "Fiora": ChampionStaticProfile(0.65, 0.00, 0.35, 0.25, 0.60, 0.0, 0.50),
    "Camille": ChampionStaticProfile(0.55, 0.05, 0.40, 0.65, 0.55, 0.0, 0.51),
    "Irelia": ChampionStaticProfile(0.75, 0.20, 0.05, 0.50, 0.10, 0.0, 0.49),
    "Riven": ChampionStaticProfile(0.95, 0.00, 0.05, 0.55, 0.20, 0.0, 0.51),

    # AP Mages / AP Assassins
    "Ahri": ChampionStaticProfile(0.05, 0.85, 0.10, 0.60, 0.10, 1.0, 0.50),
    "Syndra": ChampionStaticProfile(0.05, 0.90, 0.05, 0.60, 0.40, 1.0, 0.50),
    "Orianna": ChampionStaticProfile(0.10, 0.90, 0.00, 0.70, 0.40, 1.0, 0.49),
    "Viktor": ChampionStaticProfile(0.05, 0.95, 0.00, 0.45, 0.55, 1.0, 0.50),
    "Hwei": ChampionStaticProfile(0.05, 0.95, 0.00, 0.65, 0.45, 1.0, 0.50),
    "Azir": ChampionStaticProfile(0.05, 0.95, 0.00, 0.65, 0.70, 1.0, 0.47),
    "Anivia": ChampionStaticProfile(0.05, 0.95, 0.00, 0.75, 0.50, 1.0, 0.51),
    "Cassiopeia": ChampionStaticProfile(0.05, 0.95, 0.00, 0.55, 0.60, 1.0, 0.51),
    "Lux": ChampionStaticProfile(0.05, 0.95, 0.00, 0.65, 0.15, 1.0, 0.50),
    "Xerath": ChampionStaticProfile(0.05, 0.95, 0.00, 0.55, 0.25, 1.0, 0.51),
    "Ziggs": ChampionStaticProfile(0.05, 0.95, 0.00, 0.45, 0.35, 1.0, 0.51),
    "Leblanc": ChampionStaticProfile(0.10, 0.90, 0.00, 0.45, -0.20, 1.0, 0.48),
    "Fizz": ChampionStaticProfile(0.10, 0.90, 0.00, 0.50, 0.10, 0.0, 0.51),
    "Ekko": ChampionStaticProfile(0.15, 0.85, 0.00, 0.55, 0.30, 0.0, 0.50),
    "Katarina": ChampionStaticProfile(0.30, 0.70, 0.00, 0.05, 0.20, 0.0, 0.50),
    "Akali": ChampionStaticProfile(0.20, 0.80, 0.00, 0.20, 0.15, 0.0, 0.49),
    "Sylas": ChampionStaticProfile(0.10, 0.90, 0.00, 0.55, 0.30, 0.0, 0.50),

    # Enchanters / Utility Supports
    "Janna": ChampionStaticProfile(0.05, 0.90, 0.05, 0.70, 0.30, 1.0, 0.51),
    "Lulu": ChampionStaticProfile(0.10, 0.85, 0.05, 0.65, 0.30, 1.0, 0.50),
    "Nami": ChampionStaticProfile(0.10, 0.90, 0.00, 0.75, 0.10, 1.0, 0.50),
    "Sona": ChampionStaticProfile(0.05, 0.95, 0.00, 0.55, 0.60, 1.0, 0.52),
    "Soraka": ChampionStaticProfile(0.05, 0.95, 0.00, 0.45, 0.25, 1.0, 0.51),
    "Karma": ChampionStaticProfile(0.10, 0.90, 0.00, 0.45, -0.25, 1.0, 0.49),
    "Milio": ChampionStaticProfile(0.05, 0.95, 0.00, 0.55, 0.40, 1.0, 0.50),
    "Yuumi": ChampionStaticProfile(0.05, 0.95, 0.00, 0.35, 0.45, 1.0, 0.48),
    "Bard": ChampionStaticProfile(0.20, 0.80, 0.00, 0.75, 0.35, 1.0, 0.51),
    "Thresh": ChampionStaticProfile(0.25, 0.70, 0.05, 0.85, 0.10, 1.0, 0.50),
    "Pyke": ChampionStaticProfile(0.85, 0.00, 0.15, 0.75, -0.40, 0.0, 0.50),
}


# Pre-compute static array of profiles indexed by champion integer ID
STATIC_PROFILES_BY_ID: list[ChampionStaticProfile] = []
for idx, name in enumerate(CANONICAL_CHAMPIONS):
    norm = normalize_champion_name(name)
    prof = _CHAMPION_PROFILES.get(norm, DEFAULT_PROFILE)
    STATIC_PROFILES_BY_ID.append(prof)


def get_champion_profile(name: str | None) -> ChampionStaticProfile:
    """Return static domain profile for a champion name string.

    Resolution order:
    1. champion_to_id(name) → STATIC_PROFILES_BY_ID[cid]  (canonical ID-based, alias-safe)
    2. _CHAMPION_PROFILES.get(normalize_champion_name(name))  (dict fallback)
    3. DEFAULT_PROFILE
    """
    cid = champion_to_id(name)
    if cid != UNK_CHAMPION_ID and 0 <= cid < len(STATIC_PROFILES_BY_ID):
        return STATIC_PROFILES_BY_ID[cid]
    # Fallback: direct dict lookup (handles edge cases where registry diverges)
    norm = normalize_champion_name(name)
    return _CHAMPION_PROFILES.get(norm, DEFAULT_PROFILE)


def get_champion_profile_by_id(cid: int) -> ChampionStaticProfile:
    """Fast integer index lookup for champion profile."""
    if 0 <= cid < len(STATIC_PROFILES_BY_ID):
        return STATIC_PROFILES_BY_ID[cid]
    return DEFAULT_PROFILE
