"""Canonical League of Legends champion registry and alias mapping.

Provides deterministic, zero-collision mapping from champion strings to
contiguous integer IDs [0, NUM_CHAMPIONS - 1].
"""
from __future__ import annotations

UNK_CHAMPION_ID = 0


def _clean_key(name: str) -> str:
    """Strip all non-alphanumeric characters and lowercase."""
    return "".join(c for c in name if c.isalnum()).lower()


# Common Riot API aliases and formatting variations
_RAW_ALIASES: dict[str, str] = {
    "wukong": "MonkeyKing",
    "monkeyking": "MonkeyKing",
    "nunu & willump": "Nunu",
    "nunu": "Nunu",
    "renata glasc": "Renata",
    "renata": "Renata",
    "drmundo": "DrMundo",
    "dr. mundo": "DrMundo",
    "jarvan iv": "JarvanIV",
    "kog'maw": "KogMaw",
    "reksai": "RekSai",
    "rek'sai": "RekSai",
    "vel'koz": "Velkoz",
    "velkoz": "Velkoz",
    "k'sante": "KSante",
    "ksante": "KSante",
    "bel'veth": "Belveth",
    "belveth": "Belveth",
    "kai'sa": "Kaisa",
    "kaisa": "Kaisa",
    "cho'gath": "Chogath",
    "chogath": "Chogath",
    "kha'zix": "Khazix",
    "khazix": "Khazix",
    "leblanc": "Leblanc",
}

# 168+ canonical champions as of current League of Legends
_RAW_CHAMPIONS: list[str] = [
    "UNKNOWN",
    "Aatrox", "Ahri", "Akali", "Akshan", "Alistar", "Ambessa", "Amumu", "Anivia",
    "Annie", "Aphelios", "Ashe", "AurelionSol", "Aurora", "Azir", "Bard",
    "Belveth", "Blitzcrank", "Brand", "Braum", "Briar", "Caitlyn", "Camille",
    "Cassiopeia", "Chogath", "Corki", "Darius", "Diana", "Draven", "DrMundo",
    "Ekko", "Elise", "Evelynn", "Ezreal", "Fiddlesticks", "Fiora", "Fizz",
    "Galio", "Gangplank", "Garen", "Gnar", "Gragas", "Graves", "Gwen",
    "Hecarim", "Heimerdinger", "Hwei", "Illaoi", "Irelia", "Ivern", "Janna",
    "JarvanIV", "Jax", "Jayce", "Jhin", "Jinx", "Kaisa", "Kalista",
    "Karma", "Karthus", "Kassadin", "Katarina", "Kayle", "Kayn", "Kennen",
    "Khazix", "Kindred", "Kled", "KogMaw", "KSante", "Leblanc", "LeeSin",
    "Leona", "Lillia", "Lissandra", "Lucian", "Lulu", "Lux", "Malphite",
    "Malzahar", "Maokai", "MasterYi", "Milio", "MissFortune", "MonkeyKing",
    "Mordekaiser", "Morgana", "Naafiri", "Nami", "Nasus", "Nautilus",
    "Neeko", "Nidalee", "Nilah", "Nocturne", "Nunu", "Olaf", "Orianna",
    "Ornn", "Pantheon", "Poppy", "Pyke", "Qiyana", "Quinn", "Rakan",
    "Rammus", "RekSai", "Rell", "Renata", "Renekton", "Rengar", "Riven",
    "Rumble", "Ryze", "Samira", "Sejuani", "Senna", "Seraphine", "Sett",
    "Shaco", "Shen", "Shyvana", "Singed", "Sion", "Sivir", "Skarner",
    "Smolder", "Sona", "Soraka", "Swain", "Sylas", "Syndra", "TahmKench",
    "Taliyah", "Talon", "Taric", "Teemo", "Thresh", "Tristana", "Trundle",
    "Tryndamere", "TwistedFate", "Twitch", "Udyr", "Urgot", "Varus",
    "Vayne", "Veigar", "Velkoz", "Vex", "Vi", "Viego", "Viktor",
    "Vladimir", "Volibear", "Warwick", "Xayah", "Xerath", "XinZhao",
    "Yasuo", "Yone", "Yorick", "Yuumi", "Zac", "Zed", "Zeri",
    "Ziggs", "Zilean", "Zoe", "Zyra",
]

CANONICAL_CHAMPIONS: tuple[str, ...] = tuple(_RAW_CHAMPIONS)
NUM_CHAMPIONS: int = len(CANONICAL_CHAMPIONS)

_CHAMP_TO_ID: dict[str, int] = {_clean_key(name): idx for idx, name in enumerate(CANONICAL_CHAMPIONS)}
_ID_TO_CHAMP: dict[int, str] = {idx: name for idx, name in enumerate(CANONICAL_CHAMPIONS)}
_ALIASES: dict[str, str] = {_clean_key(k): v for k, v in _RAW_ALIASES.items()}


def normalize_champion_name(name: str | None) -> str:
    """Normalize champion name, resolving known aliases, casing, spaces, and punctuation."""
    if not isinstance(name, str):
        return "UNKNOWN"
    clean = name.strip()
    if not clean:
        return "UNKNOWN"
    key = _clean_key(clean)
    if key in _ALIASES:
        return _ALIASES[key]
    # Check if key matches a known canonical champion
    if key in _CHAMP_TO_ID:
        return _ID_TO_CHAMP[_CHAMP_TO_ID[key]]
    return clean


def champion_to_id(name: str | None) -> int:
    """Return the integer ID for a champion, defaulting to UNK_CHAMPION_ID (0)."""
    norm = normalize_champion_name(name)
    return _CHAMP_TO_ID.get(_clean_key(norm), UNK_CHAMPION_ID)


def id_to_champion(champ_id: int) -> str:
    """Return the canonical champion name for an ID."""
    if not isinstance(champ_id, int):
        return "UNKNOWN"
    return _ID_TO_CHAMP.get(champ_id, "UNKNOWN")
