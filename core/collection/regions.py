from typing import Final, Optional

TAG_TO_REGION: Final[dict[str, str]] = {
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


def resolve_region(tag: Optional[str]) -> str:
    """Convert a Riot tag or platform suffix to the canonical region code.

    Falls back to EUW if the tag is unrecognised, empty, or None.
    """
    if not tag:
        return "EUW"
    return TAG_TO_REGION.get(tag.upper(), "EUW")
