"""
feature_labels.py - Multi-output label extraction for NNriot.

Single responsibility: convert a raw Riot Match-V5 match dict into a flat
labels dict suitable for the multi-output Keras model.

Returns ``None`` for malformed matches (missing teams, no clear winner, etc.)
so callers can skip them. See ``MULTI_OUTPUT_MODEL_PLAN.md`` section 2 for the
full design.
"""

from __future__ import annotations

# Ordered list of all label keys produced by ``extract_labels``.
# Used by the trainer (target dict assembly) and by tests.
LABEL_KEYS = [
    "winner",
    "winner_kills",
    "kill_handicap",
    "total_kills",
    "team_a_kills",
    "team_b_kills",
    "kills_odd",
    "first_blood",
    "first_baron",
    "first_inhibitor",
    "first_tower",
    "total_barons",
    "total_dragons",
    "total_towers",
    "both_baron",
    "both_inhibitor",
    "both_dragon",
    "elder_dragon",
]


def _safe_dict(value) -> dict:
    """Return *value* if it is a dict, else an empty dict."""
    return value if isinstance(value, dict) else {}


def _safe_int(value, default: int = 0) -> int:
    """Coerce *value* to int, returning *default* on None / wrong type."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _obj_kills(obj: dict, key: str) -> int:
    """Return ``obj[key]['kills']`` defensively."""
    sub = _safe_dict(obj.get(key))
    return _safe_int(sub.get("kills"), 0)


def _obj_first(obj: dict, key: str) -> bool:
    """Return ``bool(obj[key]['first'])`` defensively."""
    sub = _safe_dict(obj.get(key))
    return bool(sub.get("first"))


def _any_elder(obj: dict) -> bool:
    """
    Detect elder dragon presence for one team's objectives.

    Prefers the newer Riot schema ``objectives.elderDragon.kills >= 1``;
    falls back to the proxy ``objectives.dragon.kills >= 4`` (four elemental
    drakes are required before the elder dragon spawns).
    """
    elder = obj.get("elderDragon")
    if isinstance(elder, dict) and "kills" in elder:
        return _safe_int(elder.get("kills"), 0) >= 1
    # Proxy: any team that has killed 4+ elemental dragons means elder spawned
    # (and was almost certainly contested / taken at some point).
    return _obj_kills(obj, "dragon") >= 4


def extract_labels(match_details: dict) -> dict | None:
    """
    Compute all 18 multi-output labels for a single match.

    Parameters
    ----------
    match_details:
        Riot Match-V5 dict, either the top-level response (with ``info`` key)
        or the inner ``info`` dict.

    Returns
    -------
    dict | None
        Flat dict keyed by :data:`LABEL_KEYS`, or ``None`` for malformed
        matches (missing team, both teams marked win, both marked loss, etc.).
    """
    if not isinstance(match_details, dict):
        return None

    info = match_details.get("info", match_details)
    if not isinstance(info, dict):
        return None

    teams = info.get("teams") or []
    participants = info.get("participants") or []
    if not isinstance(teams, list) or not isinstance(participants, list):
        return None

    team_a = next(
        (t for t in teams if isinstance(t, dict) and t.get("teamId") == 100),
        None,
    )
    team_b = next(
        (t for t in teams if isinstance(t, dict) and t.get("teamId") == 200),
        None,
    )
    if not team_a or not team_b:
        return None

    # Exactly one winner required.  Returns None for draws, both-win, both-loss.
    a_win = bool(team_a.get("win"))
    b_win = bool(team_b.get("win"))
    if a_win == b_win:
        return None

    a_kills = sum(
        _safe_int(p.get("kills"), 0)
        for p in participants
        if isinstance(p, dict) and p.get("teamId") == 100
    )
    b_kills = sum(
        _safe_int(p.get("kills"), 0)
        for p in participants
        if isinstance(p, dict) and p.get("teamId") == 200
    )
    total_kills = a_kills + b_kills

    obj_a = _safe_dict(team_a.get("objectives"))
    obj_b = _safe_dict(team_b.get("objectives"))

    def first_team(key: str) -> int:
        """Return 0 if team A claimed it first, 1 for team B, 2 if neither."""
        if _obj_first(obj_a, key):
            return 0
        if _obj_first(obj_b, key):
            return 1
        return 2

    a_barons = _obj_kills(obj_a, "baron")
    b_barons = _obj_kills(obj_b, "baron")
    a_dragons = _obj_kills(obj_a, "dragon")
    b_dragons = _obj_kills(obj_b, "dragon")
    a_towers = _obj_kills(obj_a, "tower")
    b_towers = _obj_kills(obj_b, "tower")
    a_inhibs = _obj_kills(obj_a, "inhibitor")
    b_inhibs = _obj_kills(obj_b, "inhibitor")

    return {
        "winner": int(b_win),                            # 0 = team A win, 1 = team B win
        "winner_kills": int(b_kills > a_kills),          # 1 iff team B has more kills (ties -> 0)
        "kill_handicap": a_kills - b_kills,              # signed int (team A perspective)
        "total_kills": total_kills,                      # int
        "team_a_kills": a_kills,                         # int
        "team_b_kills": b_kills,                         # int
        "kills_odd": total_kills % 2,                    # 0/1
        "first_blood": first_team("champion"),           # 0/1/2  (first champion kill)
        "first_baron": first_team("baron"),              # 0/1/2
        "first_inhibitor": first_team("inhibitor"),      # 0/1/2
        "first_tower": first_team("tower"),              # 0/1/2
        "total_barons": a_barons + b_barons,             # int
        "total_dragons": a_dragons + b_dragons,          # int
        "total_towers": a_towers + b_towers,             # int
        "both_baron": int(a_barons >= 1 and b_barons >= 1),
        "both_inhibitor": int(a_inhibs >= 1 and b_inhibs >= 1),
        "both_dragon": int(a_dragons >= 1 and b_dragons >= 1),
        "elder_dragon": int(_any_elder(obj_a) or _any_elder(obj_b)),
    }
