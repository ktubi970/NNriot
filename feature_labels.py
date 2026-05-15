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
    "team_b_kill_lead",
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
    Return True if the team likely killed an Elder Dragon.

    Checks ``objectives.elderDragon.kills >= 1`` when that field is present
    (newer Riot Match-V5 schema). Falls back to a proxy:
    ``objectives.dragon.kills >= 5`` — a team with 5+ elemental drakes
    has almost certainly taken elder, since elder spawns after the 4th drake.

    TODO(P5): verify against real Match-V5 samples during backfill —
    Riot's exact schema for elderDragon may have changed across game versions,
    and the >=5 proxy is still a known over-estimate (could fire when elder
    spawned but wasn't taken).
    """
    elder = obj.get("elderDragon")
    if isinstance(elder, dict) and "kills" in elder:
        return _safe_int(elder.get("kills"), 0) >= 1
    # Proxy: 5+ elemental drakes implies elder spawned (after 4th drake) and
    # was almost certainly taken on the next cycle. Known over-estimate;
    # see TODO(P5) above.
    return _obj_kills(obj, "dragon") >= 5


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
        # team_b_kill_lead: 0 if team A has >= kills (ties favor A), 1 if team B strictly more.
        # Intentional: ties (~1-2% of matches) collapse to team-A side.
        "team_b_kill_lead": int(b_kills > a_kills),          # 1 iff team B has more kills (ties -> 0)
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


# ---------------------------------------------------------------------------
# Timeline-based labels (Sprint 4a)
# ---------------------------------------------------------------------------

# Kill thresholds for first-to-N-kills labels.
TIMELINE_KILL_THRESHOLDS = (5, 10, 15, 20)
TIMELINE_LABEL_KEYS = [f"first_to_{n}_kills" for n in TIMELINE_KILL_THRESHOLDS]


def extract_timeline_labels(timeline_data: dict) -> dict | None:
    """
    Extract first-to-N-kills labels from a Riot Match-V5 timeline.

    For N in (5, 10, 15, 20), returns:
        0 if team 100 (blue) reaches N team-kills first,
        1 if team 200 (red) reaches N first,
        2 if neither team reaches N during the match.

    Riot timelines have a `info.participants` list mapping participantId -> puuid,
    plus `info.frames[].events[]` with `CHAMPION_KILL` events. We use the
    convention: participantId 1-5 = team 100, 6-10 = team 200.

    Returns None if the timeline is malformed.
    """
    try:
        if not isinstance(timeline_data, dict):
            return None
        info = timeline_data.get("info", timeline_data)
        if not isinstance(info, dict):
            return None
        frames = info.get("frames", [])
        if not isinstance(frames, list) or not frames:
            return None

        team_a_kills = 0
        team_b_kills = 0
        result = {key: 2 for key in TIMELINE_LABEL_KEYS}  # default: neither team

        for frame in frames:
            if not isinstance(frame, dict):
                continue
            events = frame.get("events", [])
            if not isinstance(events, list):
                continue
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                if ev.get("type") != "CHAMPION_KILL":
                    continue
                killer_id = ev.get("killerId")
                if killer_id is None:
                    continue
                # Convention: participantId 1-5 = team_a (100), 6-10 = team_b (200)
                if 1 <= killer_id <= 5:
                    team_a_kills += 1
                    team_for_threshold = 0
                    counter = team_a_kills
                elif 6 <= killer_id <= 10:
                    team_b_kills += 1
                    team_for_threshold = 1
                    counter = team_b_kills
                else:
                    continue  # executor / monster / unknown

                # Check each threshold
                for n in TIMELINE_KILL_THRESHOLDS:
                    key = f"first_to_{n}_kills"
                    if result[key] == 2 and counter >= n:
                        result[key] = team_for_threshold

        return result
    except Exception:
        return None
