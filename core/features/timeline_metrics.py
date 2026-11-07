"""Extraction of early-game differential metrics from Riot Match-V5 timelines."""
from __future__ import annotations

from typing import Any
import math

FIRST_BLOOD_CUTOFF_MS: int = 600_000  # 10 minutes in milliseconds
TARGET_FRAME_MINUTE: int = 15


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert value to float, defaulting on None, ValueError, or TypeError."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default



def _normalize_role(role_raw: str) -> str:
    r = role_raw.strip().upper()
    return {
        "MID": "MIDDLE",
        "BOT": "BOTTOM",
        "SUPPORT": "UTILITY",
        "SUP": "UTILITY",
        "ADC": "BOTTOM",
    }.get(r, r)

def extract_timeline_lane_metrics(
    timeline_data: dict[str, Any] | None,
    participants: list[dict[str, Any]] | None = None,
    *,
    require_observations: bool = False,
) -> dict[str, dict[str, float]]:
    """Extract GD@15, CSD@15, XPD@15, and FBI@10 for all 10 participants.

    Args:
        timeline_data: Parsed Match-V5 timeline payload containing info.frames and info.participants.
        participants: Optional fallback list of participant metadata dictionaries from Match-V5.
        require_observations: Reject absent or invalid measurements/events for
            serving-aligned priors. False retains the legacy exploratory fallback.

    Returns:
        Mapping of puuid -> {"gd15": float, "csd15": float, "xpd15": float, "fbi10": float}.
        Returns empty dictionary if timeline data is missing, malformed, or incomplete.
    """
    if not isinstance(timeline_data, dict):
        return {}

    info = timeline_data.get("info", timeline_data)
    if not isinstance(info, dict):
        return {}

    raw_frames = info.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        return {}

    part_meta = info.get("participants")
    pid_to_puuid: dict[int, str] = {}
    pid_to_role: dict[int, str] = {}
    if isinstance(part_meta, list):
        for p in part_meta:
            if isinstance(p, dict):
                pid_raw = p.get("participantId")
                puuid = p.get("puuid")
                role_raw = p.get("teamPosition") or p.get("role") or ""
                if pid_raw is not None and puuid:
                    try:
                        pid = int(pid_raw)
                        pid_to_puuid[pid] = str(puuid)
                        if role_raw:
                            pid_to_role[pid] = _normalize_role(str(role_raw))
                    except (ValueError, TypeError):
                        pass

    # Fallback to participants list if provided
    if isinstance(participants, list):
        for idx, p in enumerate(participants, start=1):
            if isinstance(p, dict):
                pid_raw = p.get("participantId")
                puuid = p.get("puuid")
                role_raw = p.get("teamPosition") or p.get("role") or p.get("individualPosition") or ""
                if puuid:
                    try:
                        pid = int(pid_raw) if pid_raw is not None else idx
                    except (ValueError, TypeError):
                        pid = idx
                    if pid not in pid_to_puuid:
                        pid_to_puuid[pid] = str(puuid)
                    if pid not in pid_to_role and role_raw:
                        pid_to_role[pid] = _normalize_role(str(role_raw))

    # Require all participants 1..10 to be present for valid lane differential
    if not all(pid in pid_to_puuid for pid in range(1, 11)):
        return {}

    # Target frame: minute 15, or last available frame if game ended earlier
    frame_idx = min(TARGET_FRAME_MINUTE, len(raw_frames) - 1)
    target_frame = raw_frames[frame_idx]
    if not isinstance(target_frame, dict):
        return {}

    p_frames = target_frame.get("participantFrames")
    if require_observations:
        # A numeric zero is an observation; a missing field cannot become one.
        # FBI@10 also needs an observed event list, even when it is empty.
        if (not isinstance(p_frames, dict) or any(
                not isinstance(frame, dict) or not isinstance(frame.get('events'), list)
                for frame in raw_frames)):
            return {}
        for pid in range(1, 11):
            observed = p_frames.get(str(pid))
            if not isinstance(observed, dict):
                return {}
            for key in ('totalGold', 'minionsKilled', 'jungleMinionsKilled', 'xp'):
                value = observed.get(key)
                try:
                    valid = type(value) in (int, float) and math.isfinite(value)
                except OverflowError:
                    valid = False
                if not valid:
                    return {}
    if not isinstance(p_frames, dict):
        p_frames = {}

    # Extract per-participant stats safely
    stats: dict[int, dict[str, float]] = {}
    for pid in range(1, 11):
        pf = p_frames.get(str(pid))
        if not isinstance(pf, dict):
            pf = {}
        gold = _safe_float(pf.get("totalGold"))
        minions = _safe_float(pf.get("minionsKilled"))
        monsters = _safe_float(pf.get("jungleMinionsKilled"))
        xp = _safe_float(pf.get("xp"))
        stats[pid] = {
            "gold": gold,
            "cs": minions + monsters,
            "xp": xp,
        }

    # Detect First Blood involvement before 10 min
    # In Riot timelines, neutral executions (minion/tower) have killerId == 0.
    # Only killerId in range(1, 11) counts as genuine First Blood.
    fb_involved_pids: set[int] = set()
    found_first_blood = False

    for frame in raw_frames:
        if not isinstance(frame, dict):
            continue
        events = frame.get("events")
        if not isinstance(events, list):
            continue
        for ev in events:
            if isinstance(ev, dict) and ev.get("type") == "CHAMPION_KILL":
                try:
                    killer_raw = ev.get("killerId")
                    killer = int(killer_raw) if killer_raw is not None else None
                except (ValueError, TypeError):
                    killer = None

                # Neutral execution or invalid killer: do not mark as First Blood
                if killer is None or killer not in range(1, 11):
                    continue

                ts = _safe_float(ev.get("timestamp"), default=0.0)

                assists_raw = ev.get("assistingParticipantIds")
                assists: list[int] = []
                if isinstance(assists_raw, list):
                    for a in assists_raw:
                        try:
                            a_int = int(a)
                            if a_int in range(1, 11):
                                assists.append(a_int)
                        except (ValueError, TypeError):
                            pass

                if ts <= FIRST_BLOOD_CUTOFF_MS:
                    fb_involved_pids.add(killer)
                    for a in assists:
                        fb_involved_pids.add(a)

                found_first_blood = True
                break
        if found_first_blood:
            break

    # Calculate differentials against opposite lane counterpart (1 <-> 6, 2 <-> 7, etc.)
    result: dict[str, dict[str, float]] = {}
    
    blue_roles: dict[str, int] = {}
    red_roles: dict[str, int] = {}
    for pid in range(1, 11):
        role = pid_to_role.get(pid, "")
        if not role:
            continue
        if pid <= 5:
            blue_roles[role] = pid
        else:
            red_roles[role] = pid

    for pid in range(1, 11):
        puuid = pid_to_puuid[pid]
        role = pid_to_role.get(pid, "")
        opp_pid = None
        
        if role:
            if pid <= 5:
                opp_pid = red_roles.get(role)
            else:
                opp_pid = blue_roles.get(role)
        
        if opp_pid is None:
            opp_pid = pid + 5 if pid <= 5 else pid - 5
            
        my_stat = stats.get(pid, {"gold": 0.0, "cs": 0.0, "xp": 0.0})
        opp_stat = stats.get(opp_pid, {"gold": 0.0, "cs": 0.0, "xp": 0.0})

        gd15 = my_stat["gold"] - opp_stat["gold"]
        csd15 = my_stat["cs"] - opp_stat["cs"]
        xpd15 = my_stat["xp"] - opp_stat["xp"]
        fbi10 = 1.0 if pid in fb_involved_pids else 0.0

        result[puuid] = {
            "gd15": float(gd15),
            "csd15": float(csd15),
            "xpd15": float(xpd15),
            "fbi10": float(fbi10),
        }

    return result
