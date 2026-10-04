"""Extraction of early-game differential metrics from Riot Match-V5 timelines."""
from __future__ import annotations

from typing import Any

FIRST_BLOOD_CUTOFF_MS: int = 600_000  # 10 minutes in milliseconds
TARGET_FRAME_MINUTE: int = 15


def extract_timeline_lane_metrics(
    timeline_data: dict[str, Any] | None,
    participants: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, float]]:
    """Extract GD@15, CSD@15, XPD@15, and FBI@10 for all 10 participants.

    Args:
        timeline_data: Parsed Match-V5 timeline payload containing info.frames and info.participants.
        participants: Optional fallback list of participant metadata dictionaries from Match-V5.

    Returns:
        Mapping of puuid -> {"gd15": float, "csd15": float, "xpd15": float, "fbi10": float}.
        Returns empty dictionary if timeline data is missing, malformed, or incomplete.
    """
    if not isinstance(timeline_data, dict):
        return {}

    info = timeline_data.get("info", timeline_data)
    if not isinstance(info, dict):
        return {}

    frames = info.get("frames") or []
    if not frames:
        return {}

    part_meta = info.get("participants") or []
    pid_to_puuid: dict[int, str] = {}
    if isinstance(part_meta, list):
        for p in part_meta:
            if isinstance(p, dict):
                pid = p.get("participantId")
                puuid = p.get("puuid")
                if pid is not None and puuid:
                    pid_to_puuid[int(pid)] = str(puuid)

    # Fallback to participants list if provided
    if participants and isinstance(participants, list):
        for idx, p in enumerate(participants, start=1):
            if isinstance(p, dict):
                pid = p.get("participantId") or idx
                puuid = p.get("puuid")
                if puuid and int(pid) not in pid_to_puuid:
                    pid_to_puuid[int(pid)] = str(puuid)

    if len(pid_to_puuid) < 10:
        # Require all 10 participants for valid lane differential
        return {}

    # Target frame: minute 15, or last available frame if game ended earlier
    frame_idx = min(TARGET_FRAME_MINUTE, len(frames) - 1)
    target_frame = frames[frame_idx]
    p_frames = target_frame.get("participantFrames") or {}

    # Extract per-participant stats
    stats: dict[int, dict[str, float]] = {}
    for pid in range(1, 11):
        pf = p_frames.get(str(pid)) or {}
        gold = float(pf.get("totalGold") or 0.0)
        minions = float(pf.get("minionsKilled") or 0.0)
        monsters = float(pf.get("jungleMinionsKilled") or 0.0)
        xp = float(pf.get("xp") or 0.0)
        stats[pid] = {
            "gold": gold,
            "cs": minions + monsters,
            "xp": xp,
        }

    # Detect First Blood involvement before 10 min
    fb_involved_pids: set[int] = set()
    found_first_blood = False

    for frame in frames:
        events = frame.get("events") or []
        for ev in events:
            if isinstance(ev, dict) and ev.get("type") == "CHAMPION_KILL":
                ts = ev.get("timestamp") or 0
                if ts <= FIRST_BLOOD_CUTOFF_MS:
                    killer = ev.get("killerId")
                    assists = ev.get("assistingParticipantIds") or []
                    if killer:
                        fb_involved_pids.add(int(killer))
                    for a in assists:
                        fb_involved_pids.add(int(a))
                found_first_blood = True
                break
        if found_first_blood:
            break

    # Calculate differentials against opposite lane counterpart (1 <-> 6, 2 <-> 7, etc.)
    result: dict[str, dict[str, float]] = {}
    for pid in range(1, 11):
        puuid = pid_to_puuid.get(pid)
        if not puuid:
            continue

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
