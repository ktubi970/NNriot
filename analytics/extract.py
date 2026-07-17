"""Normalize Riot Match-V5 payloads into validated analytical facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REGIONS = {"NA1": "NA", "EUW1": "EUW", "EUN1": "EUN", "KR": "KR"}
ROLES = {"TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"}


class ExtractionError(ValueError):
    """Expected source-data rejection with a machine-stable reason."""

    def __init__(self, code: str, field: str):
        super().__init__(f"{code}: {field}")
        self.code = code
        self.field = field


@dataclass(frozen=True)
class ExtractionResult:
    match: dict[str, Any]
    teams: tuple[dict[str, Any], dict[str, Any]]
    players: tuple[dict[str, Any], ...]


def _objective(team: dict, key: str) -> tuple[int, int]:
    value = team.get("objectives", {}).get(key, {})
    return int(value.get("kills", 0) or 0), int(bool(value.get("first", False)))


def _region(match_id: str) -> tuple[str, str]:
    prefix = match_id.split("_", 1)[0].upper()
    return prefix, REGIONS.get(prefix, "OTHER")


def _patch(version: str) -> str:
    parts = version.split(".")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise ExtractionError("INVALID_GAME_VERSION", "info.gameVersion")
    return f"{int(parts[0])}.{int(parts[1])}"


def _integer(container: dict, key: str, field: str) -> int:
    value = container.get(key)
    if value is None:
        raise ExtractionError("MISSING_FIELD", field)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ExtractionError("INVALID_FIELD", field) from exc


def _number(container: dict, key: str, field: str) -> float:
    value = container.get(key)
    if value is None:
        raise ExtractionError("MISSING_FIELD", field)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ExtractionError("INVALID_FIELD", field) from exc


def _validate_teams(raw_teams: Any) -> tuple[dict, dict]:
    if not isinstance(raw_teams, list) or len(raw_teams) != 2:
        raise ExtractionError("INVALID_TEAM_COUNT", "info.teams")
    by_id: dict[int, dict] = {}
    for team in raw_teams:
        if not isinstance(team, dict):
            raise ExtractionError("INVALID_TEAM", "info.teams")
        team_id = _integer(team, "teamId", "info.teams.teamId")
        if team_id in by_id:
            raise ExtractionError("DUPLICATE_TEAM", "info.teams.teamId")
        by_id[team_id] = team
    if set(by_id) != {100, 200}:
        raise ExtractionError("INVALID_TEAM_IDS", "info.teams.teamId")
    if sum(bool(team.get("win")) for team in by_id.values()) != 1:
        raise ExtractionError("INVALID_WINNER_COUNT", "info.teams.win")
    return by_id[100], by_id[200]


def _validate_participants(raw_participants: Any) -> tuple[dict, ...]:
    if not isinstance(raw_participants, list) or len(raw_participants) != 10:
        raise ExtractionError("INVALID_PARTICIPANT_COUNT", "info.participants")
    seen: set[str] = set()
    counts = {100: 0, 200: 0}
    participants: list[dict] = []
    for index, participant in enumerate(raw_participants):
        field = f"info.participants[{index}]"
        if not isinstance(participant, dict):
            raise ExtractionError("INVALID_PARTICIPANT", field)
        puuid = participant.get("puuid")
        if not isinstance(puuid, str) or not puuid.strip():
            raise ExtractionError("MISSING_PARTICIPANT_ID", f"{field}.puuid")
        if puuid in seen:
            raise ExtractionError("DUPLICATE_PARTICIPANT", f"{field}.puuid")
        team_id = _integer(participant, "teamId", f"{field}.teamId")
        if team_id not in counts:
            raise ExtractionError("INVALID_PARTICIPANT_TEAM", f"{field}.teamId")
        seen.add(puuid)
        counts[team_id] += 1
        participants.append(participant)
    if counts != {100: 5, 200: 5}:
        raise ExtractionError("INVALID_TEAM_COMPOSITION", "info.participants.teamId")
    return tuple(participants)


def _player_row(
    match_id: str,
    participant: dict,
    winner_team_id: int,
    index: int,
) -> dict[str, Any]:
    team_id = int(participant["teamId"])
    prefix = f"info.participants[{index}]"
    champion = participant.get("championName")
    if not isinstance(champion, str) or not champion.strip():
        raise ExtractionError("MISSING_FIELD", f"{prefix}.championName")
    role = str(participant.get("teamPosition") or "UNKNOWN").upper()
    if role not in ROLES:
        role = "UNKNOWN"
    return {
        "match_id": match_id,
        "puuid": participant["puuid"],
        "team_id": team_id,
        "side": "BLUE" if team_id == 100 else "RED",
        "win": int(team_id == winner_team_id),
        "champion": champion,
        "role": role,
        "kills": _integer(participant, "kills", f"{prefix}.kills"),
        "deaths": _integer(participant, "deaths", f"{prefix}.deaths"),
        "assists": _integer(participant, "assists", f"{prefix}.assists"),
        "gold": _integer(participant, "goldEarned", f"{prefix}.goldEarned"),
        "champion_damage": _integer(
            participant,
            "totalDamageDealtToChampions",
            f"{prefix}.totalDamageDealtToChampions",
        ),
        "vision_score": _number(
            participant, "visionScore", f"{prefix}.visionScore"
        ),
        "lane_cs": _integer(
            participant, "totalMinionsKilled", f"{prefix}.totalMinionsKilled"
        ),
        "jungle_cs": _integer(
            participant, "neutralMinionsKilled", f"{prefix}.neutralMinionsKilled"
        ),
    }


def _team_row(
    match_id: str,
    team: dict,
    players: tuple[dict[str, Any], ...],
    winner_team_id: int,
) -> dict[str, Any]:
    team_id = int(team["teamId"])
    members = [player for player in players if player["team_id"] == team_id]
    barons, first_baron = _objective(team, "baron")
    dragons, first_dragon = _objective(team, "dragon")
    towers, first_tower = _objective(team, "tower")
    inhibitors, first_inhibitor = _objective(team, "inhibitor")
    rift_heralds, _ = _objective(team, "riftHerald")
    atakhan, _ = _objective(team, "atakhan")
    _, first_blood = _objective(team, "champion")
    return {
        "match_id": match_id,
        "team_id": team_id,
        "side": "BLUE" if team_id == 100 else "RED",
        "win": int(team_id == winner_team_id),
        "kills": sum(player["kills"] for player in members),
        "deaths": sum(player["deaths"] for player in members),
        "assists": sum(player["assists"] for player in members),
        "gold": sum(player["gold"] for player in members),
        "champion_damage": sum(player["champion_damage"] for player in members),
        "vision_score": sum(player["vision_score"] for player in members),
        "lane_cs": sum(player["lane_cs"] for player in members),
        "jungle_cs": sum(player["jungle_cs"] for player in members),
        "barons": barons,
        "dragons": dragons,
        "towers": towers,
        "inhibitors": inhibitors,
        "rift_heralds": rift_heralds,
        "atakhan": atakhan,
        "first_blood": first_blood,
        "first_baron": first_baron,
        "first_dragon": first_dragon,
        "first_tower": first_tower,
        "first_inhibitor": first_inhibitor,
    }


def extract_match(match_id: str, payload: dict) -> ExtractionResult:
    """Validate a complete match, then return its normalized fact rows."""

    info = payload.get("info") if isinstance(payload, dict) else None
    if not isinstance(info, dict):
        raise ExtractionError("MISSING_FIELD", "info")
    if info.get("gameMode") != "CLASSIC":
        raise ExtractionError("UNSUPPORTED_GAME_MODE", "info.gameMode")

    version = info.get("gameVersion")
    if not isinstance(version, str):
        raise ExtractionError("MISSING_FIELD", "info.gameVersion")
    patch = _patch(version)
    teams = _validate_teams(info.get("teams"))
    participants = _validate_participants(info.get("participants"))
    winner_team_id = next(int(team["teamId"]) for team in teams if team.get("win"))

    player_rows = tuple(
        _player_row(match_id, participant, winner_team_id, index)
        for index, participant in enumerate(participants)
    )
    team_rows = tuple(
        _team_row(match_id, team, player_rows, winner_team_id) for team in teams
    )
    platform_prefix, region = _region(match_id)
    created_ms = _integer(info, "gameCreation", "info.gameCreation")
    duration_s = _integer(info, "gameDuration", "info.gameDuration")
    queue_id = info.get("queueId")
    if queue_id is not None:
        try:
            queue_id = int(queue_id)
        except (TypeError, ValueError) as exc:
            raise ExtractionError("INVALID_FIELD", "info.queueId") from exc

    match_row = {
        "match_id": match_id,
        "region": region,
        "platform_prefix": platform_prefix,
        "game_version": version,
        "patch": patch,
        "queue_id": queue_id,
        "game_creation_ms": created_ms,
        "game_duration_s": duration_s,
        "winner_team_id": winner_team_id,
        "participant_count": len(participants),
        "source_season_id": int(patch.split(".", 1)[0]),
    }
    return ExtractionResult(
        match=match_row,
        teams=(team_rows[0], team_rows[1]),
        players=player_rows,
    )
