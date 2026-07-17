"""Allowlisted metrics and leakage classification."""

from dataclasses import dataclass
from enum import Enum


class MetricKind(str, Enum):
    CONTINUOUS = "continuous"
    BINARY = "binary"


class FeatureClass(str, Enum):
    PRE_MATCH = "pre_match"
    IN_GAME_TEMPORAL = "in_game_temporal"
    FINAL_GAME_RETROSPECTIVE = "final_game_retrospective"
    TARGET_FORBIDDEN = "target_forbidden"


@dataclass(frozen=True)
class MetricSpec:
    name: str
    column: str
    kind: MetricKind
    unit: str
    feature_class: FeatureClass
    aliases: tuple[str, ...]


def _continuous(name: str, unit: str, *aliases: str) -> MetricSpec:
    return MetricSpec(
        name=name,
        column=name,
        kind=MetricKind.CONTINUOUS,
        unit=unit,
        feature_class=FeatureClass.FINAL_GAME_RETROSPECTIVE,
        aliases=tuple(aliases),
    )


def _binary(name: str, column: str, *aliases: str) -> MetricSpec:
    return MetricSpec(
        name=name,
        column=column,
        kind=MetricKind.BINARY,
        unit="points de victoire",
        feature_class=FeatureClass.FINAL_GAME_RETROSPECTIVE,
        aliases=tuple(aliases),
    )


METRICS = {
    spec.name: spec
    for spec in (
        _continuous("kills", "eliminations", "kills", "éliminations"),
        _continuous("deaths", "morts", "deaths", "morts"),
        _continuous("assists", "assistances", "assists", "assistances"),
        _continuous("gold", "or", "gold", "or"),
        _continuous("champion_damage", "dégâts", "damage", "dégâts"),
        _continuous("vision_score", "points de vision", "vision"),
        _continuous("lane_cs", "sbires de voie", "cs", "farm"),
        _continuous("jungle_cs", "monstres de jungle", "jungle farm"),
        _continuous("barons", "barons", "baron"),
        _continuous("dragons", "dragons", "dragon"),
        _continuous("towers", "tours", "tour", "tours"),
        _continuous("inhibitors", "inhibiteurs", "inhibiteur"),
        _continuous("rift_heralds", "hérauts", "héraut"),
        _continuous("atakhan", "Atakhan", "atakhan"),
        _binary("first_blood", "first_blood", "premier sang", "first blood"),
        _binary("first_baron", "first_baron", "premier baron"),
        _binary("first_dragon", "first_dragon", "premier dragon"),
        _binary("first_tower", "first_tower", "première tour"),
        _binary(
            "first_inhibitor",
            "first_inhibitor",
            "premier inhibiteur",
        ),
    )
}


def get_metric(name: str) -> MetricSpec:
    try:
        return METRICS[name]
    except KeyError as exc:
        raise ValueError(f"UNSUPPORTED_FILTER: metric={name}") from exc
