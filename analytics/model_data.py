"""Leakage-safe named matrix for retrospective interpretable models."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from .db import connect_analytics


MODEL_FEATURES = (
    "kills",
    "deaths",
    "assists",
    "gold",
    "champion_damage",
    "vision_score",
    "lane_cs",
    "jungle_cs",
    "barons",
    "dragons",
    "towers",
    "inhibitors",
    "rift_heralds",
    "atakhan",
    "first_blood",
    "first_baron",
    "first_dragon",
    "first_tower",
    "first_inhibitor",
)
FORBIDDEN_FEATURES = {"win", "winner_team_id", "target", "winner_label"}


@dataclass(frozen=True)
class ModelDataset:
    x: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    match_ids: np.ndarray
    creation_ms: np.ndarray
    regions: np.ndarray
    patches: np.ndarray
    match_puuids: tuple[frozenset[str], ...]


def assert_allowed_features(names: tuple[str, ...]) -> None:
    forbidden = {
        name
        for name in names
        if name in FORBIDDEN_FEATURES
        or name.removesuffix("_delta") in FORBIDDEN_FEATURES
    }
    if forbidden:
        raise ValueError("TARGET_FORBIDDEN: " + ",".join(sorted(forbidden)))
    allowed = {f"{name}_delta" for name in MODEL_FEATURES}
    unknown = set(names) - allowed
    if unknown:
        raise ValueError(
            "UNREGISTERED_MODEL_FEATURE: " + ",".join(sorted(unknown))
        )


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def rows_to_matrix(rows: Iterable[tuple]) -> ModelDataset:
    source_rows = list(rows)
    names = tuple(f"{name}_delta" for name in MODEL_FEATURES)
    assert_allowed_features(names)
    x = np.empty((len(source_rows), len(MODEL_FEATURES)), dtype=float)
    for row_index, (_, blue, red, _, _, _, _) in enumerate(source_rows):
        x[row_index] = [
            float(blue[name]) - float(red[name]) for name in MODEL_FEATURES
        ]
    y = np.asarray(
        [blue_win for _, _, _, blue_win, _, _, _ in source_rows], dtype=np.int8
    )
    if not np.isfinite(x).all() or not set(np.unique(y).tolist()) <= {0, 1}:
        raise ValueError("SOURCE_DATA_INVALID")
    return ModelDataset(
        x=_readonly(x),
        y=_readonly(y),
        feature_names=names,
        match_ids=_readonly(
            np.asarray([match_id for match_id, *_ in source_rows], dtype=str)
        ),
        creation_ms=_readonly(
            np.asarray(
                [created for *_, created in source_rows], dtype=np.int64
            )
        ),
        regions=_readonly(
            np.asarray(
                [region for _, _, _, _, region, _, _ in source_rows], dtype=str
            )
        ),
        patches=_readonly(
            np.asarray(
                [patch for _, _, _, _, _, patch, _ in source_rows], dtype=str
            )
        ),
        match_puuids=tuple(frozenset() for _ in source_rows),
    )


def load_model_dataset(path: str) -> ModelDataset:
    """Load one chronological blue-minus-red row per analytical match."""

    columns = ",".join(
        f"blue.{name} AS blue_{name},red.{name} AS red_{name}"
        for name in MODEL_FEATURES
    )
    with connect_analytics(path, readonly=True) as conn:
        expected = int(conn.execute("SELECT COUNT(*) FROM match_facts").fetchone()[0])
        records = conn.execute(
            "SELECT m.match_id,m.region,m.patch,m.game_creation_ms,blue.win AS blue_win,"
            + columns
            + " FROM match_facts m "
            "JOIN team_facts blue ON blue.match_id=m.match_id AND blue.team_id=100 "
            "JOIN team_facts red ON red.match_id=m.match_id AND red.team_id=200 "
            "ORDER BY m.game_creation_ms,m.match_id"
        ).fetchall()
        participant_rows = conn.execute(
            "SELECT match_id,puuid FROM player_facts ORDER BY match_id,puuid"
        ).fetchall()
    if len(records) != expected:
        raise ValueError("SOURCE_DATA_INVALID: missing team facts")

    puuids_by_match: dict[str, set[str]] = defaultdict(set)
    for row in participant_rows:
        puuids_by_match[str(row["match_id"])].add(str(row["puuid"]))
    rows = []
    match_puuids: list[frozenset[str]] = []
    for record in records:
        match_id = str(record["match_id"])
        puuids = frozenset(puuids_by_match[match_id])
        if len(puuids) != 10:
            raise ValueError("SOURCE_DATA_INVALID: participant identities")
        blue = {
            name: float(record[f"blue_{name}"]) for name in MODEL_FEATURES
        }
        red = {name: float(record[f"red_{name}"]) for name in MODEL_FEATURES}
        rows.append(
            (
                match_id,
                blue,
                red,
                int(record["blue_win"]),
                str(record["region"]),
                str(record["patch"]),
                int(record["game_creation_ms"]),
            )
        )
        match_puuids.append(puuids)
    dataset = rows_to_matrix(rows)
    return replace(dataset, match_puuids=tuple(match_puuids))
