"""Deterministic temporal and player-disjoint evaluation splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IndexSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def __iter__(self):
        yield self.train
        yield self.validation
        yield self.test


@dataclass(frozen=True)
class GroupedSplit:
    train: np.ndarray
    test: np.ndarray
    discarded: np.ndarray
    available: bool
    code: str | None


def _indices(values: list[int]) -> np.ndarray:
    result = np.asarray(values, dtype=int)
    result.setflags(write=False)
    return result


def chronological_split(order: np.ndarray) -> IndexSplit:
    indices = np.asarray(order, dtype=int)
    if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
        raise ValueError("SOURCE_DATA_INVALID")
    first = int(len(indices) * 0.70)
    second = int(len(indices) * 0.85)
    return IndexSplit(
        _indices(indices[:first].tolist()),
        _indices(indices[first:second].tolist()),
        _indices(indices[second:].tolist()),
    )


def player_disjoint_split(
    match_puuids: tuple[frozenset[str], ...],
    min_test_matches: int = 500,
) -> GroupedSplit:
    if min_test_matches < 1:
        raise ValueError("SOURCE_DATA_INVALID")
    target_test = max(min_test_matches, int(len(match_puuids) * 0.15))
    test_start = max(0, len(match_puuids) - target_test)
    test = list(range(test_start, len(match_puuids)))
    test_players = (
        set().union(*(match_puuids[index] for index in test)) if test else set()
    )
    train: list[int] = []
    discarded: list[int] = []
    for index in range(test_start):
        if match_puuids[index] & test_players:
            discarded.append(index)
        else:
            train.append(index)
    available = len(test) >= min_test_matches and bool(train)
    return GroupedSplit(
        train=_indices(train),
        test=_indices(test),
        discarded=_indices(discarded),
        available=available,
        code=None if available else "INSUFFICIENT_MATCHES",
    )
