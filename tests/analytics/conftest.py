"""Shared analytics fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.analytics.factories import classic_match as make_classic_match
from tests.analytics.factories import write_source_db


@pytest.fixture
def classic_match() -> dict:
    return make_classic_match()


@pytest.fixture
def source_db_factory(tmp_path):
    def factory(path: Path | None = None, match_count: int = 1) -> Path:
        target = Path(path) if path is not None else tmp_path / "source.db"
        matches = [
            make_classic_match(
                match_id=f"EUW1_{index + 1}",
                created_ms=index + 1,
            )
            for index in range(match_count)
        ]
        return write_source_db(target, matches)

    return factory


@pytest.fixture
def source_db(source_db_factory) -> Path:
    return source_db_factory()
