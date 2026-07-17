"""Shared analytics fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.analytics.factories import classic_match as make_classic_match
from tests.analytics.factories import seed_fact_rows
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


@pytest.fixture(scope="session")
def cohort_db(tmp_path_factory) -> Path:
    return seed_fact_rows(
        tmp_path_factory.mktemp("cohort-powered") / "analytics.db",
        {"16.14": 500},
    )


@pytest.fixture(scope="session")
def cohort_db_underpowered_latest(tmp_path_factory) -> Path:
    return seed_fact_rows(
        tmp_path_factory.mktemp("cohort-widened") / "analytics.db",
        {"16.14": 200, "16.13": 400},
    )


@pytest.fixture(scope="session")
def cohort_db_tiny(tmp_path_factory) -> Path:
    return seed_fact_rows(
        tmp_path_factory.mktemp("cohort-tiny") / "analytics.db",
        {"16.14": 100, "16.13": 100, "16.12": 100},
    )


@pytest.fixture(scope="session")
def powered_analytics_db(tmp_path_factory) -> Path:
    return seed_fact_rows(
        tmp_path_factory.mktemp("analytics-powered") / "analytics.db",
        {"16.12": 700, "16.13": 700, "16.14": 700},
        region_shifts={"EUW": 10.0, "KR": 8.0, "NA": 9.0, "EUN": 7.0},
    )
