"""Shared analytics fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from analytics.types import EvidencePacket
from tests.analytics.factories import classic_match as make_classic_match
from tests.analytics.factories import seed_fact_rows
from tests.analytics.factories import seed_model_fact_rows
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


@pytest.fixture(scope="session")
def model_db(tmp_path_factory) -> Path:
    return seed_model_fact_rows(
        tmp_path_factory.mktemp("analytics-model") / "analytics.db"
    )


@pytest.fixture
def sample_evidence() -> EvidencePacket:
    return EvidencePacket(
        evidence_id="a" * 64,
        question="L'or est-il associé à la victoire en EUW ?",
        analysis_request={
            "intent": "explain_metric",
            "metric": "gold",
            "filters": {"regions": ["EUW"], "queue_ids": [420]},
        },
        cohort={
            "mode": "CLASSIC",
            "queue_ids": [420],
            "patches": ["16.14"],
            "regions": ["EUW"],
            "date_range": {"start_ms": None, "end_ms": None},
            "champion": None,
            "role": None,
            "widened": False,
        },
        sample={
            "matches": 600,
            "players": 300,
            "events_by_group": {"matches": 600},
        },
        estimate={
            "metric": "gold",
            "value": 850.25,
            "unit": "or",
            "baseline": 0.0,
            "effect_size": 0.42,
            "confidence_interval_95": [710.5, 990.0],
            "p_value": 0.001,
            "q_value": 0.004,
            "odds_ratio": None,
        },
        method={
            "name": "paired_winner_loser_bootstrap",
            "version": "1.0.0",
            "seed": 20260716,
            "bootstrap_resamples": 2000,
            "multiple_testing": "benjamini_hochberg",
            "candidate_family_size": 1,
        },
        robustness={
            "temporal_folds": [
                {"fold": 1, "matches": 300, "value": 820.0, "direction": 1},
                {"fold": 2, "matches": 300, "value": 880.0, "direction": 1},
            ],
            "regional_effects": [],
            "player_disjoint_check": {"available": False},
            "passed": True,
        },
        recommendation={
            "allowed": True,
            "strength": "moderate",
            "wording_key": "association_moderate",
        },
        limitations=("RETROSPECTIVE_FINAL_STATE_ASSOCIATION_ONLY",),
        provenance={
            "dataset_version": "b" * 64,
            "source_match_ids_hash": "c" * 64,
            "generated_at": "2026-07-17T12:00:00+00:00",
        },
    )
