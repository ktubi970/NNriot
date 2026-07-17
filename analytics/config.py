"""Configuration and statistical thresholds for analytics."""

from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class AnalyticsConfig:
    """Environment-backed configuration with explicit evidence thresholds."""

    source_path: str = field(
        default_factory=lambda: os.getenv("NNRIOT_DB_PATH", "training_data.db")
    )
    analytics_path: str = field(
        default_factory=lambda: os.getenv(
            "NNRIOT_ANALYTICS_DB_PATH", "analytics.db"
        )
    )
    seed: int = field(
        default_factory=lambda: int(
            os.getenv("NNRIOT_ANALYTICS_SEED", "20260716")
        )
    )
    bootstrap_resamples: int = 2000
    min_matches: int = 500
    min_players: int = 100
    min_group_events: int = 100
    strong_matches: int = 2000
    method_version: str = "1.0.0"
    extractor_version: str = "1.0.0"
