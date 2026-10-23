from dataclasses import dataclass, field
import os

from core import config as core_config


@dataclass(frozen=True)
class AnalyticsConfig:
    """Configuration and evidence thresholds for analytics."""

    source_path: str = field(default_factory=core_config.get_db_path)
    analytics_path: str = field(default_factory=core_config.get_analytics_db_path)
    seed: int = field(
        default_factory=lambda: int(os.getenv("NNRIOT_ANALYTICS_SEED", "20260716"))
    )
    bootstrap_resamples: int = 2000
    min_matches: int = 500
    min_players: int = 100
    min_group_events: int = 100
    strong_matches: int = 2000
    method_version: str = "1.0.0"
    extractor_version: str = "1.0.0"
