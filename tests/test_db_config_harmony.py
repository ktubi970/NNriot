import os
from pathlib import Path
import pytest
from unittest.mock import patch

def test_analytics_config_prefers_nnriot_prefix_and_falls_back():
    from analytics.config import AnalyticsConfig
    root = Path(__file__).resolve().parents[1]
    
    with patch.dict(os.environ, {"NNRIOT_ANALYTICS_DB_PATH": "custom_nn.db", "ANALYTICS_DB_PATH": "custom_plain.db"}):
        cfg = AnalyticsConfig()
        assert cfg.analytics_path == str(root / "custom_nn.db")

    with patch.dict(os.environ, {"ANALYTICS_DB_PATH": "custom_plain.db"}, clear=True):
        cfg = AnalyticsConfig()
        assert cfg.analytics_path == str(root / "custom_plain.db")

    with patch.dict(os.environ, {}, clear=True):
        cfg = AnalyticsConfig()
        assert cfg.analytics_path == str(root / "analytics.db")

def test_core_config_harmony():
    import importlib
    import core.config
    root = Path(__file__).resolve().parents[1]
    
    with patch.dict(os.environ, {"NNRIOT_ANALYTICS_DB_PATH": "custom_nn.db", "ANALYTICS_DB_PATH": "custom_plain.db"}):
        importlib.reload(core.config)
        assert core.config.ANALYTICS_DB_PATH == str(root / "custom_nn.db")

    with patch.dict(os.environ, {"ANALYTICS_DB_PATH": "custom_plain.db"}, clear=True):
        importlib.reload(core.config)
        assert core.config.ANALYTICS_DB_PATH == str(root / "custom_plain.db")

    with patch.dict(os.environ, {}, clear=True):
        importlib.reload(core.config)
        assert core.config.ANALYTICS_DB_PATH == str(root / "analytics.db")
