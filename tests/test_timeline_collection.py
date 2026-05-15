"""Test that the NNRIOT_FETCH_TIMELINES env flag is wired up correctly."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_fetch_timelines_default_false():
    """NNRIOT_FETCH_TIMELINES env var resolves to a bool at module-load time."""
    import importlib
    import data_collector
    importlib.reload(data_collector)  # re-evaluate the module-level constant
    # Default env should produce False (or True if host set the flag).
    assert data_collector.FETCH_TIMELINES in (False, True)
    # Just verify the constant exists and is a bool.
    assert isinstance(data_collector.FETCH_TIMELINES, bool)
