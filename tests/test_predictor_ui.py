"""Test that the predictor template ships the multi-output rendering hooks."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pytest

from app import app as flask_app
from app import core as app_core


@pytest.fixture
def client(monkeypatch):
    from unittest.mock import MagicMock
    monkeypatch.setattr(app_core, "global_trainer", MagicMock())
    monkeypatch.setattr(app_core, "tf_available", True)
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


def test_predictor_html_contains_markets_container(client):
    resp = client.get("/predictor")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert 'id="markets-container"' in body, "markets-container div missing"


def test_predictor_html_contains_all_market_sections(client):
    resp = client.get("/predictor")
    body = resp.get_data(as_text=True)
    # IDs for the 5 sections (one per market group)
    required_ids = [
        "market-total-kills",
        "market-team-a-kills",
        "market-team-b-kills",
        "market-handicap",
        "market-kills-odd-bar",
        "market-team-b-kill-lead-bar",
        "market-first-blood-a-bar",
        "market-first-baron-a-bar",
        "market-first-inhibitor-a-bar",
        "market-first-tower-a-bar",
        "market-total-barons",
        "market-total-dragons",
        "market-total-towers",
        "market-both-baron-bar",
        "market-both-inhibitor-bar",
        "market-both-dragon-bar",
        "market-elder-dragon-bar",
        # Timeline first-to-N-kills cards (12 IDs: 4 thresholds x {a,b,n})
        "market-first-kills_5-a-bar",
        "market-first-kills_5-b-bar",
        "market-first-kills_5-n-bar",
        "market-first-kills_10-a-bar",
        "market-first-kills_10-b-bar",
        "market-first-kills_10-n-bar",
        "market-first-kills_15-a-bar",
        "market-first-kills_15-b-bar",
        "market-first-kills_15-n-bar",
        "market-first-kills_20-a-bar",
        "market-first-kills_20-b-bar",
        "market-first-kills_20-n-bar",
    ]
    for el_id in required_ids:
        assert f'id="{el_id}"' in body, f"required element id={el_id} not found"


def test_predictor_html_includes_render_function(client):
    resp = client.get("/predictor")
    body = resp.get_data(as_text=True)
    assert "function renderMultiOutput" in body, "renderMultiOutput function not in JS"


def test_predictor_html_no_old_winner_kills_reference(client):
    """After the rename, the template should not reference winner_kills."""
    resp = client.get("/predictor")
    body = resp.get_data(as_text=True)
    assert "winner_kills" not in body, "old `winner_kills` reference still in HTML/JS"
