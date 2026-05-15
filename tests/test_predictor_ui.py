"""Test that the predictor template ships the multi-output rendering hooks."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pytest


@pytest.fixture
def client(monkeypatch):
    from unittest.mock import MagicMock
    import final_web_app
    monkeypatch.setattr(final_web_app, "global_trainer", MagicMock())
    monkeypatch.setattr(final_web_app, "tf_available", True)
    final_web_app.app.config["TESTING"] = True
    return final_web_app.app.test_client()


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
