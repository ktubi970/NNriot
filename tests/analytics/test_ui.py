import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_analytics_page_is_french_and_contains_required_regions():
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True

    response = app.test_client().get("/analytics")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'lang="fr"' in html
    assert 'id="analytics-chat"' in html
    assert 'id="active-cohort"' in html
    assert 'id="evidence-list"' in html
    assert 'id="dataset-freshness"' in html
    assert "association" in html.lower()
    assert "causal" in html.lower()


def test_layout_links_to_analytics():
    from app import create_app

    app = create_app()
    app.config["TESTING"] = True

    html = app.test_client().get("/").get_data(as_text=True)

    assert 'href="/analytics"' in html


def test_browser_code_uses_text_content_not_html_injection():
    script = (open("static/analytics.js", encoding="utf-8").read())

    assert ".textContent" in script
    assert "innerHTML" not in script
    assert "/api/analytics/chat" in script
    assert "/api/analytics/health" in script
