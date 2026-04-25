import pytest
from import_liquipedia import resolve_pro_name

def test_resolve_pro_name_success(monkeypatch):
    class MockResponse:
        def __init__(self, text):
            self.text = text
        def raise_for_status(self):
            pass

    def mock_get(url, headers):
        html = """
        <div class="infobox-description">Riot ID</div>
        <div>Razork#EUW</div>
        """
        return MockResponse(html)

    monkeypatch.setattr("import_liquipedia.requests.get", mock_get)

    result = resolve_pro_name("Razork")
    assert result == ("Razork", "EUW")

def test_resolve_pro_name_not_found(monkeypatch):
    class MockResponse:
        def __init__(self, text):
            self.text = text
        def raise_for_status(self):
            pass

    def mock_get(url, headers):
        html = "<div>No id here</div>"
        return MockResponse(html)

    monkeypatch.setattr("import_liquipedia.requests.get", mock_get)

    result = resolve_pro_name("UnknownPlayer")
    assert result is None
