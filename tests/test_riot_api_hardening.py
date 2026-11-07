import os
from unittest.mock import MagicMock, patch
import pytest
import requests

from core.riot_api import RiotAPI, RiotAuthError


def test_riot_api_raises_auth_error_on_401():
    with patch.dict("os.environ", {"RIOT_API_KEY": "RGAPI-test-expired", "RIOT_DISABLE_RATE_LIMIT": "1"}):
        api = RiotAPI(region="EUW")
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.headers = {}

        with patch.object(api.session, "get", return_value=mock_resp):
            with patch.object(api, "reload_key", return_value=None):
                with pytest.raises(RiotAuthError) as exc_info:
                    api._make_request("https://europe.api.riotgames.com/test")
                assert "401" in str(exc_info.value) or "unauthorized" in str(exc_info.value).lower()
        api.close()


def test_riot_api_raises_auth_error_on_403():
    with patch.dict("os.environ", {"RIOT_API_KEY": "RGAPI-test-forbidden", "RIOT_DISABLE_RATE_LIMIT": "1"}):
        api = RiotAPI(region="EUW")
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.headers = {}

        with patch.object(api.session, "get", return_value=mock_resp):
            with patch.object(api, "reload_key", return_value=None):
                with pytest.raises(RiotAuthError) as exc_info:
                    api._make_request("https://europe.api.riotgames.com/test")
                assert "403" in str(exc_info.value) or "rejected" in str(exc_info.value).lower()
        api.close()


def test_riot_api_uses_persistent_session():
    with patch.dict("os.environ", {"RIOT_API_KEY": "RGAPI-test-key", "RIOT_DISABLE_RATE_LIMIT": "1"}):
        api = RiotAPI(region="KR")
        assert hasattr(api, "session")
        assert isinstance(api.session, requests.Session)
        api.close()


def test_riot_api_close_delegates_to_session():
    with patch.dict("os.environ", {"RIOT_API_KEY": "RGAPI-test-key", "RIOT_DISABLE_RATE_LIMIT": "1"}):
        api = RiotAPI(region="EUW")
        with patch.object(api.session, "close") as mock_close:
            api.close()
            mock_close.assert_called_once()


def test_riot_api_context_manager():
    with patch.dict("os.environ", {"RIOT_API_KEY": "RGAPI-test-key", "RIOT_DISABLE_RATE_LIMIT": "1"}):
        with patch.object(requests.Session, "close") as mock_close:
            with RiotAPI(region="NA") as api:
                assert isinstance(api.session, requests.Session)
        mock_close.assert_called_once()


def test_riot_api_reloads_key_and_retries_on_auth_failure():
    with patch.dict("os.environ", {"RIOT_API_KEY": "RGAPI-old-key", "RIOT_DISABLE_RATE_LIMIT": "1"}):
        api = RiotAPI(region="EUW")

        mock_resp_401 = MagicMock()
        mock_resp_401.status_code = 401
        mock_resp_401.headers = {}

        mock_resp_200 = MagicMock()
        mock_resp_200.status_code = 200
        mock_resp_200.headers = {}

        calls = []

        def fake_get(url, headers=None, timeout=None):
            calls.append(dict(headers))
            if len(calls) == 1:
                return mock_resp_401
            return mock_resp_200

        def fake_reload():
            api.api_key = "RGAPI-new-key"
            return "RGAPI-new-key"

        with patch.object(api.session, "get", side_effect=fake_get):
            with patch.object(api, "reload_key", side_effect=fake_reload):
                resp = api._make_request("https://europe.api.riotgames.com/test")

        assert resp == mock_resp_200
        assert len(calls) == 2
        assert calls[0].get("X-Riot-Token") == "RGAPI-old-key"
        assert calls[1].get("X-Riot-Token") == "RGAPI-new-key"
        api.close()
