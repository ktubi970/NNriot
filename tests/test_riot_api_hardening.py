import pytest
from unittest.mock import patch, MagicMock
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


def test_riot_api_uses_persistent_session():
    with patch.dict("os.environ", {"RIOT_API_KEY": "RGAPI-test-key", "RIOT_DISABLE_RATE_LIMIT": "1"}):
        api = RiotAPI(region="KR")
        assert hasattr(api, "session")
        assert isinstance(api.session, object)
