import time
from unittest.mock import MagicMock, patch
import pytest

from core.riot_api import RiotAPI, RiotRateLimiter


def test_rate_limiter_acquires_under_limit():
    limiter = RiotRateLimiter(per_second=5, per_2min=10)
    t0 = time.time()
    for _ in range(4):
        limiter.acquire()
    assert time.time() - t0 < 0.5


def test_rate_limiter_delays_when_second_limit_reached():
    limiter = RiotRateLimiter(per_second=2, per_2min=10)
    t0 = time.time()
    limiter.acquire()
    limiter.acquire()
    # 3rd acquire in same second should wait until 1s elapsed
    limiter.acquire()
    elapsed = time.time() - t0
    assert elapsed >= 0.9


def test_get_match_ids_supports_queue_and_type():
    api = RiotAPI(region="EUW")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = ["EUW1_123", "EUW1_456"]
    mock_resp.headers = {}

    with patch("core.riot_api.requests.get", return_value=mock_resp) as mock_get:
        res = api.get_match_ids("fake_puuid", count=10, queue=420, type="ranked")

    assert res == ["EUW1_123", "EUW1_456"]
    assert mock_get.called
    called_url = mock_get.call_args[0][0]
    assert "queue=420" in called_url
    assert "type=ranked" in called_url
    assert "count=10" in called_url
