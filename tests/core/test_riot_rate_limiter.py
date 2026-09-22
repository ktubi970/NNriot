from unittest.mock import MagicMock, patch
import pytest

from core.riot_api import RiotAPI, RiotRateLimiter


class VirtualClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_advances_virtual_clock_when_second_limit_is_reached():
    clock = VirtualClock()
    limiter = RiotRateLimiter(
        per_second=2,
        per_2min=10,
        clock=clock,
        sleep=clock.sleep,
    )

    limiter.acquire()
    limiter.acquire()
    limiter.acquire()

    assert len(clock.sleeps) == 1
    assert clock.sleeps[0] == pytest.approx(1.05)
    assert clock.now == pytest.approx(1.05)


def test_get_match_ids_supports_queue_and_type(monkeypatch):
    monkeypatch.setenv("RIOT_API_KEY", "test_key_123")
    api = RiotAPI(region="EUW")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = ["EUW1_123", "EUW1_456"]
    mock_resp.headers = {}

    with patch.object(api.session, "get", return_value=mock_resp) as mock_get:
        res = api.get_match_ids("fake_puuid", count=10, queue=420, type="ranked")

    assert res == ["EUW1_123", "EUW1_456"]
    assert mock_get.called
    called_url = mock_get.call_args[0][0]
    assert "queue=420" in called_url
    assert "type=ranked" in called_url
    assert "count=10" in called_url
