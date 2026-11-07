import threading
import time
from unittest.mock import patch
import pytest

from core.collection.tracker import CollectionTracker, get_default_tracker


def test_tracker_initial_state():
    tracker = CollectionTracker()
    state = tracker.to_dict()

    assert state["status"] == "idle"
    assert state["total"] == 0
    assert state["current"] == 0
    assert state["player"] == ""
    assert state["logs"] == []
    assert state["speed_matches_per_min"] == 0.0
    assert state["eta_minutes"] is None
    assert state["elapsed_seconds"] == 0.0
    assert state["error"] is None


def test_tracker_lifecycle_running_and_completed():
    tracker = CollectionTracker()
    tracker.set_running(total=100, player="Faker#KR1")

    state = tracker.to_dict()
    assert state["status"] == "running"
    assert state["total"] == 100
    assert state["current"] == 0
    assert state["player"] == "Faker#KR1"

    tracker.step(increment=10)
    assert tracker.to_dict()["current"] == 10

    tracker.step()  # Default step is 1
    assert tracker.to_dict()["current"] == 11

    tracker.set_completed()
    completed_state = tracker.to_dict()
    assert completed_state["status"] == "completed"
    assert completed_state["current"] == 11
    assert completed_state["eta_minutes"] == 0.0


def test_tracker_set_player():
    tracker = CollectionTracker()
    tracker.set_running(total=50, player="Player1#EUW")
    assert tracker.to_dict()["player"] == "Player1#EUW"

    tracker.set_player("Player2#KR")
    assert tracker.to_dict()["player"] == "Player2#KR"


def test_tracker_error_handling():
    tracker = CollectionTracker()
    tracker.set_running(total=50, player="Test#KR")
    tracker.set_error("API key quota exceeded")

    state = tracker.to_dict()
    assert state["status"] == "error"
    assert state["error"] == "API key quota exceeded"
    assert any("API key quota exceeded" in log for log in state["logs"])


def test_tracker_rolling_log_buffer():
    tracker = CollectionTracker(max_logs=5)
    for i in range(12):
        tracker.add_log(f"Message {i}")

    state = tracker.to_dict()
    assert len(state["logs"]) == 5
    assert state["logs"] == [f"Message {i}" for i in range(7, 12)]


def test_tracker_speed_and_eta_calculation():
    tracker = CollectionTracker()

    with patch("time.time") as mock_time:
        # T0: Start collection of 120 matches
        mock_time.return_value = 1000.0
        tracker.set_running(total=120, player="Chovy#KR1")

        # T1: 60 seconds later, 30 matches collected
        mock_time.return_value = 1060.0
        tracker.step(increment=30)

        state = tracker.to_dict()
        assert state["elapsed_seconds"] == pytest.approx(60.0, rel=1e-3)
        # speed = (30 / 60) * 60 = 30.0 matches/min
        assert state["speed_matches_per_min"] == pytest.approx(30.0, rel=1e-3)
        # remaining = 120 - 30 = 90 matches. eta = 90 / 30 = 3.0 min
        assert state["eta_minutes"] == pytest.approx(3.0, rel=1e-3)

        # T2: When current >= total, ETA is 0.0
        mock_time.return_value = 1120.0
        tracker.step(increment=90)  # total 120
        state = tracker.to_dict()
        assert state["current"] == 120
        assert state["eta_minutes"] == 0.0


def test_tracker_zero_speed_eta():
    tracker = CollectionTracker()
    with patch("time.time") as mock_time:
        mock_time.return_value = 1000.0
        tracker.set_running(total=100)
        # 10s elapsed but 0 items processed -> speed is 0.0, eta is None
        mock_time.return_value = 1010.0
        state = tracker.to_dict()
        assert state["speed_matches_per_min"] == 0.0
        assert state["eta_minutes"] is None


def test_tracker_reset():
    tracker = CollectionTracker(max_logs=10)
    tracker.set_running(total=50, player="Test#KR")
    tracker.step(20)
    tracker.add_log("Processing")
    tracker.set_error("Failed")

    tracker.reset()
    state = tracker.to_dict()
    assert state["status"] == "idle"
    assert state["total"] == 0
    assert state["current"] == 0
    assert state["player"] == ""
    assert state["logs"] == []
    assert state["error"] is None
    assert state["speed_matches_per_min"] == 0.0
    assert state["eta_minutes"] is None


def test_tracker_thread_safety_concurrent_updates():
    tracker = CollectionTracker(max_logs=150)
    tracker.set_running(total=1000, player="MultiThread#KR")

    num_threads = 10
    steps_per_thread = 100

    def worker(worker_id: int):
        for i in range(steps_per_thread):
            tracker.step(1)
            tracker.add_log(f"Worker {worker_id} processed {i}")
            _ = tracker.to_dict()

    threads = [
        threading.Thread(target=worker, args=(tid,))
        for tid in range(num_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    state = tracker.to_dict()
    assert state["current"] == num_threads * steps_per_thread  # 1000
    assert len(state["logs"]) <= 150


def test_get_default_tracker_singleton():
    t1 = get_default_tracker()
    t2 = get_default_tracker()
    assert t1 is t2
    assert isinstance(t1, CollectionTracker)

    t3 = CollectionTracker()
    assert t3 is not t1
