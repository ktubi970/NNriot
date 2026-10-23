"""Thread-safe in-memory status tracker for unified collection operations."""

from collections import deque
import threading
import time
from typing import Optional


class CollectionTracker:
    """Thread-safe tracker for collection progress, speed, ETA, and logging."""

    def __init__(self, max_logs: int = 150) -> None:
        self._max_logs = max(1, max_logs)
        self._lock = threading.Lock()
        self._status: str = "idle"  # idle | running | completed | error
        self._total: int = 0
        self._current: int = 0
        self._player: str = ""
        self._logs: deque[str] = deque(maxlen=self._max_logs)
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._error: Optional[str] = None

    @property
    def status(self) -> str:
        """Return the current collection status in a thread-safe manner."""
        with self._lock:
            return self._status

    def set_running(self, total: int = 0, player: str = "") -> None:
        """Start tracking a new collection job."""
        with self._lock:
            self._status = "running"
            self._total = max(0, total)
            self._current = 0
            self._player = player
            self._logs.clear()
            self._start_time = time.time()
            self._end_time = None
            self._error = None

    def set_completed(self) -> None:
        """Mark collection as successfully completed."""
        with self._lock:
            self._status = "completed"
            self._end_time = time.time()

    def set_error(self, err_msg: str) -> None:
        """Mark collection as failed with an error message."""
        with self._lock:
            self._status = "error"
            self._error = str(err_msg)
            self._end_time = time.time()
            self._logs.append(f"ERROR: {err_msg}")

    def set_player(self, player: str) -> None:
        """Update the currently active player being processed."""
        with self._lock:
            self._player = str(player)

    def step(self, increment: int = 1) -> None:
        """Increment the current progress counter."""
        with self._lock:
            self._current += increment

    def add_log(self, msg: str) -> None:
        """Append a message to the rolling log buffer."""
        with self._lock:
            self._logs.append(str(msg))

    def reset(self) -> None:
        """Reset the tracker back to initial idle state."""
        with self._lock:
            self._status = "idle"
            self._total = 0
            self._current = 0
            self._player = ""
            self._logs.clear()
            self._start_time = None
            self._end_time = None
            self._error = None

    def to_dict(self) -> dict:
        """Return a thread-safe snapshot dictionary of tracker status and metrics."""
        with self._lock:
            if self._start_time is None:
                elapsed_seconds = 0.0
            elif self._end_time is not None:
                elapsed_seconds = max(0.0, self._end_time - self._start_time)
            else:
                elapsed_seconds = max(0.0, time.time() - self._start_time)

            if elapsed_seconds > 0.0 and self._current > 0:
                speed_matches_per_min = (self._current / elapsed_seconds) * 60.0
            else:
                speed_matches_per_min = 0.0

            remaining = max(0, self._total - self._current)
            if self._status == "completed" or (self._total > 0 and remaining == 0):
                eta_minutes = 0.0
            elif speed_matches_per_min > 0.0:
                eta_minutes = remaining / speed_matches_per_min
            else:
                eta_minutes = None

            return {
                "status": self._status,
                "total": self._total,
                "current": self._current,
                "player": self._player,
                "logs": list(self._logs),
                "speed_matches_per_min": speed_matches_per_min,
                "eta_minutes": eta_minutes,
                "elapsed_seconds": elapsed_seconds,
                "error": self._error,
            }


_default_tracker: Optional[CollectionTracker] = None
_default_lock = threading.Lock()


def get_default_tracker() -> CollectionTracker:
    """Return the global default CollectionTracker singleton instance."""
    global _default_tracker
    if _default_tracker is None:
        with _default_lock:
            if _default_tracker is None:
                _default_tracker = CollectionTracker()
    return _default_tracker
