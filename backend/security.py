import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    """Small in-memory sliding-window limiter (per process; fine for a single
    uvicorn worker - use Redis or a gateway if you scale out)."""

    def __init__(self, max_events, window_seconds):
        self.max_events = max_events
        self.window = window_seconds
        self._events = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, key, now):
        q = self._events[key]
        while q and now - q[0] > self.window:
            q.popleft()
        return q

    def retry_after(self, key):
        """Seconds until `key` may try again, or 0 if it is not blocked."""
        now = time.monotonic()
        with self._lock:
            q = self._prune(key, now)
            if len(q) >= self.max_events:
                return max(1, int(self.window - (now - q[0])))
            return 0

    def record(self, key):
        with self._lock:
            self._events[key].append(time.monotonic())

    def reset(self, key):
        with self._lock:
            self._events.pop(key, None)
