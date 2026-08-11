from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Condition


@dataclass
class InferenceLease:
    model: str | None = None
    route: str | None = None
    idle_cleanup: Callable[[], None] | None = None


class InferenceQueue:
    def __init__(self) -> None:
        self._condition = Condition()
        self._active = False
        self._waiting = 0
        self._started_at: str | None = None
        self._last_completed_at: str | None = None
        self._lease: InferenceLease | None = None

    @contextmanager
    def acquire(self) -> Iterator[InferenceLease]:
        lease = InferenceLease()
        with self._condition:
            self._waiting += 1
            try:
                while self._active:
                    self._condition.wait()
                self._active = True
                self._started_at = datetime.now(UTC).isoformat()
                self._lease = lease
            finally:
                self._waiting -= 1
        try:
            yield lease
        finally:
            with self._condition:
                if self._waiting == 0 and lease.idle_cleanup is not None:
                    try:
                        lease.idle_cleanup()
                    except Exception:
                        pass
                self._active = False
                self._started_at = None
                self._lease = None
                self._last_completed_at = datetime.now(UTC).isoformat()
                self._condition.notify_all()

    def status(self) -> dict[str, object]:
        with self._condition:
            return {
                "active": self._active,
                "waiting": self._waiting,
                "started_at": self._started_at,
                "last_completed_at": self._last_completed_at,
                "model": self._lease.model if self._lease else None,
                "route": self._lease.route if self._lease else None,
            }
