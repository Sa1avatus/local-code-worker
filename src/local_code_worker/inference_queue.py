from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Condition, Event, Thread

logger = logging.getLogger(__name__)


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
        # Delayed unload support
        self._unload_policy: str = "immediate"  # "immediate", "never", or minutes
        self._unload_timer: Event | None = None
        self._unload_thread: Thread | None = None

    def set_unload_policy(self, policy: str) -> None:
        """Set the unload policy. 'immediate', 'never', or a number of minutes."""
        with self._condition:
            self._unload_policy = policy
            if policy == "immediate" or policy == "never":
                self._cancel_pending_unload()

    def _cancel_pending_unload(self) -> None:
        """Cancel any pending delayed unload."""
        if self._unload_timer is not None:
            self._unload_timer.set()
            self._unload_timer = None

    def _schedule_delayed_unload(self, cleanup: Callable[[], None], minutes: float) -> None:
        """Schedule a delayed unload after the given number of minutes."""
        self._cancel_pending_unload()
        timer = Event()
        self._unload_timer = timer

        def _delayed() -> None:
            if timer.wait(timeout=minutes * 60):
                # Timer was cancelled
                return
            with self._condition:
                if self._waiting == 0 and self._unload_timer is timer:
                    self._unload_timer = None
                    try:
                        cleanup()
                        logger.info("Model unloaded after %s minutes of inactivity", minutes)
                    except Exception:
                        logger.warning("Delayed unload failed", exc_info=True)

        thread = Thread(target=_delayed, daemon=True)
        self._unload_thread = thread
        thread.start()

    @contextmanager
    def acquire(self) -> Iterator[InferenceLease]:
        lease = InferenceLease()
        with self._condition:
            # Cancel any pending delayed unload since we have a new request
            self._cancel_pending_unload()
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
                self._handle_idle_cleanup(lease)
                self._active = False
                self._started_at = None
                self._lease = None
                self._last_completed_at = datetime.now(UTC).isoformat()
                self._condition.notify_all()

    def _handle_idle_cleanup(self, lease: InferenceLease) -> None:
        """Handle idle cleanup based on the current unload policy."""
        if lease.idle_cleanup is None or self._waiting > 0:
            return

        policy = self._unload_policy

        if policy == "never":
            return
        elif policy == "immediate":
            try:
                lease.idle_cleanup()
            except Exception:
                pass
        else:
            # Try to parse as minutes
            try:
                minutes = float(policy)
                if minutes <= 0:
                    # Treat <= 0 as immediate
                    try:
                        lease.idle_cleanup()
                    except Exception:
                        pass
                else:
                    self._schedule_delayed_unload(lease.idle_cleanup, minutes)
            except (ValueError, TypeError):
                # Invalid policy, fall back to immediate
                try:
                    lease.idle_cleanup()
                except Exception:
                    pass

    def status(self) -> dict[str, object]:
        with self._condition:
            return {
                "active": self._active,
                "waiting": self._waiting,
                "started_at": self._started_at,
                "last_completed_at": self._last_completed_at,
                "model": self._lease.model if self._lease else None,
                "route": self._lease.route if self._lease else None,
                "unload_policy": self._unload_policy,
            }
