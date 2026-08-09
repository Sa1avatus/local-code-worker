import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock

from ..providers.base import ProviderMessage


class PreviousResponseNotFound(ValueError):
    pass


@dataclass(frozen=True)
class StoredResponse:
    messages: tuple[ProviderMessage, ...]
    expires_at: float


class ResponseStateStore:
    def __init__(
        self,
        *,
        max_entries: int = 128,
        ttl_seconds: float = 900,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._items: OrderedDict[str, StoredResponse] = OrderedDict()
        self._lock = Lock()

    def put(self, response_id: str, messages: list[ProviderMessage]) -> None:
        with self._lock:
            self._prune_expired()
            self._items[response_id] = StoredResponse(
                messages=tuple(messages),
                expires_at=self._clock() + self.ttl_seconds,
            )
            self._items.move_to_end(response_id)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def get(self, response_id: str) -> list[ProviderMessage]:
        with self._lock:
            self._prune_expired()
            stored = self._items.get(response_id)
            if stored is None:
                raise PreviousResponseNotFound(
                    f"previous_response_id was not found or expired: {response_id}"
                )
            self._items.move_to_end(response_id)
            return list(stored.messages)

    def _prune_expired(self) -> None:
        now = self._clock()
        expired = [key for key, value in self._items.items() if value.expires_at <= now]
        for key in expired:
            del self._items[key]
