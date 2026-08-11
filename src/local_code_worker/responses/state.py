import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock

from ..providers.base import ProviderMessage
from ..routing.models import RouteLease


class PreviousResponseNotFound(ValueError):
    pass


@dataclass(frozen=True)
class StoredResponse:
    messages: tuple[ProviderMessage, ...]
    route_lease: RouteLease | None
    expires_at: float


class ResponseStateStore:
    def __init__(
        self,
        *,
        max_entries: int = 256,
        ttl_seconds: float = 7200,
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

    def put(
        self,
        response_id: str,
        messages: list[ProviderMessage],
        route_lease: RouteLease | None = None,
    ) -> None:
        with self._lock:
            self._prune_expired()
            self._items[response_id] = StoredResponse(
                messages=tuple(messages),
                route_lease=route_lease,
                expires_at=self._clock() + self.ttl_seconds,
            )
            self._items.move_to_end(response_id)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def get(self, response_id: str) -> list[ProviderMessage]:
        return list(self.get_stored(response_id).messages)

    def get_stored(self, response_id: str) -> StoredResponse:
        with self._lock:
            self._prune_expired()
            stored = self._items.get(response_id)
            if stored is None:
                raise PreviousResponseNotFound(
                    f"previous_response_id was not found or expired: {response_id}"
                )
            self._items.move_to_end(response_id)
            return stored

    def _prune_expired(self) -> None:
        now = self._clock()
        expired = [key for key, value in self._items.items() if value.expires_at <= now]
        for key in expired:
            del self._items[key]
