import pytest

from local_code_worker.providers.base import ProviderMessage
from local_code_worker.responses.state import (
    PreviousResponseNotFound,
    ResponseStateStore,
)


def test_response_state_expires_entries() -> None:
    now = [10.0]
    store = ResponseStateStore(ttl_seconds=5, clock=lambda: now[0])
    store.put("resp_1", [ProviderMessage(role="user", content="hello")])

    assert store.get("resp_1")[0].content == "hello"
    now[0] = 15.0

    with pytest.raises(PreviousResponseNotFound, match="not found or expired"):
        store.get("resp_1")


def test_response_state_evicts_least_recently_used_entry() -> None:
    store = ResponseStateStore(max_entries=2)
    message = [ProviderMessage(role="user", content="hello")]
    store.put("resp_1", message)
    store.put("resp_2", message)
    store.get("resp_1")
    store.put("resp_3", message)

    with pytest.raises(PreviousResponseNotFound):
        store.get("resp_2")
    assert store.get("resp_1")[0].content == "hello"
    assert store.get("resp_3")[0].content == "hello"
