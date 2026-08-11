import threading

from local_code_worker.inference_queue import InferenceQueue


def test_queue_reports_activity_and_runs_cleanup_when_idle() -> None:
    queue = InferenceQueue()
    cleaned = []

    with queue.acquire() as lease:
        lease.model = "qwen:test"
        lease.route = "local"
        lease.idle_cleanup = lambda: cleaned.append("qwen:test")
        status = queue.status()
        assert status["active"] is True
        assert status["waiting"] == 0
        assert status["model"] == "qwen:test"
        assert status["route"] == "local"

    assert cleaned == ["qwen:test"]
    assert queue.status()["active"] is False


def test_queue_does_not_cleanup_while_another_request_is_waiting() -> None:
    queue = InferenceQueue()
    first_acquired = threading.Event()
    release_first = threading.Event()
    cleaned = []

    def first_request() -> None:
        with queue.acquire() as lease:
            lease.idle_cleanup = lambda: cleaned.append("first")
            first_acquired.set()
            assert release_first.wait(timeout=2)

    first = threading.Thread(target=first_request)
    first.start()
    assert first_acquired.wait(timeout=2)
    second = threading.Thread(target=lambda: _complete_request(queue, cleaned))
    second.start()
    while queue.status()["waiting"] != 1:
        pass
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert cleaned == ["second"]


def _complete_request(queue: InferenceQueue, cleaned: list[str]) -> None:
    with queue.acquire() as lease:
        lease.idle_cleanup = lambda: cleaned.append("second")

def test_set_unload_policy_stored() -> None:
    queue = InferenceQueue()
    assert queue.status()["unload_policy"] == "immediate"
    queue.set_unload_policy("10")
    assert queue.status()["unload_policy"] == "10"
    queue.set_unload_policy("never")
    assert queue.status()["unload_policy"] == "never"


def test_never_policy_skips_cleanup() -> None:
    queue = InferenceQueue()
    queue.set_unload_policy("never")
    cleaned = []
    with queue.acquire() as lease:
        lease.idle_cleanup = lambda: cleaned.append("model")
    assert cleaned == []


def test_immediate_policy_runs_cleanup() -> None:
    queue = InferenceQueue()
    queue.set_unload_policy("immediate")
    cleaned = []
    with queue.acquire() as lease:
        lease.idle_cleanup = lambda: cleaned.append("model")
    assert cleaned == ["model"]


def test_delayed_policy_does_not_cleanup_immediately() -> None:
    queue = InferenceQueue()
    queue.set_unload_policy("5")
    cleaned = []
    with queue.acquire() as lease:
        lease.idle_cleanup = lambda: cleaned.append("model")
    # Should not have cleaned up yet (delay is 5 minutes)
    assert cleaned == []


def test_delayed_policy_cancels_on_new_request() -> None:
    queue = InferenceQueue()
    queue.set_unload_policy("0.01")  # 0.01 minutes = 0.6 seconds
    cleaned = []
    with queue.acquire() as lease:
        lease.idle_cleanup = lambda: cleaned.append("first")
    # Start a new request before the timer fires
    import time
    time.sleep(0.1)
    with queue.acquire() as lease:
        lease.idle_cleanup = lambda: cleaned.append("second")
    # The first cleanup should have been cancelled
    # The second should not have fired yet either
    assert "first" not in cleaned


def test_delayed_policy_fires_after_timeout() -> None:
    queue = InferenceQueue()
    queue.set_unload_policy("0.01")  # 0.01 minutes = 0.6 seconds
    cleaned = []
    event = threading.Event()
    def do_cleanup():
        cleaned.append("model")
        event.set()
    with queue.acquire() as lease:
        lease.idle_cleanup = do_cleanup
    # Wait for the delayed unload to fire
    assert event.wait(timeout=3), "Delayed unload did not fire within 3 seconds"
    assert cleaned == ["model"]


def test_status_includes_unload_policy() -> None:
    queue = InferenceQueue()
    status = queue.status()
    assert "unload_policy" in status
    assert status["unload_policy"] == "immediate"
