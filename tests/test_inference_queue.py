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
