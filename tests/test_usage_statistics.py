from pathlib import Path

from local_code_worker.models import (
    GenerationMetadata,
    JsonMode,
    ProviderName,
    ResponseAttempt,
    ResponseStatus,
)
from local_code_worker.usage_statistics import (
    record_model_call,
    record_worker_attempt,
    summarize_model_calls,
)


def test_statistics_aggregate_tokens_speed_and_code_results(tmp_path: Path) -> None:
    path = tmp_path / "statistics.json"
    metadata = GenerationMetadata(
        provider=ProviderName.OLLAMA, model="qwen:test", base_url="http://localhost",
        started_at="2026-01-01T00:00:00Z", completed_at="2026-01-01T00:00:02Z",
        duration_seconds=2, prompt_characters=0, output_characters=0, streaming=True,
        response_format_mode=JsonMode.NONE, usage={"prompt_tokens": 4, "completion_tokens": 10},
    )
    record_model_call(metadata, kind="chat", outcome="completed", path=path)
    record_worker_attempt(
        ResponseAttempt(attempt=1, status=ResponseStatus.VALID, duration_seconds=1,
                        provider=ProviderName.OLLAMA, model="qwen:test",
                        usage={"prompt_tokens": 2, "completion_tokens": 5}), path,
    )
    item = summarize_model_calls(path)["models"][0]
    assert item["requests"] == 2
    assert item["prompt_tokens"] == 6
    assert item["completion_tokens"] == 15
    assert item["tokens_per_second"] == 5.0
    assert item["code_valid"] == 1
    assert item["code_invalid"] == 0
