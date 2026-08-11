import json

import httpx

from local_code_worker.benchmarks import BenchmarkCase, load_cases, run_case


def test_load_cases_validates_representative_dataset(tmp_path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "simple",
                    "category": "simple_edit",
                    "prompt": "Small edit",
                    "expected_tier": "local",
                }
            ]
        ),
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert cases == [
        BenchmarkCase(
            id="simple",
            category="simple_edit",
            prompt="Small edit",
            expected_tier="local",
        )
    ]


def test_run_case_records_route_and_tokens_without_prompt(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/responses":
            return httpx.Response(
                200,
                json={
                    "id": "resp-1",
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                },
            )
        return httpx.Response(
            200,
            json={
                "actual": {
                    "tier": "local",
                    "model": "local-model",
                    "method": "deterministic",
                }
            },
        )

    with httpx.Client(
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = run_case(
            client,
            BenchmarkCase(
                id="simple",
                category="simple_edit",
                prompt="private benchmark prompt",
                expected_tier="local",
            ),
        )

    assert result.selected_route == "local"
    assert result.input_tokens == 10
    assert "private benchmark prompt" not in result.model_dump_json()
