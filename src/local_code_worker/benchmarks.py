import argparse
import json
import time
from pathlib import Path

import httpx
from pydantic import Field

from .models import StrictModel
from .virtual_models import ModelTier


class BenchmarkCase(StrictModel):
    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    expected_tier: ModelTier
    expected_files: list[str] = Field(default_factory=list)
    validation_command: str | None = None
    expected_behavior: str | None = None


class BenchmarkResult(StrictModel):
    benchmark_id: str
    selected_route: str | None
    selected_model: str | None
    route_score: float | None
    escalations: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cloud_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    success: bool
    validation_result: str


def load_cases(path: Path) -> list[BenchmarkCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("benchmark file must contain a JSON array")
    return [BenchmarkCase.model_validate(item) for item in payload]


def run_case(client: httpx.Client, case: BenchmarkCase) -> BenchmarkResult:
    started = time.perf_counter()
    response = client.post(
        "/v1/responses",
        json={
            "model": "local-code-worker/auto",
            "input": case.prompt,
            "store": False,
            "max_output_tokens": 128,
        },
    )
    latency_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    payload = response.json()
    decision_response = client.get(
        "/api/v2/router/decision",
        params={"response_id": payload["id"]},
    )
    decision_response.raise_for_status()
    decision = decision_response.json()["actual"]
    usage = payload.get("usage") or {}
    selected_route = decision.get("tier")
    token_total = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
    return BenchmarkResult(
        benchmark_id=case.id,
        selected_route=selected_route,
        selected_model=decision.get("model"),
        route_score=decision.get("routellm_score"),
        escalations=1 if decision.get("method") == "fallback" else 0,
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        cloud_tokens=token_total if selected_route == ModelTier.STRONG.value else 0,
        latency_ms=latency_ms,
        success=response.is_success,
        validation_result=(
            "expected_route" if selected_route == case.expected_tier.value else "unexpected_route"
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LCW routing benchmarks")
    parser.add_argument("--cases", type=Path, default=Path("benchmarks/cases.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    cases = load_cases(args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=args.base_url, timeout=600) as client:
        results = [run_case(client, case) for case in cases]
    with args.output.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(result.model_dump_json() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
