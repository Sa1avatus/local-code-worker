import pytest

from local_code_worker.models import ProviderName
from local_code_worker.providers.base import ProviderMessage, ProviderRequest
from local_code_worker.routing.engine import route_request
from local_code_worker.routing.models import (
    GatewayRoutingSettings,
    RoutingMethod,
    TierConfig,
)
from local_code_worker.routing.routellm_adapter import (
    LmsysRouteLlmBackend,
    RouteLlmBackendCache,
)
from local_code_worker.virtual_models import ModelTier


class FakeBackend:
    def __init__(self, result: float | Exception) -> None:
        self.result = result
        self.calls = 0

    def score(self, request: ProviderRequest) -> float:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeMfRouter:
    def __init__(self, score: float) -> None:
        self.result = score
        self.prompt: str | None = None

    def calculate_strong_win_rate(self, prompt: str) -> float:
        self.prompt = prompt
        return self.result


def request(content: str = "Implement the requested change") -> ProviderRequest:
    return ProviderRequest(
        messages=[ProviderMessage(role="user", content=content)],
        max_output_characters=100,
    )


def settings() -> GatewayRoutingSettings:
    return GatewayRoutingSettings(
        routellm_enabled=True,
        routellm_threshold=0.6,
        tiers={
            tier: TierConfig(provider=ProviderName.OLLAMA, model=f"model-{tier.value}")
            for tier in ModelTier
        },
    )


@pytest.mark.parametrize(
    ("score", "tier"),
    [(0.59, ModelTier.LOCAL), (0.6, ModelTier.STRONG)],
)
def test_routellm_maps_ambiguous_request_to_weak_or_strong(
    score: float,
    tier: ModelTier,
) -> None:
    decision = route_request(
        request(),
        "local-code-worker/auto",
        settings(),
        routellm_backend=FakeBackend(score),
    )

    assert decision.tier is tier
    assert decision.method is RoutingMethod.ROUTELLM
    assert decision.routellm_score == score


def test_routellm_failure_uses_deterministic_fallback() -> None:
    decision = route_request(
        request(),
        "local-code-worker/auto",
        settings(),
        routellm_backend=FakeBackend(RuntimeError("private backend detail")),
    )

    assert decision.tier is ModelTier.LOCAL
    assert decision.method is RoutingMethod.DETERMINISTIC
    assert decision.routing_backend_failure is True
    assert "private backend detail" not in decision.model_dump_json()


def test_forced_tier_never_calls_routellm() -> None:
    backend = FakeBackend(1.0)

    decision = route_request(
        request("Security architecture"),
        "local-code-worker/local",
        settings(),
        routellm_backend=backend,
    )

    assert decision.tier is ModelTier.LOCAL
    assert decision.method is RoutingMethod.FORCED
    assert backend.calls == 0


def test_confident_hard_rule_never_calls_routellm() -> None:
    backend = FakeBackend(0.0)

    decision = route_request(
        request("Security review"),
        "local-code-worker/auto",
        settings(),
        routellm_backend=backend,
    )

    assert decision.tier is ModelTier.STRONG
    assert decision.method is RoutingMethod.DETERMINISTIC
    assert backend.calls == 0


def test_lmsys_adapter_scores_only_the_last_message() -> None:
    router = FakeMfRouter(0.72)
    backend = LmsysRouteLlmBackend(router)
    provider_request = ProviderRequest(
        messages=[
            ProviderMessage(role="user", content="old"),
            ProviderMessage(role="user", content="current"),
        ],
        max_output_characters=100,
    )

    assert backend.score(provider_request) == 0.72
    assert router.prompt == "current"


def test_lmsys_adapter_reports_missing_optional_dependency(monkeypatch) -> None:
    def missing_module(name: str):
        raise ImportError(name)

    monkeypatch.setattr(
        "local_code_worker.routing.routellm_adapter.import_module",
        missing_module,
    )

    with pytest.raises(RuntimeError, match="optional dependency"):
        LmsysRouteLlmBackend.load_mf()


def test_backend_cache_initializes_each_checkpoint_once(monkeypatch) -> None:
    backend = FakeBackend(0.4)
    calls: list[str | None] = []

    def load(checkpoint_path: str | None):
        calls.append(checkpoint_path)
        return backend

    monkeypatch.setattr(LmsysRouteLlmBackend, "load_mf", load)
    cache = RouteLlmBackendCache()

    assert cache.get("checkpoint") is backend
    assert cache.get("checkpoint") is backend
    assert calls == ["checkpoint"]


def test_backend_cache_converts_initialization_error_to_failure_backend(monkeypatch) -> None:
    def fail(checkpoint_path: str | None):
        raise RuntimeError("private checkpoint detail")

    monkeypatch.setattr(LmsysRouteLlmBackend, "load_mf", fail)
    backend = RouteLlmBackendCache().get(None)

    with pytest.raises(RuntimeError, match="initialization failed") as captured:
        backend.score(request())
    assert "private checkpoint detail" not in str(captured.value)
