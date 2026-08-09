from importlib import import_module
from threading import Lock
from typing import Protocol, cast

from ..providers.base import ProviderRequest


class RouteLlmBackend(Protocol):
    def score(self, request: ProviderRequest) -> float: ...


class StrongWinRateRouter(Protocol):
    def calculate_strong_win_rate(self, prompt: str) -> float: ...


class LmsysRouteLlmBackend:
    def __init__(self, router: StrongWinRateRouter) -> None:
        self._router = router

    @classmethod
    def load_mf(cls, checkpoint_path: str | None = None) -> "LmsysRouteLlmBackend":
        try:
            controller_module = import_module("routellm.controller")
        except ImportError as error:
            raise RuntimeError(
                "RouteLLM is not installed; install the 'routellm' optional dependency"
            ) from error
        controller_type = getattr(controller_module, "Controller", None)
        if controller_type is None:
            raise RuntimeError("Installed RouteLLM package has no Controller")
        config = {"mf": {"checkpoint_path": checkpoint_path}} if checkpoint_path else None
        controller = controller_type(
            routers=["mf"],
            strong_model="strong",
            weak_model="weak",
            config=config,
        )
        routers = getattr(controller, "routers", None)
        if not isinstance(routers, dict) or "mf" not in routers:
            raise RuntimeError("RouteLLM Controller did not initialize the MF router")
        return cls(cast(StrongWinRateRouter, routers["mf"]))

    def score(self, request: ProviderRequest) -> float:
        if not request.messages:
            raise ValueError("RouteLLM requires at least one message")
        return self._router.calculate_strong_win_rate(request.messages[-1].content)


class _FailedRouteLlmBackend:
    def score(self, request: ProviderRequest) -> float:
        raise RuntimeError("RouteLLM backend initialization failed")


class RouteLlmBackendCache:
    def __init__(self) -> None:
        self._lock = Lock()
        self._backends: dict[str | None, RouteLlmBackend] = {}

    def get(self, checkpoint_path: str | None) -> RouteLlmBackend:
        with self._lock:
            cached = self._backends.get(checkpoint_path)
            if cached is not None:
                return cached
            try:
                backend: RouteLlmBackend = LmsysRouteLlmBackend.load_mf(checkpoint_path)
            except Exception:
                backend = _FailedRouteLlmBackend()
            self._backends[checkpoint_path] = backend
            return backend


ROUTELLM_BACKENDS = RouteLlmBackendCache()


def validated_score(backend: RouteLlmBackend, request: ProviderRequest) -> float:
    score = backend.score(request)
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("RouteLLM score must be numeric")
    normalized = float(score)
    if not 0 <= normalized <= 1:
        raise ValueError("RouteLLM score must be between zero and one")
    return normalized
