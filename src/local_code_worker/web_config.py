from os import environ
from pathlib import Path

from dotenv import dotenv_values, set_key, unset_key
from pydantic import SecretStr

from .config import WorkerSettings
from .models import ProviderName
from .routing.models import GatewayRoutingSettings, RoutingMode, TierConfig
from .virtual_models import ModelTier
from .web_models import GatewaySettingsInput, ProviderSettingsInput

WEB_API_KEY_ENV = "LOCAL_CODE_WORKER_UI_API_KEY"
TIER_API_KEY_ENV = {tier: f"LOCAL_CODE_WORKER_{tier.value.upper()}_API_KEY" for tier in ModelTier}


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean configuration value: {value}")


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or not str(value).strip():
        return None
    parsed = int(str(value).strip())
    if parsed < 1 or parsed > 64:
        raise ValueError(f"num_parallel must be between 1 and 64, got {parsed}")
    return parsed


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None or not str(value).strip():
        return None
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean configuration value: {value}")


def load_gateway_routing_settings(
    env_path: Path = Path(".env"),
) -> GatewayRoutingSettings:
    file_values = dotenv_values(env_path)

    def value(name: str) -> str | None:
        raw = environ.get(name, file_values.get(name))
        return str(raw).strip() if raw is not None and str(raw).strip() else None

    tiers: dict[ModelTier, TierConfig] = {}
    for tier in ModelTier:
        prefix = f"GATEWAY_{tier.value.upper()}"
        provider = value(f"{prefix}_PROVIDER")
        model = value(f"{prefix}_MODEL")
        if (provider is None) != (model is None):
            raise ValueError(f"{prefix}_PROVIDER and {prefix}_MODEL must be configured together")
        if provider is not None and model is not None:
            enabled = value(f"{prefix}_ENABLED")
            tiers[tier] = TierConfig(
                provider=ProviderName(provider),
                model=model,
                enabled=_parse_bool(enabled, default=True),
                base_url=value(f"{prefix}_BASE_URL"),
                context_length=int(value(f"{prefix}_CONTEXT_LENGTH") or "16384"),
                num_parallel=_parse_optional_int(value(f"{prefix}_NUM_PARALLEL")) or 1,
                think=_parse_optional_bool(value(f"{prefix}_THINK")),
                api_key_env=value(f"{prefix}_API_KEY_ENV"),
            )

    return GatewayRoutingSettings(
        mode=RoutingMode(
            value("GATEWAY_ROUTING_MODE") or value("ROUTER_MODE") or RoutingMode.LEGACY.value
        ),
        tiers=tiers,
        policy_version=value("GATEWAY_POLICY_VERSION") or "1",
        routellm_enabled=_parse_bool(value("GATEWAY_ROUTELLM_ENABLED"), default=False),
        routellm_threshold=float(value("GATEWAY_ROUTELLM_THRESHOLD") or "0.5"),
        routellm_ambiguity_confidence=float(
            value("GATEWAY_ROUTELLM_AMBIGUITY_CONFIDENCE") or "0.65"
        ),
        routellm_checkpoint_path=value("GATEWAY_ROUTELLM_CHECKPOINT_PATH"),
        local_threshold=float(value("GATEWAY_LOCAL_THRESHOLD") or "0.3"),
        strong_threshold=float(value("GATEWAY_STRONG_THRESHOLD") or "0.7"),
        canary_percent=int(value("GATEWAY_CANARY_PERCENT") or "10"),
        max_escalations_per_lease=int(value("GATEWAY_MAX_ESCALATIONS_PER_LEASE") or "2"),
    )


def public_gateway_settings(env_path: Path = Path(".env")) -> dict[str, object]:
    settings = load_gateway_routing_settings(env_path)
    legacy = load_web_worker_settings(env_path)
    file_values = dotenv_values(env_path)
    tiers: dict[str, object] = {}
    for tier in ModelTier:
        config = settings.tiers.get(tier) or TierConfig(
            provider=legacy.llm_provider,
            model=legacy.llm_model,
            base_url=legacy.llm_base_url,
            context_length=legacy.llm_num_ctx,
            api_key_env=TIER_API_KEY_ENV[tier],
        )
        key_name = config.api_key_env
        configured = bool(key_name and (environ.get(key_name) or file_values.get(key_name)))
        tiers[tier.value] = {
            "enabled": config.enabled,
            "provider": config.provider.value,
            "base_url": str(config.base_url) if config.base_url else None,
            "model": config.model,
            "context_length": config.context_length,
            "num_parallel": config.num_parallel,
            "think": config.think,
            "api_key_configured": configured,
            "api_key_env": key_name,
        }
    return {
        "mode": settings.mode.value,
        "tiers": tiers,
        "routellm_enabled": settings.routellm_enabled,
        "routellm_threshold": settings.routellm_threshold,
        "local_threshold": settings.local_threshold,
        "strong_threshold": settings.strong_threshold,
        "canary_percent": settings.canary_percent,
        "max_escalations_per_lease": settings.max_escalations_per_lease,
    }


def save_gateway_settings(
    value: GatewaySettingsInput,
    env_path: Path = Path(".env"),
) -> dict[str, object]:
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.touch(exist_ok=True)
    path = str(env_path)
    set_key(path, "GATEWAY_ROUTING_MODE", value.mode.value)
    set_key(path, "GATEWAY_ROUTELLM_ENABLED", str(value.routellm_enabled).lower())
    set_key(path, "GATEWAY_ROUTELLM_THRESHOLD", str(value.routellm_threshold))
    set_key(path, "GATEWAY_LOCAL_THRESHOLD", str(value.local_threshold))
    set_key(path, "GATEWAY_STRONG_THRESHOLD", str(value.strong_threshold))
    set_key(path, "GATEWAY_CANARY_PERCENT", str(value.canary_percent))
    set_key(
        path,
        "GATEWAY_MAX_ESCALATIONS_PER_LEASE",
        str(value.max_escalations_per_lease),
    )
    for tier, config in value.tiers.items():
        prefix = f"GATEWAY_{tier.value.upper()}"
        key_name = TIER_API_KEY_ENV[tier]
        set_key(path, f"{prefix}_ENABLED", str(config.enabled).lower())
        set_key(path, f"{prefix}_PROVIDER", config.provider.value)
        set_key(path, f"{prefix}_BASE_URL", str(config.base_url).rstrip("/"))
        set_key(path, f"{prefix}_MODEL", config.model)
        set_key(path, f"{prefix}_CONTEXT_LENGTH", str(config.context_length))
        set_key(path, f"{prefix}_NUM_PARALLEL", str(config.num_parallel))
        if config.think is None:
            unset_key(path, f"{prefix}_THINK")
        else:
            set_key(path, f"{prefix}_THINK", str(config.think).lower())
        if config.api_key_action == "replace":
            assert config.api_key is not None
            set_key(path, f"{prefix}_API_KEY_ENV", key_name)
            set_key(path, key_name, config.api_key.get_secret_value())
        elif config.api_key_action == "clear":
            unset_key(path, f"{prefix}_API_KEY_ENV")
            unset_key(path, key_name)
    return public_gateway_settings(env_path)


def public_settings(settings: WorkerSettings) -> dict[str, object]:
    if settings.llm_provider is ProviderName.OLLAMA:
        api_key, api_key_env = None, None
    else:
        api_key, api_key_env = settings.resolve_api_key()
    return {
        "provider": settings.llm_provider.value,
        "base_url": str(settings.llm_base_url),
        "model": settings.llm_model,
        "context_length": settings.llm_num_ctx,
        "api_key_configured": bool(api_key),
        "api_key_env": api_key_env,
    }


def load_public_settings(env_path: Path = Path(".env")) -> dict[str, object]:
    return public_settings(load_web_worker_settings(env_path))


def load_web_worker_settings(env_path: Path = Path(".env")) -> WorkerSettings:
    settings = WorkerSettings(_env_file=env_path)
    if settings.llm_api_key is not None:
        return settings
    variable = settings.llm_api_key_env
    if variable is None:
        return settings
    file_values = dotenv_values(env_path)
    secret = environ.get(variable) or file_values.get(variable)
    if not secret:
        return settings
    return WorkerSettings(_env_file=env_path, llm_api_key=SecretStr(str(secret)))


def initialize_container_settings(env_path: Path) -> None:
    if env_path.exists() or environ.get("LOCAL_CODE_WORKER_CONTAINER") != "1":
        return
    save_provider_settings(
        ProviderSettingsInput.model_validate(
            {
                "provider": "ollama",
                "base_url": "http://host.docker.internal:11434",
                "model": "qwen2.5-coder:3b",
                "context_length": 16_384,
            }
        ),
        env_path,
    )


def save_provider_settings(
    value: ProviderSettingsInput,
    env_path: Path = Path(".env"),
) -> dict[str, object]:
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.touch(exist_ok=True)
    path = str(env_path)
    set_key(path, "LLM_PROVIDER", value.provider.value)
    set_key(path, "LLM_BASE_URL", str(value.base_url).rstrip("/"))
    set_key(path, "LLM_MODEL", value.model)
    set_key(path, "LLM_NUM_CTX", str(value.context_length))

    if value.api_key_action == "clear":
        unset_key(path, "LLM_API_KEY")
        unset_key(path, "LLM_API_KEY_ENV")
        unset_key(path, WEB_API_KEY_ENV)
    elif value.api_key_action == "replace":
        assert value.api_key is not None
        unset_key(path, "LLM_API_KEY")
        set_key(path, "LLM_API_KEY_ENV", WEB_API_KEY_ENV)
        set_key(path, WEB_API_KEY_ENV, value.api_key.get_secret_value())

    api_key_configured = False
    api_key_env: str | None = None
    if value.provider is ProviderName.OPENAI_COMPATIBLE:
        if value.api_key_action == "replace":
            api_key_configured = True
            api_key_env = WEB_API_KEY_ENV
        elif value.api_key_action == "keep":
            current = load_web_worker_settings(env_path)
            key, api_key_env = current.resolve_api_key()
            api_key_configured = bool(key)
    return {
        "provider": value.provider.value,
        "base_url": str(value.base_url).rstrip("/"),
        "model": value.model,
        "context_length": value.context_length,
        "api_key_configured": api_key_configured,
        "api_key_env": api_key_env,
    }
