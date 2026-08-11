from pathlib import Path

import pytest

from local_code_worker.models import ProviderName
from local_code_worker.routing.models import RoutingMode
from local_code_worker.virtual_models import ModelTier
from local_code_worker.web_config import load_gateway_routing_settings


def write_env(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_routing_config_defaults_to_non_disruptive_legacy_mode(tmp_path) -> None:
    settings = load_gateway_routing_settings(tmp_path / "missing.env")

    assert settings.mode is RoutingMode.LEGACY
    assert settings.tiers == {}
    assert settings.routellm_enabled is False


def test_routing_config_loads_tiers_and_routellm_controls(tmp_path) -> None:
    env_path = write_env(
        tmp_path / ".env",
        "\n".join(
            [
                "GATEWAY_ROUTING_MODE=observe_only",
                "GATEWAY_LOCAL_PROVIDER=ollama",
                "GATEWAY_LOCAL_MODEL=qwen-local",
                "GATEWAY_MID_PROVIDER=openai-compatible",
                "GATEWAY_MID_MODEL=mid-model",
                "GATEWAY_MID_ENABLED=false",
                "GATEWAY_ROUTELLM_ENABLED=true",
                "GATEWAY_ROUTELLM_THRESHOLD=0.7",
                "GATEWAY_ROUTELLM_CHECKPOINT_PATH=custom-mf-checkpoint",
                "GATEWAY_POLICY_VERSION=policy-2",
                "GATEWAY_LOCAL_THRESHOLD=0.2",
                "GATEWAY_STRONG_THRESHOLD=0.8",
                "GATEWAY_CANARY_PERCENT=25",
                "GATEWAY_MAX_ESCALATIONS_PER_LEASE=3",
            ]
        ),
    )

    settings = load_gateway_routing_settings(env_path)

    assert settings.mode is RoutingMode.OBSERVE_ONLY
    assert settings.tiers[ModelTier.LOCAL].provider is ProviderName.OLLAMA
    assert settings.tiers[ModelTier.LOCAL].model == "qwen-local"
    assert settings.tiers[ModelTier.MID].enabled is False
    assert settings.routellm_enabled is True
    assert settings.routellm_threshold == 0.7
    assert settings.routellm_checkpoint_path == "custom-mf-checkpoint"
    assert settings.policy_version == "policy-2"
    assert settings.local_threshold == 0.2
    assert settings.strong_threshold == 0.8
    assert settings.canary_percent == 25
    assert settings.max_escalations_per_lease == 3


def test_router_mode_alias_is_supported(tmp_path) -> None:
    settings = load_gateway_routing_settings(write_env(tmp_path / ".env", "ROUTER_MODE=shadow\n"))

    assert settings.mode is RoutingMode.SHADOW


def test_routing_config_rejects_partial_tier(tmp_path) -> None:
    env_path = write_env(tmp_path / ".env", "GATEWAY_STRONG_MODEL=strong-model\n")

    with pytest.raises(ValueError, match="configured together"):
        load_gateway_routing_settings(env_path)


def test_routing_config_rejects_invalid_boolean(tmp_path) -> None:
    env_path = write_env(tmp_path / ".env", "GATEWAY_ROUTELLM_ENABLED=maybe\n")

    with pytest.raises(ValueError, match="Invalid boolean"):
        load_gateway_routing_settings(env_path)
