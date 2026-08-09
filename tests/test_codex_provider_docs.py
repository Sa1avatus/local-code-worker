from pathlib import Path


def test_codex_provider_documentation_matches_gateway_contract() -> None:
    documentation = (Path(__file__).parents[1] / "docs" / "codex_model_provider.md").read_text(
        encoding="utf-8"
    )

    assert 'base_url = "http://127.0.0.1:8765/v1"' in documentation
    assert 'wire_api = "responses"' in documentation
    assert 'model = "local-code-worker/auto"' in documentation
    assert "local-code-worker/local" in documentation
    assert "Do not claim Codex Beta integration is complete" in documentation
    assert "Client disconnect" in documentation
    assert "30,000-character context" in documentation
    assert "TOML profile itself remains unverified" in documentation
    assert '$env:INSTALL_ROUTELLM = "1"' in documentation
    assert "GATEWAY_ROUTELLM_ENABLED=true" in documentation
    assert "did not finish within a 10-minute bounded check" in documentation
