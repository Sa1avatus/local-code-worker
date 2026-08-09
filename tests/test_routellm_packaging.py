from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_default_container_keeps_routellm_optional() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")

    assert "ARG INSTALL_ROUTELLM=0" in dockerfile
    assert 'if [ "$INSTALL_ROUTELLM" = "1" ]' in dockerfile
    assert 'pip install --no-cache-dir ".[dev,routellm]"' in dockerfile
    assert "INSTALL_ROUTELLM: ${INSTALL_ROUTELLM:-0}" in compose


def test_project_pins_reviewed_routellm_release() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"routellm==0.2.0"' in project
