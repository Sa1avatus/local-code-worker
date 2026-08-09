import pytest

from local_code_worker.virtual_models import VIRTUAL_MODEL_REGISTRY, ModelTier


def test_virtual_model_catalog_is_stable_and_ordered() -> None:
    models = VIRTUAL_MODEL_REGISTRY.list_models()

    assert [model.id for model in models] == [
        "local-code-worker/auto",
        "local-code-worker/local",
        "local-code-worker/mid",
        "local-code-worker/strong",
    ]
    assert models[0].forced_tier is None


@pytest.mark.parametrize(
    ("model_id", "tier"),
    [
        ("local-code-worker/local", ModelTier.LOCAL),
        ("local-code-worker/mid", ModelTier.MID),
        ("local-code-worker/strong", ModelTier.STRONG),
    ],
)
def test_forced_virtual_model_resolves_tier(model_id: str, tier: ModelTier) -> None:
    assert VIRTUAL_MODEL_REGISTRY.resolve(model_id).forced_tier is tier


def test_unknown_virtual_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown virtual model"):
        VIRTUAL_MODEL_REGISTRY.resolve("local-code-worker/unknown")
