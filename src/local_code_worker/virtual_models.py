from enum import StrEnum

from pydantic import Field

from .models import StrictModel


class ModelTier(StrEnum):
    LOCAL = "local"
    MID = "mid"
    STRONG = "strong"


class VirtualModel(StrictModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    forced_tier: ModelTier | None = None


class VirtualModelRegistry:
    def __init__(self, models: list[VirtualModel]) -> None:
        self._models = {model.id: model for model in models}
        if len(self._models) != len(models):
            raise ValueError("virtual model IDs must be unique")

    def list_models(self) -> list[VirtualModel]:
        return list(self._models.values())

    def resolve(self, model_id: str) -> VirtualModel:
        model = self._models.get(model_id)
        if model is None:
            raise ValueError(f"Unknown virtual model: {model_id}")
        return model


VIRTUAL_MODEL_REGISTRY = VirtualModelRegistry(
    [
        VirtualModel(
            id="local-code-worker/auto",
            description="Automatic Local Code Worker routing",
        ),
        VirtualModel(
            id="local-code-worker/local",
            description="Force the LOCAL model tier",
            forced_tier=ModelTier.LOCAL,
        ),
        VirtualModel(
            id="local-code-worker/mid",
            description="Force the MID model tier",
            forced_tier=ModelTier.MID,
        ),
        VirtualModel(
            id="local-code-worker/strong",
            description="Force the STRONG model tier",
            forced_tier=ModelTier.STRONG,
        ),
    ]
)
