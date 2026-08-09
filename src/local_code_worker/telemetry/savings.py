from enum import StrEnum

from pydantic import Field

from ..models import StrictModel


class BaselineMethod(StrEnum):
    EXPLICIT_CLOUD_TOKEN_BUDGET = "explicit_cloud_token_budget"


class TokenSavings(StrictModel):
    baseline_cloud_tokens: int = Field(default=0, ge=0)
    actual_cloud_tokens: int = Field(default=0, ge=0)
    baseline_method: BaselineMethod = BaselineMethod.EXPLICIT_CLOUD_TOKEN_BUDGET
    estimated: bool = True

    @property
    def cloud_tokens_saved(self) -> int:
        return max(self.baseline_cloud_tokens - self.actual_cloud_tokens, 0)

    @property
    def cloud_tokens_saved_percent(self) -> float:
        if self.baseline_cloud_tokens == 0:
            return 0.0
        return round(self.cloud_tokens_saved / self.baseline_cloud_tokens * 100, 2)
