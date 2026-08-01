from typing import Literal

from pydantic import BaseModel, ConfigDict, HttpUrl, SecretStr, field_validator, model_validator

from .models import ProviderName


def validate_model_name(value: str) -> str:
    model = value.strip()
    if not 1 <= len(model) <= 200:
        raise ValueError("Model name must contain between 1 and 200 characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in model):
        raise ValueError("Model name contains an ASCII control character")
    return model


class ProviderSettingsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    provider: ProviderName
    base_url: HttpUrl
    model: str
    api_key_action: Literal["keep", "replace", "clear"] = "keep"
    api_key: SecretStr | None = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        return validate_model_name(value)

    @model_validator(mode="after")
    def validate_provider_and_key(self) -> "ProviderSettingsInput":
        if self.provider is ProviderName.OLLAMA and self.base_url.host not in {
            "localhost",
            "127.0.0.1",
            "::1",
            "host.docker.internal",
        }:
            raise ValueError("Ollama base URL must use a loopback host")

        if self.api_key_action == "replace":
            if self.api_key is None or not self.api_key.get_secret_value():
                raise ValueError("A non-empty API key is required for replace")
            secret = self.api_key.get_secret_value()
            if any(ord(character) < 32 or ord(character) == 127 for character in secret):
                raise ValueError("API key contains an ASCII control character")
        elif self.api_key is not None:
            raise ValueError("API key must be omitted unless action is replace")
        return self
