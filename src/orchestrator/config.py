from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings, loaded from environment / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    anthropic_api_key: str
    redis_url: str = "redis://localhost:6379/0"
    working_memory_ttl_seconds: int = 3600


def get_settings() -> Settings:
    return Settings()