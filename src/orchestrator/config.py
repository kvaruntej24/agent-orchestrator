from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings, loaded from environment / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    anthropic_api_key: str
    redis_url: str = "redis://localhost:6379/0"
    working_memory_ttl_seconds: int = 3600
    postgres_dsn: str = "postgresql+psycopg://postgres:postgres@localhost:5432/agent_orchestrator"
    chroma_persist_dir: str = "chroma_data"
    hitl_confidence_threshold: float = 0.75
    hitl_consecutive_failure_cap: int = 2
    max_task_cost_usd: float = 0.50
    max_task_tokens: int = 50000


def get_settings() -> Settings:
    return Settings()