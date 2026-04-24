from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # API keys
    ANTHROPIC_API_KEY: SecretStr
    GOOGLE_API_KEY: SecretStr
    TAVILY_API_KEY: SecretStr

    # Logging
    LOG_LEVEL: str = "INFO"

    # Research parameters
    MAX_SUBQUESTIONS: int = 5
    MAX_SOURCES_PER_QUERY: int = 20
    DEFAULT_BUDGET_USD: float = 0.50

    # Models
    PLANNER_MODEL: str = "claude-3-5-sonnet-latest"
    SYNTHESIZER_MODEL: str = "claude-3-5-sonnet-latest"
    CRITIC_MODEL: str = "claude-3-5-sonnet-latest"


settings = Settings()
