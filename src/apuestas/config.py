"""Configuración global tipada con pydantic-settings + SecretStr."""

from enum import StrEnum
from functools import lru_cache
from typing import Annotated

from pydantic import (
    Field,
    PostgresDsn,
    RedisDsn,
    SecretStr,
    computed_field,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    LOCAL = "local"
    CI = "ci"
    PROD = "prod"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    postgres_user: str = Field(default="apuestas")
    postgres_password: SecretStr
    postgres_db: str = Field(default="apuestas")
    postgres_host: str = Field(default="postgres")
    postgres_port: int = Field(default=5432)
    postgres_host_port: int = Field(default=5433)

    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)
    pool_pre_ping: bool = Field(default=True)
    pool_recycle_seconds: int = Field(default=1800)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def url(self) -> PostgresDsn:
        return PostgresDsn(
            f"postgresql+asyncpg://{self.postgres_user}:"
            f"{self.postgres_password.get_secret_value()}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sync_url(self) -> PostgresDsn:
        """URL síncrona para Alembic (usa psycopg)."""
        return PostgresDsn(
            f"postgresql+psycopg://{self.postgres_user}:"
            f"{self.postgres_password.get_secret_value()}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


class ValkeySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    valkey_host: str = Field(default="valkey")
    valkey_port: int = Field(default=6379)
    valkey_password: SecretStr
    taskiq_broker_url: RedisDsn | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def url(self) -> RedisDsn:
        return RedisDsn(
            f"redis://:{self.valkey_password.get_secret_value()}@"
            f"{self.valkey_host}:{self.valkey_port}/0"
        )


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    # Backend selector: "llama_local" (Qwen GGUF GPU) | "deepseek" (API remota).
    # Si tienes cuenta DeepSeek, pon "deepseek" y rellena DEEPSEEK_API_KEY.
    llm_backend: str = Field(default="llama_local")

    # llama.cpp local (modo GPU)
    llama_server_url: str = Field(default="http://llm:8080")
    llama_model: str = Field(default="qwen2.5-7b-instruct-q4_k_m")
    llama_ctx_size: int = Field(default=8192)
    llama_temperature: float = Field(default=0.2)
    llama_max_tokens: int = Field(default=1024)

    # DeepSeek API (OpenAI-compatible)
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = Field(default="https://api.deepseek.com")
    # "deepseek-chat" → V3.2 (general); "deepseek-reasoner" → R1 para razonamiento.
    deepseek_model: str = Field(default="deepseek-chat")
    deepseek_temperature: float = Field(default=0.2)
    deepseek_max_tokens: int = Field(default=1024)

    tei_url: str = Field(default="http://embed:80")
    embed_model: str = Field(default="BAAI/bge-m3")
    embed_dim: int = Field(default=1024)


class MCPSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    apuestas_use_mcp: bool = Field(default=True)
    cuba_memorys_stdio_cmd: str = Field(default="")
    cuba_search_stdio_cmd: str = Field(default="")


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    api_football_key: SecretStr | None = None
    the_odds_api_key: SecretStr | None = None
    openweathermap_key: SecretStr | None = None
    visual_crossing_key: SecretStr | None = None
    # Fuentes gratis (opcionales, cero cost)
    football_data_org_key: SecretStr | None = None
    thesportsdb_key: str = Field(default="3")

    reddit_client_id: SecretStr | None = None
    reddit_client_secret: SecretStr | None = None
    reddit_user_agent: str = Field(default="apuestas-bot/0.1")

    telegram_bot_token: SecretStr | None = None
    telegram_chat_id: str | None = None


class BettingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    default_bankroll_units: float = Field(default=200.0)
    kelly_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = Field(default=0.25)
    kelly_max_stake_pct: Annotated[float, Field(gt=0.0, le=0.5)] = Field(default=0.05)
    ev_threshold: Annotated[float, Field(gt=0.0, le=0.5)] = Field(default=0.03)
    min_odds: Annotated[float, Field(gt=1.0)] = Field(default=1.50)
    max_odds: Annotated[float, Field(gt=1.0)] = Field(default=4.00)

    @field_validator("max_odds")
    @classmethod
    def max_odds_gt_min(cls, v: float, info: object) -> float:
        values = getattr(info, "data", {}) if info else {}
        min_odds = values.get("min_odds", 1.5)
        if v <= min_odds:
            msg = f"max_odds ({v}) must be > min_odds ({min_odds})"
            raise ValueError(msg)
        return v


class ObservabilitySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    otel_exporter_otlp_endpoint: str = Field(default="http://signoz-otel:4317")
    otel_service_name: str = Field(default="apuestas")
    sentry_dsn: SecretStr | None = None


class Settings(BaseSettings):
    """Raíz de configuración. Carga desde .env y variables de entorno."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    apuestas_env: Environment = Field(default=Environment.LOCAL)
    apuestas_log_level: LogLevel = Field(default=LogLevel.INFO)
    apuestas_tz: str = Field(default="America/Mexico_City")

    # Flags
    apuestas_paper_trading: bool = Field(default=True)
    apuestas_auto_postmortem: bool = Field(default=True)
    apuestas_enable_llm_enrichment: bool = Field(default=True)
    apuestas_enable_rag: bool = Field(default=True)
    apuestas_enable_mcp: bool = Field(default=True)

    # Sub-settings se cargan lazy para no romper si faltan secrets durante tests
    @computed_field  # type: ignore[prop-decorator]
    @property
    def database(self) -> DatabaseSettings:
        return DatabaseSettings()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def valkey(self) -> ValkeySettings:
        return ValkeySettings()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def llm(self) -> LLMSettings:
        return LLMSettings()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mcp(self) -> MCPSettings:
        return MCPSettings()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def apis(self) -> APISettings:
        return APISettings()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def betting(self) -> BettingSettings:
        return BettingSettings()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def obs(self) -> ObservabilitySettings:
        return ObservabilitySettings()

    @property
    def is_prod(self) -> bool:
        return self.apuestas_env == Environment.PROD


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton tipado. Cachea para evitar re-parsing."""
    return Settings()
