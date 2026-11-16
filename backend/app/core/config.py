"""Backend settings loaded from environment variables (prefix ``KIVU_``) or ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MIN_PRODUCTION_API_KEY_LENGTH = 24


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KIVU_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Kivu Smart Restoration API"
    environment: Literal["development", "test", "production"] = "development"

    # PostgreSQL/PostGIS is the target database. SQLite is accepted only for
    # offline tests and quick demos; spatial SQL is then evaluated in Python.
    database_url: str = "postgresql+psycopg://kivu:kivu@localhost:5432/kivu_smart"
    database_echo: bool = False
    auto_create_tables: bool = False

    api_key: SecretStr | None = None
    cors_origins: list[str] = ["http://localhost:8501", "http://127.0.0.1:8501"]
    allowed_hosts: list[str] = ["*"]
    rate_limit_per_minute: int = Field(120, ge=1)
    max_upload_mb: float = Field(50, gt=0, le=2048)
    docs_enabled: bool = True

    data_dir: Path = PROJECT_ROOT / "data"
    config_dir: Path = PROJECT_ROOT / "config"

    earthdata_token: SecretStr | None = None
    earthdata_username: str | None = None
    earthdata_password: SecretStr | None = None
    nasa_timeout_s: float = Field(60, gt=0, le=600)
    nasa_max_retries: int = Field(3, ge=0, le=10)
    nasa_cache_ttl_hours: float = Field(24, ge=0)
    nasa_max_download_mb: float = Field(2048, gt=0)

    @field_validator("data_dir", "config_dir", mode="after")
    @classmethod
    def _resolve_relative(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @model_validator(mode="after")
    def _check_production(self) -> "Settings":
        if self.environment == "production":
            key = self.api_key.get_secret_value() if self.api_key else ""
            if len(key) < MIN_PRODUCTION_API_KEY_LENGTH:
                raise ValueError(
                    "KIVU_API_KEY must be set (>= 24 characters) in production."
                )
            if "*" in self.cors_origins:
                raise ValueError("Wildcard CORS origins are refused in production.")
            if self.database_url.startswith("sqlite"):
                raise ValueError("Production requires PostgreSQL/PostGIS, not SQLite.")
        return self

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def nasa_raw_dir(self) -> Path:
        return self.data_dir / "raw" / "nasa"

    @property
    def nasa_cache_dir(self) -> Path:
        return self.data_dir / "cache" / "nasa"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    def ensure_directories(self) -> None:
        for directory in (
            self.upload_dir,
            self.nasa_raw_dir,
            self.nasa_cache_dir,
            self.processed_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
