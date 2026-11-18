"""Load configuration from .env. Secrets are wrapped in SecretStr so they never show up in logs or reprs."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str | None = None
    companies_house_api_key: SecretStr | None = None
    inpi_username: str | None = None
    inpi_password: SecretStr | None = None
    sec_user_agent: str | None = None
    openalex_email: str | None = None
    openalex_api_key: SecretStr | None = None
    duckdb_path: Path = PROJECT_ROOT / "data" / "demo.duckdb"

    def configured(self) -> dict[str, bool]:
        """Which settings are filled in (True/False only, never the values)."""
        return {name: bool(getattr(self, name)) for name in type(self).model_fields}


def _env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


@lru_cache
def get_settings(env_file: Path | None = None) -> Settings:
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
    db_path = _env("DUCKDB_PATH")
    return Settings(
        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
        anthropic_model=_env("ANTHROPIC_MODEL"),
        companies_house_api_key=_env("COMPANIES_HOUSE_API_KEY"),
        inpi_username=_env("INPI_USERNAME"),
        inpi_password=_env("INPI_PASSWORD"),
        sec_user_agent=_env("SEC_USER_AGENT"),
        openalex_email=_env("OPENALEX_EMAIL"),
        openalex_api_key=_env("OPENALEX_API_KEY"),
        **({"duckdb_path": Path(db_path)} if db_path else {}),
    )
