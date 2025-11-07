# app/config.py
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # only what the running code needs; ignore the rest
    supabase_url: str = Field(..., alias="SUPABASE_URL")
    supabase_anon_key: str = Field(..., alias="SUPABASE_ANON_KEY")

    # keep Azure fields handy if you will use them later
    azure_endpoint: str | None = Field(None, alias="AZURE_ENDPOINT")
    azure_deployment: str | None = Field(None, alias="AZURE_DEPLOYMENT")
    azure_api_key: str | None = Field(None, alias="AZURE_API_KEY")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",              # <- critical: ignore all other keys in your .env
        populate_by_name=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
