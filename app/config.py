# app/config.py
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Supabase
    supabase_url: str = Field(..., alias="SUPABASE_URL")
    supabase_anon_key: str = Field(..., alias="SUPABASE_ANON_KEY")

    # Azure (optional)
    azure_endpoint: str | None = Field(None, alias="AZURE_ENDPOINT")
    azure_deployment: str | None = Field(None, alias="AZURE_DEPLOYMENT")
    azure_api_key: str | None = Field(None, alias="AZURE_API_KEY")

    # JWT Authentication - ADD THESE
    jwt_secret: str = Field(
        "change-this-to-a-secure-random-string-min-32-chars",
        alias="JWT_SECRET"
    )
    jwt_algorithm: str = Field("HS256", alias="JWT_ALGORITHM")
    jwt_expiry_hours: int = Field(24, alias="JWT_EXPIRY_HOURS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Convenience accessor for settings
settings = get_settings()


# -----------------------------
# Static boilerplate used by /generate_proposal
# -----------------------------
TERMS_AND_CONDITIONS = [
    "Engagement scope is limited to the services and deliverables defined in this Statement of Work.",
    "All activities are performed remotely unless explicitly stated otherwise in the scope.",
    "Customer will provide timely access to systems, environments, data, and stakeholders as needed.",
    "Any change in scope, assumptions, or prerequisites requires a mutually approved change request.",
    "Dependencies on third-party products, licenses, or infrastructure are the customer's responsibility.",
    "Project schedules are subject to customer resource availability and environment readiness.",
    "Travel, lodging, and other out-of-pocket expenses are billable at actuals, if applicable.",
    "All fees exclude applicable taxes and duties; taxes will be charged as per prevailing law.",
    "Deliverables are provided on a best-effort basis aligned to stated acceptance criteria.",
    "Post-implementation support outside the defined scope will be treated as a new engagement.",
]