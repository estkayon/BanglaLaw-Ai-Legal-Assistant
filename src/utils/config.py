from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # App
    app_env: str = "development"
    app_debug: bool = True
    app_log_level: str = "DEBUG"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_rate_limit: int = 30

    # LLM
    groq_api_key: str = ""

    # Vector DB
    chroma_persist_dir: str = "chroma_db"
    chroma_collection_name: str = "banglalaw"

    # Scraper
    scraper_base_url: str = "http://bdlaws.minlaw.gov.bd"
    scraper_delay_seconds: int = 2

    # MLflow
    mlflow_tracking_uri: str = ""
    mlflow_experiment_name: str = "banglalaw-eval"

    # Data paths
    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


settings = Settings()
