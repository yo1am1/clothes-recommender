"""Application settings, read from environment variables and an optional `.env` file."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # env_ignore_empty: `LLM_MODEL=` in a copied .env.example must mean "not set", not "".
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_ignore_empty=True, extra="ignore"
    )

    # --- Data -------------------------------------------------------------
    data_dir: Path = Path("data")

    # --- Embeddings -------------------------------------------------------
    clip_model_name: str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

    # --- Vector store -----------------------------------------------------
    # With QDRANT_URL unset, Qdrant runs embedded and persists to QDRANT_PATH,
    # so the demo works without any external service.
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_path: Path = Path(".qdrant")
    qdrant_collection: str = "products"

    # --- LLM (any LangChain provider; OpenAI-compatible gateways via base URL)
    llm_provider: str = "openai"
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_temperature: float = 0.2
    llm_timeout_seconds: float = 60.0

    # --- UI ---------------------------------------------------------------
    api_url: str = "http://localhost:8000"

    @property
    def products_path(self) -> Path:
        return self.data_dir / "products.parquet"

    @property
    def users_path(self) -> Path:
        return self.data_dir / "users.parquet"

    @property
    def purchases_path(self) -> Path:
        return self.data_dir / "purchases.parquet"

    @property
    def shopper_purchases_path(self) -> Path:
        """Every shopper's purchases of catalog products, under anonymous ids."""
        return self.data_dir / "shopper_purchases.parquet"

    @property
    def co_purchases_path(self) -> Path:
        return self.data_dir / "co_purchases.parquet"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def query_photos_dir(self) -> Path:
        """Photos of garments outside the catalog, for trying image search."""
        return self.data_dir / "query_photos"

    @property
    def ranker_path(self) -> Path:
        return self.data_dir / "ranker.json"

    @property
    def evaluation_path(self) -> Path:
        return self.data_dir / "evaluation.json"

    @property
    def llm_configured(self) -> bool:
        return self.llm_model is not None


@lru_cache
def get_settings() -> Settings:
    return Settings()
