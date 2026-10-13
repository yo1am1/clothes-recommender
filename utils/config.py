"""This module loads environment variables from a .env file and provides a Config class to access them."""

import os
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv

_env = load_dotenv(
    dotenv_path=Path(__file__).parent.parent / ".env",
)


@dataclass
class Config:
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") if _env else None
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION") if _env else None
    AZURE_OPENAI_BASE_URL = os.getenv("AZURE_OPENAI_BASE_URL") if _env else None
    AZURE_OPENAI_SUBSCRIPTION_KEY = (
        os.getenv("AZURE_OPENAI_SUBSCRIPTION_KEY") if _env else None
    )
    # AZURE_OPENAI_EMBEDDINGS_MODEL = (
    #     os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL") if _env else None
    # )
    AZURE_OPENAI_CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_MODEL") if _env else None

    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY") if _env else None
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY") if _env else None
    AWS_REGION = os.getenv("AWS_REGION") if _env else None
    AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME") if _env else None
    AWS_FOLDER_NAME = os.getenv("AWS_FOLDER_NAME") if _env else None
