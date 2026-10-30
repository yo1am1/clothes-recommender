"""Chat model construction from settings, for any LangChain-supported provider."""

from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from clothes_recommender.config import Settings


class LLMNotConfiguredError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "No chat model configured. Set LLM_MODEL in .env "
            "(plus LLM_PROVIDER, LLM_API_KEY and LLM_BASE_URL as your provider requires)."
        )


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Create the chat model described by the LLM_* settings.

    OpenAI-compatible gateways (OmniRoute, OpenRouter, Ollama's /v1 endpoint, ...) work
    with LLM_PROVIDER=openai and LLM_BASE_URL pointing at the gateway.
    """
    if not settings.llm_configured:
        raise LLMNotConfiguredError()

    # Only forward credentials/endpoint when set, so each provider's own environment
    # variables (e.g. OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT) keep working as a fallback.
    connection: dict[str, Any] = {}
    if settings.llm_api_key is not None:
        connection["api_key"] = settings.llm_api_key.get_secret_value()
    if settings.llm_base_url:
        connection["base_url"] = settings.llm_base_url

    return init_chat_model(
        settings.llm_model,
        model_provider=settings.llm_provider,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=0,
        **connection,
    )
