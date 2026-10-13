import pytest
from pydantic import SecretStr

from clothes_recommender.chat.llm import LLMNotConfiguredError, build_chat_model
from clothes_recommender.config import Settings


def make_settings(**llm_fields: object) -> Settings:
    defaults: dict[str, object] = {
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "llm_api_key": SecretStr("sk-test"),
        "llm_base_url": None,
        "llm_temperature": 0.1,
        "llm_timeout_seconds": 12.0,
    }
    return Settings(_env_file=None, **(defaults | llm_fields))


def test_missing_model_raises_actionable_error() -> None:
    with pytest.raises(LLMNotConfiguredError, match="LLM_MODEL"):
        build_chat_model(make_settings(llm_model=None))


def test_openai_compatible_gateway_gets_key_base_url_and_options() -> None:
    from langchain_openai import ChatOpenAI

    model = build_chat_model(make_settings(llm_base_url="http://localhost:20128/v1"))

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-4o-mini"
    assert model.openai_api_key is not None
    assert model.openai_api_key.get_secret_value() == "sk-test"
    assert model.openai_api_base == "http://localhost:20128/v1"
    assert model.temperature == 0.1
    assert model.request_timeout == 12.0


def test_base_url_is_not_passed_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    model = build_chat_model(make_settings())

    assert model.openai_api_base is None


def test_native_provider_is_supported() -> None:
    langchain_anthropic = pytest.importorskip("langchain_anthropic")

    model = build_chat_model(make_settings(llm_provider="anthropic", llm_model="claude-sonnet-4-5"))

    assert isinstance(model, langchain_anthropic.ChatAnthropic)
    assert model.anthropic_api_key.get_secret_value() == "sk-test"
