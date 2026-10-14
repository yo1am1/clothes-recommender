from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from clothes_recommender.catalog import ProductCatalog
from clothes_recommender.chat.intent import SearchIntent
from clothes_recommender.domain import ProductFilter, ScoredProduct


class FakeToolCallingChatModel(GenericFakeChatModel):
    """Fake model that accepts tools, so the real `with_structured_output` code path runs."""

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeToolCallingChatModel":
        return self


type ChatModelFactory = Callable[..., FakeToolCallingChatModel]


def _as_message(reply: dict[str, Any] | str) -> AIMessage:
    if isinstance(reply, str):
        return AIMessage(content=reply)
    # A dict becomes a SearchIntent tool call, exactly as a real model would emit it.
    return AIMessage(
        content="", tool_calls=[{"name": SearchIntent.__name__, "args": reply, "id": "call_1"}]
    )


@pytest.fixture
def chat_model_replying() -> ChatModelFactory:
    """Factory for a fake model that answers successive calls with the given replies.

    Each reply is a dict (SearchIntent tool-call arguments) or a str (plain text).
    """

    def build(*replies: dict[str, Any] | str) -> FakeToolCallingChatModel:
        return FakeToolCallingChatModel(messages=iter([_as_message(r) for r in replies]))

    return build


class FakeSearch:
    """Duck-typed stand-in for ProductSearch that records its calls."""

    def __init__(self) -> None:
        self.results: list[ScoredProduct] = []
        self.calls: list[tuple[str, int, ProductFilter | None]] = []

    def by_text(
        self, query: str, top_k: int = 10, product_filter: ProductFilter | None = None
    ) -> list[ScoredProduct]:
        self.calls.append((query, top_k, product_filter))
        return self.results[:top_k]


@pytest.fixture
def catalog() -> ProductCatalog:
    rows = [
        (1, "Slim Jeans", "Trousers", "Menswear", "Blue", "5-pocket jeans in washed denim.", 29.99),
        (2, "Run Shoe", "Sneakers", "Sport", "Black", "Lightweight running shoes.", 59.99),
        (3, "Summer Dress", "Dress", "Ladieswear", "Red", "Short dress in airy cotton.", 24.99),
    ]
    columns = ["product_id", "name", "product_type", "section", "colour", "description", "price"]
    return ProductCatalog(pd.DataFrame(rows, columns=columns))


@pytest.fixture
def search() -> FakeSearch:
    return FakeSearch()
