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
        (1, "Slim Fit Blue Jeans", "jeans", "levi's", "blue", 34.79, 4.2, 285),
        (2, "Running Shoes", "shoes", "nike", "black", 59.99, 4.6, 1200),
        (3, "Red Summer Dress", "western wear", "zara", "red", 24.5, 3.9, 87),
    ]
    columns = ["product_id", "name", "category", "brand", "color", "price", "rating"]
    frame = pd.DataFrame([row[:7] for row in rows], columns=columns)
    frame["review_count"] = [row[7] for row in rows]
    frame["image_url"] = [f"https://img.example/{row[0]}.jpg" for row in rows]
    frame["product_url"] = [f"https://shop.example/p/{row[0]}" for row in rows]
    return ProductCatalog(frame)


@pytest.fixture
def search() -> FakeSearch:
    return FakeSearch()
