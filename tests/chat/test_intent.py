import logging

import pytest

from clothes_recommender.chat.intent import IntentExtractor, SearchIntent
from clothes_recommender.domain import ProductFilter

CATEGORIES = ["jeans", "shoes", "western wear"]


def test_extracts_structured_intent_from_tool_call(chat_model_replying) -> None:
    model = chat_model_replying(
        {"query": "black running shoes", "categories": ["shoes"], "max_price": 80}
    )

    intent = IntentExtractor(model, CATEGORIES).extract("black running shoes under $80", [])

    assert intent == SearchIntent(
        query="black running shoes", categories=["shoes"], max_price=80, wants_products=True
    )


def test_categories_are_matched_case_insensitively_and_unknown_ones_dropped(
    chat_model_replying,
) -> None:
    model = chat_model_replying(
        {"query": "dress", "categories": ["Western Wear", "hats", " JEANS "]}
    )

    intent = IntentExtractor(model, CATEGORIES).extract("a dress or jeans, or a hat", [])

    assert intent.categories == ["western wear", "jeans"]


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param("Sure, here are some shoes!", id="no-tool-call"),
        pytest.param({"query": "shoes", "max_price": "cheap"}, id="invalid-args"),
    ],
)
def test_unparseable_reply_falls_back_to_plain_search(
    reply, chat_model_replying, caplog: pytest.LogCaptureFixture
) -> None:
    model = chat_model_replying(reply)

    with caplog.at_level(logging.WARNING):
        intent = IntentExtractor(model, CATEGORIES).extract("comfy shoes", [])

    assert intent == SearchIntent(query="comfy shoes", wants_products=True)
    assert "Could not parse a search intent" in caplog.text


def test_to_filter_maps_every_constraint() -> None:
    intent = SearchIntent(
        query="jeans",
        categories=["jeans"],
        brands=["Levi's"],
        colors=["Blue"],
        min_price=10,
        max_price=50,
        min_rating=4,
    )

    assert intent.to_filter() == ProductFilter(
        categories=("jeans",),
        brands=("Levi's",),
        colors=("Blue",),
        min_price=10,
        max_price=50,
        min_rating=4,
    )
