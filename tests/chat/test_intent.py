import logging

import pytest

from clothes_recommender.chat.intent import IntentExtractor, SearchIntent
from clothes_recommender.domain import ProductFilter

PRODUCT_TYPES = ["Dress", "Sneakers", "Trousers"]
SECTIONS = ["Ladieswear", "Menswear"]
COLOURS = ["Black", "Dark Blue"]


def extractor(model) -> IntentExtractor:
    return IntentExtractor(model, PRODUCT_TYPES, SECTIONS, COLOURS)


def test_extracts_structured_intent_from_tool_call(chat_model_replying) -> None:
    model = chat_model_replying(
        {"query": "running shoes", "product_types": ["Sneakers"], "max_price": 80}
    )

    intent = extractor(model).extract("running shoes under 80 euros", [])

    assert intent == SearchIntent(
        query="running shoes", product_types=["Sneakers"], max_price=80, wants_products=True
    )


def test_values_are_matched_case_insensitively_and_unknown_ones_dropped(
    chat_model_replying,
) -> None:
    model = chat_model_replying(
        {
            "query": "trousers",
            "product_types": ["trousers", "hats"],
            "sections": [" MENSWEAR "],
            "colours": ["dark blue", "teal"],
        }
    )

    intent = extractor(model).extract("dark blue men's trousers or a teal hat", [])

    assert (intent.product_types, intent.sections, intent.colours) == (
        ["Trousers"],
        ["Menswear"],
        ["Dark Blue"],
    )


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
        intent = extractor(model).extract("comfy shoes", [])

    assert intent == SearchIntent(query="comfy shoes", wants_products=True)
    assert "Could not parse a search intent" in caplog.text


def test_to_filter_maps_every_constraint() -> None:
    intent = SearchIntent(
        query="jeans",
        product_types=["Trousers"],
        sections=["Menswear"],
        colours=["Blue"],
        min_price=10,
        max_price=50,
    )

    assert intent.to_filter() == ProductFilter(
        product_types=("Trousers",),
        sections=("Menswear",),
        colours=("Blue",),
        min_price=10,
        max_price=50,
    )
