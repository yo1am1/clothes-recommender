from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from clothes_recommender.catalog import ProductCatalog
from clothes_recommender.chat.assistant import MAX_HISTORY_TURNS, ChatTurn, ShoppingAssistant
from clothes_recommender.chat.intent import SearchIntent
from clothes_recommender.chat.prompts import render_products
from clothes_recommender.domain import ProductFilter, ScoredProduct


class RecordingExtractor:
    def __init__(self, intent: SearchIntent) -> None:
        self.intent = intent
        self.histories: list[list[BaseMessage]] = []

    def extract(self, message: str, history: Sequence[BaseMessage]) -> SearchIntent:
        self.histories.append(list(history))
        return self.intent


def test_reply_searches_with_intent_and_returns_grounded_products(
    chat_model_replying, search, catalog: ProductCatalog
) -> None:
    model = chat_model_replying(
        {"query": "blue jeans", "product_types": ["Trousers"], "sections": ["Menswear"]},
        "Try the **Slim Jeans**.",
    )
    search.results = [ScoredProduct(1, 0.9), ScoredProduct(999, 0.8)]

    reply = ShoppingAssistant(model, search, catalog, top_k=3).reply("blue jeans for men", [])

    assert search.calls == [
        ("blue jeans", 3, ProductFilter(product_types=("Trousers",), sections=("Menswear",)))
    ]
    # Ids missing from the catalog are skipped rather than surfaced as broken products.
    assert [p.product_id for p in reply.products] == [1]
    assert reply.text == "Try the **Slim Jeans**."
    assert reply.intent.query == "blue jeans"


def test_small_talk_skips_search(chat_model_replying, search, catalog: ProductCatalog) -> None:
    model = chat_model_replying({"query": "", "wants_products": False}, "You're welcome!")
    search.results = [ScoredProduct(1, 0.9)]

    reply = ShoppingAssistant(model, search, catalog).reply("thanks!", [])

    assert search.calls == []
    assert reply.products == []
    assert reply.text == "You're welcome!"


def test_history_is_trimmed_and_passed_as_chat_messages(
    chat_model_replying, search, catalog: ProductCatalog
) -> None:
    extractor = RecordingExtractor(SearchIntent(query="", wants_products=False))
    assistant = ShoppingAssistant(
        chat_model_replying("ok"), search, catalog, intent_extractor=extractor
    )
    history = [
        ChatTurn(role="user" if i % 2 == 0 else "assistant", content=f"turn {i}")
        for i in range(MAX_HISTORY_TURNS + 4)
    ]

    assistant.reply("and now?", history)

    (seen,) = extractor.histories
    assert len(seen) == MAX_HISTORY_TURNS
    assert seen[0] == HumanMessage("turn 4")
    assert seen[-1] == AIMessage(f"turn {MAX_HISTORY_TURNS + 3}")


def test_render_products_lists_grounding_fields(catalog: ProductCatalog) -> None:
    lines = render_products(catalog.get_many([2, 1])).splitlines()

    assert lines[0].startswith("1. [id 2] Run Shoe")
    assert "€59.99" in lines[0]
    assert "Sneakers" in lines[0]
    assert "Lightweight running shoes." in lines[0]
    assert lines[1].startswith("2. [id 1]")
    assert render_products([]) == "(none)"
