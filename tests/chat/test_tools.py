import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from clothes_recommender.chat.assistant import MAX_TOOL_CALLS, ChatTurn, ShoppingAssistant
from clothes_recommender.domain import ProductFilter, ScoredProduct


def action(name, args, call_id="tool_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def record(assistant):
    seen = []
    model = assistant._agent

    def invoke(messages):
        seen.append(list(messages))
        return model.invoke(messages)

    assistant._agent = RunnableLambda(invoke)
    return seen


def test_refines_empty_search_and_keeps_every_shopper_constraint(
    chat_model_replying, search, catalog, monkeypatch
):
    model = chat_model_replying(
        {
            "query": "cotton party",
            "product_types": ["Dress"],
            "sections": ["Ladieswear"],
            "colours": ["Red"],
            "min_price": 10,
            "max_price": 30,
        },
        action("SearchCatalog", {"query": "red cotton dress"}),
        "Summer Dress is €24.99.",
    )
    calls = []

    def find(query, top_k, product_filter):
        calls.append((query, top_k, product_filter))
        return [ScoredProduct(3, 0.9)] if query == "red cotton dress" else []

    monkeypatch.setattr(search, "by_text", find)
    assistant = ShoppingAssistant(model, search, catalog)
    seen = record(assistant)
    reply = assistant.reply("A red dress for women between €10 and €30", [])
    expected = ProductFilter(
        product_types=("Dress",),
        sections=("Ladieswear",),
        colours=("Red",),
        min_price=10,
        max_price=30,
    )
    assert calls == [("cotton party", 5, expected), ("red cotton dress", 5, expected)]
    assert [p.product_id for p in reply.products] == [3]
    assert reply.intent.query == "red cotton dress"
    result = seen[-1][-1]
    assert isinstance(result, ToolMessage) and result.tool_call_id == "tool_1"
    assert "Summer Dress" in result.content and "€24.99" in result.content


def test_inspects_prior_cards_from_server_catalog_without_a_search(
    chat_model_replying, search, catalog
):
    model = chat_model_replying(
        {"query": "", "wants_products": False},
        action("InspectProducts", {"product_ids": [2, 1, 1, 999]}),
        "The jeans cost €29.99 and the shoes €59.99.",
    )
    assistant = ShoppingAssistant(model, search, catalog)
    seen = record(assistant)
    reply = assistant.reply(
        "Compare those two",
        [ChatTurn(role="assistant", content="Here are two pieces", product_ids=[1, 2, 999])],
    )
    assert not search.calls
    assert [p.product_id for p in reply.products] == [2, 1]
    prior = seen[0][1].content
    assert "Slim Jeans" in prior and "€29.99" in prior and "Run Shoe" in prior
    assert "999" not in prior
    assert seen[-1][-1].content.count("[id 1]") == 1


@pytest.mark.parametrize(
    "name,args",
    [
        ("DeleteProduct", {"product_ids": [1]}),
        ("SearchCatalog", {"query": " "}),
        ("SearchCatalog", {"query": "jeans", "max_price": None}),
        ("InspectProducts", {"product_ids": []}),
        ("InspectProducts", {"product_ids": [0]}),
        ("InspectProducts", {"product_ids": list(range(1, 7))}),
    ],
)
def test_invalid_actions_return_tool_error_without_executing(
    name, args, chat_model_replying, search, catalog
):
    model = chat_model_replying(
        {"query": "", "wants_products": False}, action(name, args), "Please clarify the item."
    )
    assistant = ShoppingAssistant(model, search, catalog)
    seen = record(assistant)
    reply = assistant.reply("Help", [])
    assert not search.calls and not reply.products
    result = seen[-1][-1]
    assert isinstance(result, ToolMessage) and result.tool_call_id == "tool_1"
    assert "Invalid action" in result.content


def test_tool_budget_stops_large_batches_with_grounded_fallback(
    chat_model_replying, search, catalog
):
    batch = AIMessage(
        content="",
        tool_calls=[
            {"name": "SearchCatalog", "args": {"query": "jeans"}, "id": f"call_{i}"}
            for i in range(MAX_TOOL_CALLS + 2)
        ],
    )
    model = chat_model_replying({"query": "", "wants_products": False}, batch)
    search.results = [ScoredProduct(1, 0.9)]
    reply = ShoppingAssistant(model, search, catalog).reply("Find jeans", [])
    assert len(search.calls) == MAX_TOOL_CALLS
    assert "Slim Jeans" in reply.text and "€29.99" in reply.text
    assert [p.product_id for p in reply.products] == [1]


def test_repeated_tool_requests_stop_after_budget(chat_model_replying, search, catalog):
    model = chat_model_replying(
        {"query": "", "wants_products": False},
        *[
            action("SearchCatalog", {"query": "dress"}, f"call_{i}")
            for i in range(MAX_TOOL_CALLS + 1)
        ],
    )
    reply = ShoppingAssistant(model, search, catalog).reply("Find a dress", [])
    assert len(search.calls) == MAX_TOOL_CALLS
    assert "couldn't confirm" in reply.text


def test_empty_model_answer_has_useful_fallback(chat_model_replying, search, catalog):
    model = chat_model_replying({"query": "", "wants_products": False}, " ")
    reply = ShoppingAssistant(model, search, catalog).reply("Find a coat", [])
    assert reply.text and not reply.products


def test_history_product_context_is_bounded():
    for ids in [[0], list(range(1, 7))]:
        with pytest.raises(ValidationError):
            ChatTurn(role="assistant", content="Products", product_ids=ids)


@pytest.mark.parametrize("call_id", [None, ""])
def test_missing_tool_call_id_does_not_execute_actions(
    chat_model_replying, search, catalog, call_id
):
    model = chat_model_replying(
        {"query": "", "wants_products": False},
        action("SearchCatalog", {"query": "dress"}, call_id),
    )
    reply = ShoppingAssistant(model, search, catalog).reply("Help", [])
    assert not search.calls and reply.text
