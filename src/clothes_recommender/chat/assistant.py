"""Shopping assistant: extract intent, retrieve matching products, answer grounded in them."""

from collections.abc import Sequence
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field, PositiveInt, ValidationError

from clothes_recommender.catalog import ProductCatalog
from clothes_recommender.chat.intent import IntentExtractor, SearchIntent
from clothes_recommender.chat.prompts import ANSWER_PROMPT, render_products
from clothes_recommender.chat.tools import TOOLS, InspectProducts, SearchCatalog
from clothes_recommender.domain import Product
from clothes_recommender.search import ProductSearch

# Each ChatTurn is one message; older turns rarely matter and only cost tokens.
MAX_HISTORY_TURNS = 10
MAX_TOOL_CALLS = 4


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)
    product_ids: list[PositiveInt] = Field(default_factory=list, max_length=5)


class AssistantReply(BaseModel):
    text: str
    products: list[Product]
    intent: SearchIntent


class ShoppingAssistant:
    def __init__(
        self,
        chat_model: BaseChatModel,
        search: ProductSearch,
        catalog: ProductCatalog,
        top_k: int = 5,
        intent_extractor: IntentExtractor | None = None,
    ) -> None:
        self._search = search
        self._catalog = catalog
        self._top_k = top_k
        self._intent_extractor = intent_extractor or IntentExtractor(
            chat_model,
            product_types=catalog.distinct("product_type"),
            sections=catalog.distinct("section"),
            colours=catalog.distinct("colour"),
        )
        self._agent = chat_model.bind_tools(list(TOOLS.values()))

    def reply(self, message: str, history: Sequence[ChatTurn]) -> AssistantReply:
        history_messages = _to_messages(history[-MAX_HISTORY_TURNS:], self._catalog)
        intent = self._intent_extractor.extract(message, history_messages)
        products = self._find_products(intent) if intent.wants_products else []
        messages = ANSWER_PROMPT.invoke(
            {"message": message, "history": history_messages, "products": render_products(products)}
        ).to_messages()
        calls = 0
        while True:
            answer = self._agent.invoke(messages)
            if not answer.tool_calls:
                text = StrOutputParser().invoke(answer).strip() or _fallback(products)
                return AssistantReply(text=text, products=products, intent=intent)
            messages.append(answer)
            for call in answer.tool_calls:
                if calls >= MAX_TOOL_CALLS or not call.get("id"):
                    return AssistantReply(
                        text=_fallback(products), products=products, intent=intent
                    )
                calls += 1
                schema = TOOLS.get(call["name"])
                try:
                    action = schema.model_validate(call["args"]) if schema else None
                except ValidationError:
                    action = None
                if action is None:
                    result = (
                        "Invalid action. Use SearchCatalog with a visual query or "
                        "InspectProducts with one to five positive catalog product IDs."
                    )
                else:
                    if isinstance(action, SearchCatalog):
                        intent = intent.model_copy(
                            update={"query": action.query.strip(), "wants_products": True}
                        )
                        products = self._find_products(intent)
                    elif isinstance(action, InspectProducts):
                        products = self._catalog.get_many(dict.fromkeys(action.product_ids))
                    result = render_products(products)
                messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    def _find_products(self, intent: SearchIntent) -> list[Product]:
        hits = self._search.by_text(
            intent.query, top_k=self._top_k, product_filter=intent.to_filter()
        )
        return self._catalog.get_many(hit.product_id for hit in hits)


def _to_messages(turns: Sequence[ChatTurn], catalog: ProductCatalog) -> list[BaseMessage]:
    messages = []
    for turn in turns:
        if turn.role == "user":
            messages.append(HumanMessage(turn.content))
        else:
            products = catalog.get_many(turn.product_ids)
            context = (
                "\nReferenced catalog items:\n" + render_products(products) if products else ""
            )
            messages.append(AIMessage(turn.content + context))
    return messages


def _fallback(products: Sequence[Product]) -> str:
    if not products:
        return "I couldn't confirm matching pieces. Try another description or adjust your filters."
    return "Here are the pieces I could confirm:\n" + "\n".join(
        f"- {p.name} · {p.colour} · €{p.price:.2f}" for p in products
    )
