"""Shopping assistant: extract intent, retrieve matching products, answer grounded in them."""

from collections.abc import Sequence
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field, PositiveInt

from clothes_recommender.catalog import ProductCatalog
from clothes_recommender.chat.agent import MAX_TOOL_CALLS as MAX_TOOL_CALLS
from clothes_recommender.chat.agent import run_agent
from clothes_recommender.chat.intent import IntentExtractor, SearchIntent
from clothes_recommender.chat.prompts import ANSWER_PROMPT, render_products
from clothes_recommender.chat.tools import TOOLS, InspectProducts, SearchCatalog
from clothes_recommender.domain import Product
from clothes_recommender.search import ProductSearch

# Each ChatTurn is one message; older turns rarely matter and only cost tokens.
MAX_HISTORY_TURNS = 10


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)
    product_ids: list[PositiveInt] = Field(default_factory=list, max_length=5)
    drink_ids: list[PositiveInt] = Field(default_factory=list, max_length=5)


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

        def execute(action: BaseModel) -> str:
            nonlocal intent, products
            if isinstance(action, SearchCatalog):
                intent = intent.model_copy(
                    update={"query": action.query.strip(), "wants_products": True}
                )
                products = self._find_products(intent)
            elif isinstance(action, InspectProducts):
                products = self._catalog.get_many(dict.fromkeys(action.product_ids))
            return render_products(products)

        text = run_agent(self._agent, messages, TOOLS, execute)
        return AssistantReply(text=text or _fallback(products), products=products, intent=intent)

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
