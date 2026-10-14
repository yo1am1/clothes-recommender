"""Shopping assistant: extract intent, retrieve matching products, answer grounded in them."""

from collections.abc import Sequence
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from clothes_recommender.catalog import ProductCatalog
from clothes_recommender.chat.intent import IntentExtractor, SearchIntent
from clothes_recommender.chat.prompts import ANSWER_PROMPT, render_products
from clothes_recommender.domain import Product
from clothes_recommender.search import ProductSearch

# Each ChatTurn is one message; older turns rarely matter and only cost tokens.
MAX_HISTORY_TURNS = 10


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


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
        self._answer_chain = ANSWER_PROMPT | chat_model | StrOutputParser()

    def reply(self, message: str, history: Sequence[ChatTurn]) -> AssistantReply:
        history_messages = _to_messages(history[-MAX_HISTORY_TURNS:])
        intent = self._intent_extractor.extract(message, history_messages)
        products = self._find_products(intent) if intent.wants_products else []
        text = self._answer_chain.invoke(
            {
                "message": message,
                "history": history_messages,
                "products": render_products(products),
            }
        )
        return AssistantReply(text=text, products=products, intent=intent)

    def _find_products(self, intent: SearchIntent) -> list[Product]:
        hits = self._search.by_text(
            intent.query, top_k=self._top_k, product_filter=intent.to_filter()
        )
        return self._catalog.get_many(hit.product_id for hit in hits)


def _to_messages(turns: Sequence[ChatTurn]) -> list[BaseMessage]:
    return [
        HumanMessage(turn.content) if turn.role == "user" else AIMessage(turn.content)
        for turn in turns
    ]
