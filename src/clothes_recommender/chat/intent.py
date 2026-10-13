"""Structured search intent extracted from a shopper's chat message."""

import logging
from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from clothes_recommender.chat.prompts import INTENT_PROMPT
from clothes_recommender.domain import ProductFilter

logger = logging.getLogger(__name__)


class SearchIntent(BaseModel):
    """A product search request: a visual description plus metadata constraints."""

    query: str = Field(
        description="Visual description of the wanted item for image search, "
        "e.g. 'red floral summer dress'. Leave out price, rating and brand."
    )
    categories: list[str] = Field(default_factory=list, description="Only known categories.")
    brands: list[str] = Field(default_factory=list)
    colors: list[str] = Field(default_factory=list)
    min_price: float | None = None
    max_price: float | None = None
    min_rating: float | None = Field(default=None, description="Minimum rating out of 5.")
    wants_products: bool = Field(
        default=True,
        description="False for small talk or questions that need no product search.",
    )

    def to_filter(self) -> ProductFilter:
        return ProductFilter(
            categories=tuple(self.categories),
            brands=tuple(self.brands),
            colors=tuple(self.colors),
            min_price=self.min_price,
            max_price=self.max_price,
            min_rating=self.min_rating,
        )


class IntentExtractor:
    def __init__(self, chat_model: BaseChatModel, categories: Sequence[str]) -> None:
        self._categories_by_key = {category.lower(): category for category in categories}
        # Tool calling instead of OpenAI's default strict json_schema mode: many
        # OpenAI-compatible gateways and open models reject json_schema, while tool calling
        # is supported almost everywhere. include_raw=True returns parse errors instead of
        # raising, so a malformed reply degrades search quality rather than failing the chat.
        structured_model = chat_model.with_structured_output(
            SearchIntent, method="function_calling", include_raw=True
        )
        self._chain = INTENT_PROMPT.partial(categories=", ".join(categories)) | structured_model

    def extract(self, message: str, history: Sequence[BaseMessage]) -> SearchIntent:
        result = self._chain.invoke({"message": message, "history": list(history)})
        intent = result["parsed"]
        if intent is None:
            logger.warning(
                "Could not parse a search intent (%s); searching with the raw message.",
                result["parsing_error"] or "model returned no structured output",
            )
            return SearchIntent(query=message)
        return intent.model_copy(
            update={
                # An empty CLIP query would match arbitrary products; the message is a safe default.
                "query": intent.query.strip() or message,
                "categories": self._known_categories(intent.categories),
            }
        )

    def _known_categories(self, categories: Sequence[str]) -> list[str]:
        """Map to the catalog's spelling, dropping categories the catalog doesn't have."""
        known = []
        for category in categories:
            canonical = self._categories_by_key.get(category.strip().lower())
            if canonical is None:
                logger.info("Ignoring unknown category %r from the LLM.", category)
            else:
                known.append(canonical)
        return known
