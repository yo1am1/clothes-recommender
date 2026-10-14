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
        "e.g. 'red floral summer dress'. Leave out price and audience."
    )
    product_types: list[str] = Field(default_factory=list, description="Only known product types.")
    sections: list[str] = Field(default_factory=list, description="Only known sections.")
    colours: list[str] = Field(default_factory=list, description="Only known colours.")
    min_price: float | None = Field(default=None, description="Minimum price in euros.")
    max_price: float | None = Field(default=None, description="Maximum price in euros.")
    wants_products: bool = Field(
        default=True,
        description="False for small talk or questions that need no product search.",
    )

    def to_filter(self) -> ProductFilter:
        return ProductFilter(
            product_types=tuple(self.product_types),
            sections=tuple(self.sections),
            colours=tuple(self.colours),
            min_price=self.min_price,
            max_price=self.max_price,
        )


class IntentExtractor:
    def __init__(
        self,
        chat_model: BaseChatModel,
        product_types: Sequence[str],
        sections: Sequence[str],
        colours: Sequence[str],
    ) -> None:
        self._vocabularies = {
            "product_types": _by_key(product_types),
            "sections": _by_key(sections),
            "colours": _by_key(colours),
        }
        # Tool calling instead of OpenAI's default strict json_schema mode: many
        # OpenAI-compatible gateways and open models reject json_schema, while tool calling
        # is supported almost everywhere. include_raw=True returns parse errors instead of
        # raising, so a malformed reply degrades search quality rather than failing the chat.
        structured_model = chat_model.with_structured_output(
            SearchIntent, method="function_calling", include_raw=True
        )
        prompt = INTENT_PROMPT.partial(
            product_types=", ".join(product_types),
            sections=", ".join(sections),
            colours=", ".join(colours),
        )
        self._chain = prompt | structured_model

    def extract(self, message: str, history: Sequence[BaseMessage]) -> SearchIntent:
        result = self._chain.invoke({"message": message, "history": list(history)})
        intent = result["parsed"]
        if intent is None:
            logger.warning(
                "Could not parse a search intent (%s); searching with the raw message.",
                result["parsing_error"] or "model returned no structured output",
            )
            return SearchIntent(query=message)
        known_values = {
            field: _known(getattr(intent, field), vocabulary, field)
            for field, vocabulary in self._vocabularies.items()
        }
        # An empty CLIP query would match arbitrary products; the message is a safe default.
        return intent.model_copy(update={"query": intent.query.strip() or message, **known_values})


def _by_key(values: Sequence[str]) -> dict[str, str]:
    return {value.lower(): value for value in values}


def _known(values: Sequence[str], vocabulary: dict[str, str], field: str) -> list[str]:
    """Map to the catalog's spelling, dropping values the catalog doesn't have."""
    known = []
    for value in values:
        canonical = vocabulary.get(value.strip().lower())
        if canonical is None:
            logger.info("Ignoring unknown %s value %r from the LLM.", field, value)
        else:
            known.append(canonical)
    return known
