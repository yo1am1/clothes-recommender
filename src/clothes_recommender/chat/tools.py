"""Validated read-only actions available to the shopping assistant."""

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class SearchCatalog(BaseModel):
    """Refine the visual query, keeping the shopper's existing metadata and budget filters."""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=500, pattern=r"\S")


class InspectProducts(BaseModel):
    """Read exact catalog details for up to five product IDs, for questions or comparisons."""

    model_config = ConfigDict(extra="forbid")
    product_ids: list[PositiveInt] = Field(min_length=1, max_length=5)


TOOLS = {schema.__name__: schema for schema in (SearchCatalog, InspectProducts)}
