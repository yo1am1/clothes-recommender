"""Composition root: wires settings, data, models and stores into ready-to-use services."""

import logging
from dataclasses import dataclass

from clothes_recommender.catalog import CustomerData, ProductCatalog
from clothes_recommender.chat.assistant import ShoppingAssistant
from clothes_recommender.chat.llm import build_chat_model
from clothes_recommender.config import Settings
from clothes_recommender.embeddings import ClipEncoder
from clothes_recommender.recommenders.collaborative import CollaborativeRecommender
from clothes_recommender.recommenders.content_based import ContentBasedRecommender
from clothes_recommender.recommenders.popularity import PopularityRecommender
from clothes_recommender.search import ProductSearch
from clothes_recommender.vector_store import ProductVectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Services:
    catalog: ProductCatalog
    customers: CustomerData
    store: ProductVectorStore
    search: ProductSearch
    content_based: ContentBasedRecommender
    collaborative: CollaborativeRecommender
    # None when no LLM is configured: every other feature still works without an API key.
    assistant: ShoppingAssistant | None


def build_services(settings: Settings) -> Services:
    catalog = ProductCatalog.from_parquet(settings.products_path)
    customers = CustomerData.from_csv(settings.users_path, settings.purchases_path)
    store = ProductVectorStore.from_settings(settings)
    encoder = ClipEncoder(settings.clip_model_name)
    encoder.load()
    search = ProductSearch(encoder, store)
    popularity = PopularityRecommender(catalog)

    assistant = None
    if settings.llm_configured:
        assistant = ShoppingAssistant(build_chat_model(settings), search, catalog)
    else:
        logger.warning("LLM_MODEL is not set: the shopping assistant is disabled.")

    return Services(
        catalog=catalog,
        customers=customers,
        store=store,
        search=search,
        content_based=ContentBasedRecommender(customers, catalog, store, popularity),
        collaborative=CollaborativeRecommender(customers, popularity),
        assistant=assistant,
    )
