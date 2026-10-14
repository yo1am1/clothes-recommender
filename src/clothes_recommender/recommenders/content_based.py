"""Content-based recommendations: products that look like what the user already bought."""

import logging
from collections import Counter
from collections.abc import Sequence

import numpy as np

from clothes_recommender.catalog import CustomerData, ProductCatalog
from clothes_recommender.domain import ProductFilter, Purchase, ScoredProduct
from clothes_recommender.recommenders.base import UnknownUserError
from clothes_recommender.recommenders.popularity import PopularityRecommender
from clothes_recommender.vector_store import ProductVectorStore

logger = logging.getLogger(__name__)

# Tastes drift: with a 30-day half-life last month's purchases dominate the taste vector,
# while a purchase from a season ago still counts for about an eighth.
RECENCY_HALF_LIFE_DAYS = 30
TOP_PRODUCT_TYPE_COUNT = 3


class ContentBasedRecommender:
    """Searches the CLIP index with the user's "taste vector".

    The taste vector is the recency-weighted mean of the embeddings of the purchased
    products; the search is limited to the product types the user buys most.
    """

    def __init__(
        self,
        customers: CustomerData,
        catalog: ProductCatalog,
        store: ProductVectorStore,
        popular: PopularityRecommender,
    ) -> None:
        self._customers = customers
        self._catalog = catalog
        self._store = store
        self._popular = popular

    def recommend(self, user_id: int, top_k: int = 10) -> list[ScoredProduct]:
        if self._customers.get_user(user_id) is None:
            raise UnknownUserError(user_id)
        purchases = self._customers.purchases_for(user_id)
        purchased_ids = sorted({p.product_id for p in purchases})
        vectors = self._store.get_vectors(purchased_ids)
        if not vectors:
            logger.info("User %s has no embedded purchases; using popularity instead.", user_id)
            return self._popular.top(top_k, exclude_ids=purchased_ids)

        product_filter = ProductFilter(
            product_types=self._top_product_types(purchases), exclude_ids=tuple(purchased_ids)
        )
        return self._store.search(_taste_vector(purchases, vectors), top_k, product_filter)

    def _top_product_types(self, purchases: Sequence[Purchase]) -> tuple[str, ...]:
        products = self._catalog.get_many(p.product_id for p in purchases)
        counts = Counter(p.product_type for p in products)
        ranked = sorted(counts, key=lambda product_type: (-counts[product_type], product_type))
        return tuple(ranked[:TOP_PRODUCT_TYPE_COUNT])


def _taste_vector(purchases: Sequence[Purchase], vectors: dict[int, np.ndarray]) -> np.ndarray:
    embedded = [p for p in purchases if p.product_id in vectors]
    # Ages are measured from the user's own latest purchase rather than today so results don't
    # change with the wall clock; after re-normalising only the relative weights matter anyway.
    latest = max(p.purchase_date for p in embedded)
    ages_in_days = np.array([(latest - p.purchase_date).days for p in embedded])
    weights = 0.5 ** (ages_in_days / RECENCY_HALF_LIFE_DAYS)
    taste = weights @ np.stack([vectors[p.product_id] for p in embedded])
    return taste / np.linalg.norm(taste)
