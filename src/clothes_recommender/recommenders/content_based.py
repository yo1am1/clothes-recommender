"""Content-based recommendations: products that look like what the user already bought."""

import logging
from collections import defaultdict
from collections.abc import Sequence
from datetime import date

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
# Broad groups ("Garment Upper body", "Shoes") keep results relevant without narrowing them
# to near-copies of past purchases, which H&M's ~130 fine product types would do.
TOP_PRODUCT_GROUP_COUNT = 3


class ContentBasedRecommender:
    """Searches the CLIP index with one "taste vector" per interest.

    An interest is a product group the user buys, such as dresses or shoes. Its taste vector
    is the recency-weighted mean of the embeddings of the products bought in that group, and
    it is searched within that group and the sections (e.g. Ladieswear) those products came
    from. A single vector over everything would average a dress, leggings and a men's shirt
    into "a dark long garment" and lose the dresses entirely.

    The list interleaves the top groups, each getting slots in proportion to its weight,
    and is topped up with popular products when the groups run out of matches.
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
        purchased_ids = tuple(sorted({p.product_id for p in purchases}))
        vectors = self._store.get_vectors(purchased_ids)
        embedded = [p for p in purchases if p.product_id in vectors]
        if not embedded:
            logger.info("User %s has no embedded purchases; using popularity instead.", user_id)
            return self._popular.top(top_k, exclude_ids=purchased_ids)

        interests = self._interests(embedded)
        total = sum(weight for weight, _ in interests.values())
        ranked_lists = []
        for group, (weight, group_purchases) in interests.items():
            sections = {self._catalog.get(p.product_id).section for p in group_purchases}
            product_filter = ProductFilter(
                product_groups=(group,),
                sections=tuple(sorted(sections)),
                exclude_ids=purchased_ids,
            )
            taste = _taste_vector(group_purchases, vectors)
            slots = max(1, round(top_k * weight / total))
            ranked_lists.append((slots, self._store.search(taste, top_k, product_filter)))
        recommended = _interleave(ranked_lists, top_k)
        if len(recommended) < top_k:
            # A small catalog can run out of matches in the user's groups and sections; like
            # collaborative filtering, top up with popular products at score 0.
            seen = {*purchased_ids, *(r.product_id for r in recommended)}
            recommended += [
                ScoredProduct(p.product_id, 0.0)
                for p in self._popular.top(top_k - len(recommended), exclude_ids=seen)
            ]
        return recommended

    def _interests(self, purchases: Sequence[Purchase]) -> dict[str, tuple[float, list[Purchase]]]:
        """The top product groups, heaviest first, with their weight and purchases."""
        latest = max(p.purchase_date for p in purchases)
        by_group: dict[str, list[Purchase]] = defaultdict(list)
        for purchase in purchases:
            by_group[self._catalog.get(purchase.product_id).product_group].append(purchase)
        weights = {
            group: float(_recency_weights(group_purchases, latest).sum())
            for group, group_purchases in by_group.items()
        }
        ranked = sorted(weights, key=lambda group: (-weights[group], group))
        return {g: (weights[g], by_group[g]) for g in ranked[:TOP_PRODUCT_GROUP_COUNT]}


def _recency_weights(purchases: Sequence[Purchase], latest: date) -> np.ndarray:
    # Ages are measured from the user's own latest purchase rather than today so results don't
    # change with the wall clock; after re-normalising only the relative weights matter anyway.
    ages_in_days = np.array([(latest - p.purchase_date).days for p in purchases])
    return 0.5 ** (ages_in_days / RECENCY_HALF_LIFE_DAYS)


def _taste_vector(purchases: Sequence[Purchase], vectors: dict[int, np.ndarray]) -> np.ndarray:
    latest = max(p.purchase_date for p in purchases)
    taste = _recency_weights(purchases, latest) @ np.stack(
        [vectors[p.product_id] for p in purchases]
    )
    return taste / np.linalg.norm(taste)


def _interleave(
    ranked_lists: Sequence[tuple[int, list[ScoredProduct]]], top_k: int
) -> list[ScoredProduct]:
    """Round-robin over the lists (heaviest first), each up to its slots; leftovers by score.

    Leftovers fill the list when a group runs out of matches or rounding leaves slots free.
    """
    result: list[ScoredProduct] = []
    seen: set[int] = set()
    for position in range(top_k):
        for slots, ranked in ranked_lists:
            if position < min(slots, len(ranked)) and ranked[position].product_id not in seen:
                result.append(ranked[position])
                seen.add(ranked[position].product_id)
    leftovers = sorted(
        (item for _, ranked in ranked_lists for item in ranked if item.product_id not in seen),
        key=lambda item: (-item.score, item.product_id),
    )
    for item in leftovers:
        if item.product_id not in seen:
            result.append(item)
            seen.add(item.product_id)
    return result[:top_k]
