"""Non-personalised ranking: the fallback when there is no history to learn from."""

import math
from collections.abc import Collection
from itertools import islice

from clothes_recommender.catalog import ProductCatalog
from clothes_recommender.domain import ScoredProduct

# Bayesian-average prior: every product is scored as if it already had
# PRIOR_REVIEW_COUNT extra reviews of PRIOR_RATING stars.
PRIOR_RATING = 3.5
PRIOR_REVIEW_COUNT = 20


def _popularity_score(rating: float, review_count: int) -> float:
    """Bayesian-average rating x log(1 + review count).

    A raw mean lets one 5-star review outrank thousands of 4.6-star ones. Shrinking each
    rating towards the prior makes thinly reviewed products earn their rank, and the log
    term rewards proven demand with diminishing returns so mega-sellers don't swamp the list.
    """
    bayesian_rating = (PRIOR_RATING * PRIOR_REVIEW_COUNT + rating * review_count) / (
        PRIOR_REVIEW_COUNT + review_count
    )
    return bayesian_rating * math.log1p(review_count)


class PopularityRecommender:
    def __init__(self, catalog: ProductCatalog) -> None:
        scored = [
            ScoredProduct(p.product_id, _popularity_score(p.rating, p.review_count))
            for p in catalog
        ]
        self._ranking = sorted(scored, key=lambda s: (-s.score, s.product_id))

    def top(self, top_k: int, exclude_ids: Collection[int] = ()) -> list[ScoredProduct]:
        """The most popular products, skipping `exclude_ids` (e.g. what the user already bought)."""
        excluded = set(exclude_ids)
        return list(islice((s for s in self._ranking if s.product_id not in excluded), top_k))
