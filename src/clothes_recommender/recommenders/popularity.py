"""Non-personalised ranking: the fallback when there is no history to learn from."""

from collections.abc import Collection
from datetime import timedelta
from itertools import islice

from clothes_recommender.catalog import CustomerData
from clothes_recommender.domain import ScoredProduct

# Fashion demand moves fast: what sold in the last two weeks predicts next week's sales far
# better than all-time totals (it was the strongest simple baseline in the H&M competition).
POPULARITY_WINDOW_DAYS = 14


class PopularityRecommender:
    """Ranks products by how often they were bought in the most recent window of the data.

    The window ends at the latest purchase in the data, not today, so the ranking is
    reproducible for a fixed dataset.
    """

    def __init__(self, customers: CustomerData) -> None:
        purchases = customers.purchases
        if purchases.empty:
            self._ranking: list[ScoredProduct] = []
            return
        window_start = purchases["purchase_date"].max() - timedelta(days=POPULARITY_WINDOW_DAYS)
        counts = purchases.loc[
            purchases["purchase_date"] > window_start, "product_id"
        ].value_counts()
        scored = [ScoredProduct(int(pid), float(count)) for pid, count in counts.items()]
        self._ranking = sorted(scored, key=lambda s: (-s.score, s.product_id))

    def top(self, top_k: int, exclude_ids: Collection[int] = ()) -> list[ScoredProduct]:
        """The most popular products, skipping `exclude_ids` (e.g. what the user already bought)."""
        excluded = set(exclude_ids)
        return list(islice((s for s in self._ranking if s.product_id not in excluded), top_k))
