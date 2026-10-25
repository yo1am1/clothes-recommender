"""Non-personalised ranking: the fallback when there is no history to learn from."""

from collections.abc import Collection
from datetime import timedelta
from itertools import islice

import pandas as pd

from clothes_recommender.domain import ScoredProduct

# Fashion demand moves fast. On three validation weeks of the H&M sample, the last 7 days
# of sales predicted the next week better than 5 or 14 days.
POPULARITY_WINDOW_DAYS = 7


class PopularityRecommender:
    """Ranks products by how often they were bought in the most recent window of the data.

    Pass every shopper's purchases when available: counts from a few thousand sampled
    customers are too sparse to rank 14,000 products. The window ends at the latest
    purchase in the data, not today, so the ranking is reproducible for a fixed dataset.
    """

    def __init__(self, purchases: pd.DataFrame) -> None:
        """`purchases` needs product_id and purchase_date columns."""
        if purchases.empty:
            self._ranking: list[ScoredProduct] = []
            return
        dates = pd.to_datetime(purchases["purchase_date"])
        window_start = dates.max() - timedelta(days=POPULARITY_WINDOW_DAYS)
        counts = purchases.loc[dates > window_start, "product_id"].value_counts()
        scored = [ScoredProduct(int(pid), float(count)) for pid, count in counts.items()]
        self._ranking = sorted(scored, key=lambda s: (-s.score, s.product_id))

    def top(self, top_k: int, exclude_ids: Collection[int] = ()) -> list[ScoredProduct]:
        """The most popular products, skipping `exclude_ids` (e.g. what the user already bought)."""
        excluded = set(exclude_ids)
        return list(islice((s for s in self._ranking if s.product_id not in excluded), top_k))
