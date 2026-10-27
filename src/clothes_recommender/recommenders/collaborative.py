"""Item-to-item collaborative filtering: "shoppers who bought what you bought also bought..."."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from clothes_recommender.catalog import CustomerData, ProductCatalog
from clothes_recommender.domain import ScoredProduct
from clothes_recommender.recommenders.base import UnknownUserError
from clothes_recommender.recommenders.popularity import PopularityRecommender

# A candidate's co-purchase count is divided by (its number of buyers) ** POPULARITY_DAMPING,
# so best-sellers that co-occur with everything don't fill every list. 0 ranks by raw
# co-occurrence, 1 by its share of the candidate's buyers; 0.3 scored best on three
# validation weeks (0.5 was close, 0.7 clearly worse).
POPULARITY_DAMPING = 0.3
# Recent purchases say more about what a shopper wants next: a purchase a week older than
# the user's latest counts half as much. 7 days beat 14 and 28 on the validation weeks.
HISTORY_HALF_LIFE_DAYS = 7
# Reasons name at most this many of the user's purchases.
MAX_REASONS = 2


class CollaborativeRecommender:
    """Scores products by how many shoppers bought them together with the user's purchases.

    Co-occurrence is counted over every shopper's purchases (`shopper_purchases`), not just
    the sampled customers: with only a few thousand customers, most pairs of products share
    no buyer at all. Two products co-occur when one shopper bought both at any time in the
    data, so this also captures "bought later", which same-day co-purchase links miss.

    Products with no co-purchase evidence are ranked by popularity at score 0.
    """

    def __init__(
        self,
        customers: CustomerData,
        catalog: ProductCatalog,
        shopper_purchases: pd.DataFrame,
        popular: PopularityRecommender,
    ) -> None:
        """`shopper_purchases` needs shopper_id and product_id columns."""
        self._customers = customers
        self._catalog = catalog
        self._popular = popular
        products = pd.Categorical(shopper_purchases["product_id"])
        self._product_ids = products.categories.to_numpy()
        self._column = {int(pid): col for col, pid in enumerate(self._product_ids)}
        shoppers = pd.Categorical(shopper_purchases["shopper_id"]).codes
        bought = csr_array(
            (np.ones(len(shopper_purchases), dtype=np.float32), (shoppers, products.codes)),
            shape=(shoppers.max() + 1 if len(shoppers) else 0, len(self._product_ids)),
        )
        # Implicit feedback: buying an item twice is not twice the preference, so binarise.
        bought = (bought > 0).astype(np.float32)
        buyers = np.asarray(bought.sum(axis=0)).ravel()
        co_bought = (bought.T @ bought).tocsr()
        co_bought.setdiag(0)
        co_bought.eliminate_zeros()
        self._co_bought = co_bought
        self._damping = np.maximum(buyers, 1) ** -POPULARITY_DAMPING

    def recommend(self, user_id: int, top_k: int = 10) -> list[ScoredProduct]:
        """Best-scoring products, topped up with popular ones when evidence runs out."""
        scores, history = self._scores(user_id)
        candidates = np.flatnonzero(scores > 0)
        # Ties break by lower product id, which keeps results deterministic.
        order = candidates[np.lexsort((self._product_ids[candidates], -scores[candidates]))]
        recommended = [
            ScoredProduct(int(self._product_ids[col]), float(scores[col])) for col in order[:top_k]
        ]
        if len(recommended) < top_k:
            seen = history | {r.product_id for r in recommended}
            recommended += [
                ScoredProduct(p.product_id, 0.0)
                for p in self._popular.top(top_k - len(recommended), exclude_ids=seen)
            ]
        return recommended

    def reasons(self, user_id: int, product_id: int) -> list[str]:
        """Which of the user's purchases point to this product, strongest first."""
        purchases = self._recent_purchases(user_id)
        column = self._column.get(product_id)
        contributions = []
        if column is not None and product_id not in purchases:
            for pid, weight in purchases.items():
                if (row := self._column.get(pid)) is not None:
                    together = int(self._co_bought[row, column])
                    if together:
                        contributions.append((weight * together, together, pid))
        if not contributions:
            return ["Popular in the last week; no shopper bought it with this customer's items."]
        contributions.sort(key=lambda c: (-c[0], c[2]))
        return [
            f"{together} {'shopper' if together == 1 else 'shoppers'} who bought "
            f"{self._name(pid)} also bought this."
            for _, together, pid in contributions[:MAX_REASONS]
        ]

    def _scores(self, user_id: int) -> tuple[np.ndarray, set[int]]:
        purchases = self._recent_purchases(user_id)
        known = [
            (self._column[pid], weight) for pid, weight in purchases.items() if pid in self._column
        ]
        scores = np.zeros(len(self._product_ids))
        if known:
            rows, weights = zip(*known, strict=True)
            scores = np.asarray(weights) @ self._co_bought[list(rows)] * self._damping
            scores[list(rows)] = 0.0  # never re-recommend a purchase
        return scores, set(purchases)

    def _recent_purchases(self, user_id: int) -> dict[int, float]:
        """Each purchased product with its recency weight (latest purchase = 1)."""
        if self._customers.get_user(user_id) is None:
            raise UnknownUserError(user_id)
        purchases = self._customers.purchases_for(user_id)
        if not purchases:
            return {}
        latest = purchases[0].purchase_date
        weights: dict[int, float] = {}
        for purchase in purchases:  # most recent first, so the first date seen is the latest
            age = (latest - purchase.purchase_date).days
            weights.setdefault(purchase.product_id, 0.5 ** (age / HISTORY_HALF_LIFE_DAYS))
        return weights

    def _name(self, product_id: int) -> str:
        # Colourways of one style share a name ("Spring"), so the colour tells them apart.
        product = self._catalog.get(product_id)
        if product is None:
            return f"product {product_id}"
        return f"{product.name} ({product.colour.lower()})"
