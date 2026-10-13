"""User-based collaborative filtering: "people who bought what you bought also bought..."."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_array, diags_array

from clothes_recommender.catalog import CustomerData
from clothes_recommender.domain import ScoredProduct
from clothes_recommender.recommenders.base import UnknownUserError
from clothes_recommender.recommenders.popularity import PopularityRecommender

NEIGHBOUR_COUNT = 20


@dataclass(frozen=True)
class SimilarUser:
    user_id: int
    similarity: float


class CollaborativeRecommender:
    """Cosine similarity between users' sets of purchased products.

    Users without purchases (or without any overlap) have no neighbours; like the
    content-based recommender, `recommend` then falls back to popularity.
    """

    def __init__(self, customers: CustomerData, popular: PopularityRecommender) -> None:
        self._popular = popular
        self._user_ids = customers.user_ids()
        self._row_of_user = {user_id: row for row, user_id in enumerate(self._user_ids)}

        purchases = customers.purchases
        user_rows = pd.Categorical(purchases["user_id"], categories=self._user_ids).codes
        products = pd.Categorical(purchases["product_id"])
        self._product_ids = products.categories.to_numpy()
        known = user_rows >= 0  # -1 marks purchases of users missing from the users file
        counts = csr_array(
            (np.ones(known.sum()), (user_rows[known], products.codes[known])),
            shape=(len(self._user_ids), len(self._product_ids)),
        )
        # Implicit feedback: buying an item twice is not twice the preference, so binarise.
        self._purchased = (counts > 0).astype(np.float64)

        norms = np.sqrt(self._purchased.sum(axis=1))
        inverse_norms = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
        self._normalised = diags_array(inverse_norms) @ self._purchased

    def similar_users(self, user_id: int, top_k: int = 5) -> list[SimilarUser]:
        """Most similar other users, excluding anyone with zero overlap."""
        row = self._row(user_id)
        similarities = self._normalised @ self._normalised[[row]].toarray().ravel()
        similarities[row] = 0.0  # a user is not their own neighbour
        return [
            SimilarUser(user_id=self._user_ids[r], similarity=float(similarities[r]))
            for r in _top_positive(similarities, top_k)
        ]

    def recommend(self, user_id: int, top_k: int = 10) -> list[ScoredProduct]:
        """Products the neighbours bought, each scored by the summed similarity of its buyers.

        When the neighbourhood yields fewer than top_k products, the list is topped up with
        popular products at score 0 (no collaborative evidence for them).
        """
        neighbours = self.similar_users(user_id, NEIGHBOUR_COUNT)
        purchased_cols = self._purchased[[self._row(user_id)]].indices
        scores = np.zeros(len(self._product_ids))
        if neighbours:
            neighbour_rows = [self._row_of_user[n.user_id] for n in neighbours]
            weights = np.array([n.similarity for n in neighbours])
            scores = self._purchased[neighbour_rows].T @ weights
        scores[purchased_cols] = 0.0  # never re-recommend a purchase
        recommended = [
            ScoredProduct(product_id=int(self._product_ids[col]), score=float(scores[col]))
            for col in _top_positive(scores, top_k)
        ]
        if len(recommended) < top_k:
            seen = {
                *self._product_ids[purchased_cols].tolist(),
                *(r.product_id for r in recommended),
            }
            recommended += [
                ScoredProduct(p.product_id, 0.0)
                for p in self._popular.top(top_k - len(recommended), exclude_ids=seen)
            ]
        return recommended

    def _row(self, user_id: int) -> int:
        if user_id not in self._row_of_user:
            raise UnknownUserError(user_id)
        return self._row_of_user[user_id]


def _top_positive(scores: np.ndarray, top_k: int) -> np.ndarray:
    """Indices of the top_k positive scores, highest first.

    Rows and columns are ordered by ascending id, so breaking ties by lower index
    breaks them by lower id, which keeps results deterministic.
    """
    candidates = np.flatnonzero(scores > 0)
    order = np.lexsort((candidates, -scores[candidates]))
    return candidates[order[:top_k]]
