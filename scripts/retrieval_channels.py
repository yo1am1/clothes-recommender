"""Offline-only, past-observed family, recent-basket and SVD fold-in retrieval.

No channel fills with popular items. Scores are nonnegative evidence strengths, not
purchase probabilities. Crowd shopper IDs never identify personal-history users.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from scipy.sparse.linalg import svds

from clothes_recommender.catalog import CustomerData
from clothes_recommender.domain import ScoredProduct
from clothes_recommender.recommenders.base import UnknownUserError

CHANNELS = ("family", "recent_basket", "latent")
RECENT_DAYS = 28
HISTORY_HALF_LIFE = 7
FACTORS = 32
MIN_SUPPORT = 2


class ComplementaryRetrieval:
    """Fixed experimental retrieval channels; construction never writes artifacts.

    Family seeds are personal purchases in the inclusive last 28 days. Candidate
    popularity counts distinct shopper/day/item observations in that period. Basket
    edges use support * log1p(lift), with at least two distinct same-day baskets.
    SVD uses binary shopper/item membership, latest-event 28-day recency weights and
    buyer-count**-0.3 damping. Personal seeds have a seven-day half-life as of cutoff.
    """

    def __init__(
        self,
        products: pd.DataFrame,
        customers: CustomerData,
        crowd: pd.DataFrame,
        as_of: date,
    ) -> None:
        if products["product_id"].duplicated().any():
            raise ValueError("Products must have unique product_id values")
        self._customers = customers
        self._as_of = as_of
        self._recent_start = as_of - timedelta(days=RECENT_DAYS - 1)
        past = crowd.assign(purchase_date=pd.to_datetime(crowd["purchase_date"]).dt.date)
        past = past[past["purchase_date"] <= as_of]
        self._ids = np.sort(
            products.loc[products["product_id"].isin(past["product_id"]), "product_id"].to_numpy(
                dtype=np.int64
            )
        )
        self._column = {int(pid): col for col, pid in enumerate(self._ids)}
        self._families = {}
        if "product_code" in products:
            codes = products["product_code"]
            known = products[codes.notna() & codes.astype(str).str.strip().ne("")]
            self._families = dict(zip(known["product_id"], known["product_code"], strict=True))
        recent = past[past["purchase_date"] >= self._recent_start].drop_duplicates(
            ["shopper_id", "purchase_date", "product_id"]
        )
        counts = recent["product_id"].value_counts()
        self._family_popularity = 1 + np.log1p(counts.reindex(self._ids, fill_value=0).to_numpy())
        self._basket = self._basket_edges(recent)
        self._damping = np.ones(len(self._ids))
        self._factors = self._fit_svd(past)

    def _basket_edges(self, recent: pd.DataFrame) -> csr_array:
        n_items = len(self._ids)
        if recent.empty or not n_items:
            return csr_array((n_items, n_items))
        # Include baskets containing only non-catalog items in the support denominator.
        basket_ids, baskets = pd.factorize(
            pd.MultiIndex.from_frame(recent[["shopper_id", "purchase_date"]]), sort=True
        )
        columns = recent["product_id"].map(self._column)
        known = columns.notna().to_numpy()
        bought = csr_array(
            (np.ones(known.sum()), (basket_ids[known], columns[known].to_numpy(dtype=int))),
            shape=(len(baskets), n_items),
        )
        counts = np.asarray(bought.sum(axis=0)).ravel()
        pairs = (bought.T @ bought).tocoo()
        keep = (pairs.row != pairs.col) & (pairs.data >= MIN_SUPPORT)
        rows, cols, together = pairs.row[keep], pairs.col[keep], pairs.data[keep]
        support = together / len(baskets)
        lift = together * len(baskets) / (counts[rows] * counts[cols])
        return csr_array((support * np.log1p(lift), (rows, cols)), shape=(n_items, n_items))

    def _fit_svd(self, past: pd.DataFrame) -> np.ndarray:
        latest = (
            past[past["product_id"].isin(self._ids)]
            .groupby(["shopper_id", "product_id"], as_index=False)["purchase_date"]
            .max()
        )
        shoppers, identities = pd.factorize(latest["shopper_id"], sort=True)
        rank = min(FACTORS, len(identities) - 1, len(self._ids) - 1)
        if rank < 1:
            return np.empty((len(self._ids), 0))
        columns = latest["product_id"].map(self._column).to_numpy(dtype=int)
        buyers = np.bincount(columns, minlength=len(self._ids))
        self._damping = np.maximum(buyers, 1) ** -0.3
        age = (pd.Timestamp(self._as_of) - pd.to_datetime(latest["purchase_date"])).dt.days
        # A common rescaling preserves right singular vectors and avoids stale-data underflow.
        relative_age = (age - age.min()).to_numpy()
        weights = np.exp2(-relative_age / RECENT_DAYS) * self._damping[columns]
        bought = csr_array((weights, (shoppers, columns)), shape=(len(identities), len(self._ids)))
        _, _, vt = svds(bought, k=rank, return_singular_vectors="vh", rng=42)
        return vt.T

    def _user_scores(self, user_id: int) -> dict[str, np.ndarray]:
        if self._customers.get_user(user_id) is None:
            raise UnknownUserError(user_id)
        result = {name: np.zeros(len(self._ids)) for name in CHANNELS}
        history = {}
        recent_seeds = set()
        for purchase in self._customers.purchases_for(user_id):
            if purchase.purchase_date <= self._as_of:
                age = (self._as_of - purchase.purchase_date).days
                history[purchase.product_id] = max(
                    history.get(purchase.product_id, 0.0), np.exp2(-age / HISTORY_HALF_LIFE)
                )
                if age < RECENT_DAYS:
                    recent_seeds.add(purchase.product_id)
        family_weights = {}
        for pid, weight in history.items():
            if pid in self._families and pid in recent_seeds:
                family = self._families[pid]
                family_weights[family] = family_weights.get(family, 0.0) + weight
        result["family"] = (
            np.array([family_weights.get(self._families.get(int(pid)), 0.0) for pid in self._ids])
            * self._family_popularity
        )
        known = [
            (self._column[pid], weight) for pid, weight in history.items() if pid in self._column
        ]
        if known:
            columns, weights = zip(*known, strict=True)
            columns = np.asarray(columns)
            weights = np.asarray(weights)
            result["recent_basket"] = np.asarray(weights @ self._basket[columns]).ravel()
            if self._factors.shape[1]:
                profile = (weights * self._damping[columns]) @ self._factors[columns]
                projected = self._factors @ profile
                # Roundoff in orthogonal/disconnected components is not retrieval evidence.
                result["latent"] = np.where(projected > 1e-12, projected, 0.0)
        owned = [self._column[pid] for pid in history if pid in self._column]
        for values in result.values():
            values[owned] = 0.0
        return result

    def scores(self, user_id: int, product_ids: list[int] | np.ndarray) -> dict[str, np.ndarray]:
        """Channel scores aligned to supplied IDs, including duplicates; unknown IDs get zero."""
        values = self._user_scores(user_id)
        positions = np.array([self._column.get(int(pid), -1) for pid in product_ids], dtype=int)
        known = positions >= 0
        result = {name: np.zeros(len(positions)) for name in CHANNELS}
        for name in CHANNELS:
            result[name][known] = values[name][positions[known]]
        return result

    def channels(self, user_id: int, top_k: int = 50) -> dict[str, list[ScoredProduct]]:
        """Return only positive-evidence, unowned, past-observed catalog candidates."""
        values = self._user_scores(user_id)
        result = {name: [] for name in CHANNELS}
        if top_k <= 0:
            return result
        for name, scores in values.items():
            candidates = np.flatnonzero(scores > 0)
            order = candidates[np.lexsort((self._ids[candidates], -scores[candidates]))][:top_k]
            result[name] = [ScoredProduct(int(self._ids[col]), float(scores[col])) for col in order]
        return result
