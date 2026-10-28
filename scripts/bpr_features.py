"""Offline, past-only CPU BPR features; raw dot products are not probabilities.

Only observed catalog items and shoppers enter the sorted binary CSR. Personal
users map through the explicit shopper crosswalk, never through equal integer IDs.
Scores include implicit's native item-bias coordinate and preserve owned items.
"""

from collections.abc import Iterable
from datetime import date

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

FEATURE_NAMES = ("bpr_score", "bpr_candidate_reciprocal_rank", "bpr_known_vector")


def _require(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


class BPRFeatures:
    """Fit the fixed experimental BPR once; score only supplied candidate products.

    Each duplicate shopper/item event is one binary observation, irrespective of
    channel, price or frequency. Candidate ranks use unique known items, descending
    score then ascending product ID. Missing vectors get three zeros. Empty/tiny
    matrices or matrices without any unobserved alternative are not fitted.
    """

    def __init__(
        self, products: pd.DataFrame, users: pd.DataFrame, crowd: pd.DataFrame, as_of: date
    ) -> None:
        _require(products, ("product_id",), "products")
        _require(users, ("user_id", "shopper_id"), "users")
        _require(crowd, ("shopper_id", "product_id", "purchase_date"), "crowd")
        if products["product_id"].isna().any() or products["product_id"].duplicated().any():
            raise ValueError("Products need unique nonmissing product_id values")
        if (
            users[["user_id", "shopper_id"]].isna().any().any()
            or users["user_id"].duplicated().any()
            or users["shopper_id"].duplicated().any()
        ):
            raise ValueError("User/shopper crosswalk must be unique and nonmissing in both columns")
        try:
            from implicit.cpu.bpr import BayesianPersonalizedRanking
        except ImportError as error:
            raise ValueError(
                "BPRFeatures needs optional CPU implicit and its dependencies"
            ) from error

        self._catalog = set(products["product_id"])
        self._crosswalk = dict(zip(users["user_id"], users["shopper_id"], strict=True))
        self._user_factors = self._item_factors = None
        try:
            cutoff = pd.Timestamp(as_of).normalize()
            dates = pd.to_datetime(crowd["purchase_date"]).dt.normalize()
            if pd.isna(cutoff) or dates.isna().any():
                raise ValueError("missing date")
            past = crowd.loc[dates <= cutoff]
        except (ValueError, TypeError) as error:
            raise ValueError("BPR needs valid timezone-naive purchase dates and as_of") from error
        if past[["shopper_id", "product_id"]].isna().any().any():
            raise ValueError("Past crowd events need shopper_id and product_id")
        past = past[past["product_id"].isin(self._catalog)]
        shoppers = np.sort(past["shopper_id"].unique())
        items = np.sort(past["product_id"].unique())
        self._rows = {shopper: row for row, shopper in enumerate(shoppers)}
        self._columns = {pid: column for column, pid in enumerate(items)}
        matrix = csr_matrix(
            (
                np.ones(len(past), dtype=np.float32),
                (past["shopper_id"].map(self._rows), past["product_id"].map(self._columns)),
            ),
            shape=(len(shoppers), len(items)),
            dtype=np.float32,
        )
        matrix.sum_duplicates()
        matrix.data[:] = 1
        matrix.sort_indices()
        if (
            not self._crosswalk
            or min(matrix.shape) < 2
            or matrix.nnz < 2
            or not (np.diff(matrix.indptr) < matrix.shape[1]).any()
        ):
            return
        model = BayesianPersonalizedRanking(
            factors=64,
            iterations=50,
            learning_rate=0.05,
            regularization=0.01,
            random_state=42,
            num_threads=1,
            verify_negative_samples=True,
        )
        model.fit(matrix, show_progress=False)
        uf = np.asarray(model.user_factors)
        itf = np.asarray(model.item_factors)
        if (
            uf.ndim != 2
            or itf.ndim != 2
            or uf.shape[0] != len(shoppers)
            or itf.shape[0] != len(items)
            or uf.shape[1] == 0
            or uf.shape[1] != itf.shape[1]
            or uf.dtype.kind not in "fiu"
            or itf.dtype.kind not in "fiu"
            or not np.isfinite(uf).all()
            or not np.isfinite(itf).all()
        ):
            raise ValueError("BPR factor arrays must have aligned finite user/item dimensions")
        self._user_factors, self._item_factors = uf, itf

    def features(self, user_id: int, product_ids: Iterable[int]) -> np.ndarray:
        """Finite float64 (n,3), preserving duplicate rows and repeat candidates."""
        ids = list(product_ids)
        if any(pid not in self._catalog for pid in ids):
            raise ValueError("Candidate product IDs must belong to the supplied catalog")
        result = np.zeros((len(ids), len(FEATURE_NAMES)), dtype=np.float64)
        row = self._rows.get(self._crosswalk.get(user_id))
        known = sorted(set(ids) & self._columns.keys())
        if row is None or self._user_factors is None or not known:
            return result
        user = self._user_factors[row].astype(np.float64)
        factors = self._item_factors[[self._columns[pid] for pid in known]].astype(np.float64)
        scores = factors @ user
        if not np.isfinite(scores).all():
            raise ValueError("BPR scores must be finite")
        order = np.lexsort((known, -scores))
        values = {known[index]: (scores[index], 1 / rank, 1) for rank, index in enumerate(order, 1)}
        for i, pid in enumerate(ids):
            if pid in values:
                result[i] = values[pid]
        return result
