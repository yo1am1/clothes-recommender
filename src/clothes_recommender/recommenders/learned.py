"""Fixed-feature linear ranking, fitted offline from earlier purchase windows."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator
from scipy.optimize import minimize
from scipy.special import expit

from clothes_recommender.catalog import CustomerData, ProductCatalog
from clothes_recommender.domain import ScoredProduct
from clothes_recommender.recommenders.base import UnknownUserError
from clothes_recommender.recommenders.collaborative import CollaborativeRecommender
from clothes_recommender.recommenders.popularity import PopularityRecommender
from clothes_recommender.recommenders.repeat import RepeatPurchaseRecommender

FEATURES = (
    "bias",
    "repeat",
    "frequency",
    "recency_7d",
    "recency_30d",
    "collaborative",
    "collaborative_rank",
    "popularity",
    "type_affinity",
    "section_affinity",
)


PROVENANCE_FEATURES = (
    *FEATURES,
    "source_repeat",
    "source_repeat_rank",
    "source_collaborative",
    "source_popular",
    "source_popular_rank",
)


def feature_names(version: int) -> tuple[str, ...]:
    if version == 1:
        return FEATURES
    if version == 2:
        return PROVENANCE_FEATURES
    raise ValueError("Unsupported ranker feature version")


class RankerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feature_version: Literal[1, 2] = 1
    weights: list[FiniteFloat] = Field(
        min_length=len(FEATURES), max_length=len(PROVENANCE_FEATURES)
    )
    trained_through: date
    training_window_ends: list[date] = Field(min_length=3, max_length=3)
    training_rows: int = Field(gt=0)
    training_positives: int = Field(gt=0)

    @model_validator(mode="after")
    def training_metadata(self) -> RankerModel:
        if len(self.weights) != len(feature_names(self.feature_version)):
            raise ValueError("Ranker weights must match its feature version")
        if (
            self.training_positives >= self.training_rows
            or self.training_window_ends != sorted(set(self.training_window_ends))
            or self.training_window_ends[-1] != self.trained_through
        ):
            raise ValueError("Ranker needs ordered past training windows and both label classes")
        return self


def load_model(path: Path) -> RankerModel:
    with path.open("rb") as file:
        content = file.read(64 * 1024 + 1)
    if len(content) > 64 * 1024:
        raise ValueError("Ranker model exceeds 64 KiB")
    return RankerModel.model_validate_json(content)


class LearnedRecommender:
    """Linear relevance ranker over fixed, past-only repeat/collaborative/popular candidates."""

    def __init__(
        self,
        customers: CustomerData,
        catalog: ProductCatalog,
        collaborative: CollaborativeRecommender,
        popular: PopularityRecommender,
        weights: list[float] | np.ndarray,
        as_of: date,
        *,
        feature_version: int = 1,
    ) -> None:
        self._customers = customers
        self._products = {item.product_id: item for item in catalog}
        self._collaborative = collaborative
        self._popular = popular
        self._features = feature_names(feature_version)
        self._weights = np.asarray(weights, dtype=float)
        if self._weights.shape != (len(self._features),) or not np.isfinite(self._weights).all():
            raise ValueError("Ranker weights must match the finite feature vector")
        self._repeat = RepeatPurchaseRecommender(customers, popular)
        self._as_of = as_of
        self._pop_scores = {item.product_id: item.score for item in popular.top(len(catalog))}
        self._pop_ranks = {pid: rank for rank, pid in enumerate(list(self._pop_scores)[:50], 1)}
        self._pop_scale = math.log1p(max(self._pop_scores.values(), default=0)) or 1

    def candidates(
        self, user_id: int, *, extra_ids: Iterable[int] = ()
    ) -> tuple[list[int], np.ndarray]:
        """Fixed pool plus explicitly supplied experiment candidates; serving supplies none."""
        if self._customers.get_user(user_id) is None:
            raise UnknownUserError(user_id)
        history = self._customers.purchases_for(user_id)
        counts = Counter(item.product_id for item in history)
        latest = {}
        for item in history:
            latest.setdefault(item.product_id, item.purchase_date)
        repeat_ranks = {pid: rank for rank, pid in enumerate(list(latest)[:100], 1)}
        cf = self._collaborative.recommend(user_id, 50)
        cf_scores = {item.product_id: item.score for item in cf}
        cf_ranks = {item.product_id: rank for rank, item in enumerate(cf, 1)}
        cf_scale = math.log1p(max(cf_scores.values(), default=0)) or 1
        ids = list(
            dict.fromkeys(
                [
                    *repeat_ranks,
                    *(item.product_id for item in cf),
                    *self._pop_ranks,
                    *extra_ids,
                ]
            )
        )
        if any(pid not in self._products for pid in ids):
            raise ValueError("Candidates must belong to the prepared catalog")
        types = Counter(
            self._products[item.product_id].product_type
            for item in history
            if item.product_id in self._products
        )
        sections = Counter(
            self._products[item.product_id].section
            for item in history
            if item.product_id in self._products
        )
        total = len(history) or 1
        rows = []
        for pid in ids:
            product = self._products[pid]
            age = max(0, (self._as_of - latest[pid]).days) if pid in latest else None
            rows.append(
                [
                    1,
                    float(pid in latest),
                    math.log1p(counts[pid]),
                    math.exp(-age / 7) if age is not None else 0,
                    math.exp(-age / 30) if age is not None else 0,
                    math.log1p(cf_scores.get(pid, 0)) / cf_scale,
                    1 / cf_ranks[pid] if pid in cf_ranks else 0,
                    math.log1p(self._pop_scores.get(pid, 0)) / self._pop_scale,
                    types[product.product_type] / total,
                    sections[product.section] / total,
                ]
            )
            if self._features == PROVENANCE_FEATURES:
                rows[-1].extend(
                    [
                        float(pid in repeat_ranks),
                        1 / repeat_ranks[pid] if pid in repeat_ranks else 0,
                        float(pid in cf_ranks),
                        float(pid in self._pop_ranks),
                        1 / self._pop_ranks[pid] if pid in self._pop_ranks else 0,
                    ]
                )
        return ids, np.asarray(rows, dtype=float).reshape(-1, len(self._features))

    def recommend(self, user_id: int, top_k: int = 10) -> list[ScoredProduct]:
        ids, features = self.candidates(user_id)
        if top_k <= 0:
            return []
        scores = features @ self._weights
        order = np.lexsort((ids, -scores))[:top_k]
        return [ScoredProduct(ids[index], float(scores[index])) for index in order]

    def reasons(self, user_id: int, product_id: int) -> list[str]:
        if any(item.product_id == product_id for item in self._customers.purchases_for(user_id)):
            reasons = self._repeat.reasons(user_id, product_id)
        else:
            reasons = self._collaborative.reasons(user_id, product_id)
        return [
            *reasons,
            "Learned ranking uses purchase recency, frequency, co-purchases, "
            "popularity and catalog affinities. Scores are ranking signals, not probabilities.",
        ]


def fit(
    features: np.ndarray, labels: np.ndarray | list[int], *, feature_version: int = 1
) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if (
        labels.ndim != 1
        or features.ndim != 2
        or features.shape != (len(labels), len(feature_names(feature_version)))
        or not len(labels)
        or not np.isfinite(features).all()
        or not np.isin(labels, [0, 1]).all()
        or not 0 < labels.mean() < 1
    ):
        raise ValueError("Training needs finite feature rows and both binary label classes")
    penalty = 1e-4

    def loss(weights):
        scores = features @ weights
        regularized = weights.copy()
        regularized[0] = 0
        value = (
            np.mean(np.logaddexp(0, scores) - labels * scores)
            + penalty * np.dot(regularized, regularized) / 2
        )
        gradient = features.T @ (expit(scores) - labels) / len(labels) + penalty * regularized
        return value, gradient

    initial = np.zeros(features.shape[1])
    initial[0] = math.log(labels.mean() / (1 - labels.mean()))
    result = minimize(
        loss,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 200, "gtol": 1e-7, "ftol": 1e-10},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"Ranker did not converge: {result.message}")
    return result.x
