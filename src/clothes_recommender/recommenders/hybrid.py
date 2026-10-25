"""Hybrid recommendations: reciprocal rank fusion of several recommenders' lists."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

from clothes_recommender.domain import ScoredProduct
from clothes_recommender.recommenders.base import Recommender

# The constant from the original RRF paper (Cormack et al., 2009). It damps the gap between
# the first few ranks, so one recommender's top pick cannot outvote agreement between lists.
RRF_K = 60
# How deep each component list is read; products below this rank contribute nothing.
CANDIDATES_PER_RECOMMENDER = 50


@dataclass(frozen=True)
class FusedProduct:
    product_id: int
    score: float
    # Component name -> 1-based rank in that component's list.
    ranks: dict[str, int]

    def reasons(self) -> list[str]:
        return [f"Ranked #{rank} by {name}." for name, rank in self.ranks.items()]


class HybridRecommender:
    """Sums 1 / (RRF_K + rank) over the components' lists.

    Fusing ranks rather than scores needs no calibration between recommenders whose
    scores live on unrelated scales (cosine similarity, neighbour votes, graph lift).
    Equal scores are common (one list's #5 against another's #5), so ties go to the
    product ranked higher by the component listed first: list the strongest one first.
    """

    def __init__(self, components: Mapping[str, Recommender]) -> None:
        if not components:
            raise ValueError("A hybrid needs at least one component recommender")
        self._components = dict(components)

    def recommend(self, user_id: int, top_k: int = 10) -> list[ScoredProduct]:
        return [ScoredProduct(f.product_id, f.score) for f in self.explain(user_id, top_k)]

    def explain(self, user_id: int, top_k: int = 10) -> list[FusedProduct]:
        """Fused ranking with each product's rank in every component that listed it."""
        ranks: dict[int, dict[str, int]] = {}
        for name, recommender in self._components.items():
            ranked = recommender.recommend(user_id, max(top_k, CANDIDATES_PER_RECOMMENDER))
            for rank, item in enumerate(ranked, 1):
                ranks.setdefault(item.product_id, {}).setdefault(name, rank)
        fused = [
            FusedProduct(pid, sum(1 / (RRF_K + rank) for rank in by_name.values()), by_name)
            for pid, by_name in ranks.items()
        ]
        # Distinct products never share a rank within one list, so this order is total.
        fused.sort(
            key=lambda f: (-f.score, [f.ranks.get(name, math.inf) for name in self._components])
        )
        return fused[:top_k]
