"""A small, inspectable graph over co-purchases and Shopify product metadata."""

import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from typing import Literal

import pandas as pd

from clothes_recommender.catalog import CustomerData, ProductCatalog
from clothes_recommender.domain import ScoredProduct
from clothes_recommender.recommenders.base import UnknownUserError
from clothes_recommender.recommenders.popularity import PopularityRecommender

# (node kind, attribute handle, value); taxonomy uses the full category path.
Feature = tuple[str, str, str]


def _alternative_category(category: str) -> str | None:
    """Require a product-class path below broad taxonomy departments."""
    parts = [part.strip() for part in category.split(">")]
    if (
        len(parts) < 3
        or not all(parts)
        or parts[-1].casefold()
        in {"uncategorized", "apparel & accessories", "clothing", "clothing accessories"}
    ):
        return None
    return " > ".join(parts)


def _co_purchase_weight(count: int, lift: float) -> float:
    """Bound association strength and shrink links supported by few baskets."""
    return (lift / (1 + lift)) * (count / (count + 2))


class GraphRecommender:
    """Combine bounded co-purchase evidence with degree-weighted shared features.

    Each shared feature contributes 1 / (degree - 1), distributing one unit of a
    seed product's evidence across its neighbors. Popularity fills remaining slots
    with score zero, below every graph candidate. Prices are approximate, not quality.
    """

    def __init__(
        self,
        customers: CustomerData,
        catalog: ProductCatalog,
        popular: PopularityRecommender,
        co_purchases: pd.DataFrame,
    ) -> None:
        self._customers = customers
        self._products = {p.product_id: p for p in list(catalog)}
        self._features: dict[int, tuple[Feature, ...]] = {}
        self._index: dict[Feature, list[int]] = defaultdict(list)
        self._alternatives: dict[tuple[str, str, str], list[tuple[float, int]]] = defaultdict(list)
        for pid, product in sorted(self._products.items()):
            features = [
                ("attribute", handle, value)
                for handle, value in sorted(product.shopify_attributes.items())
                if handle.strip() and value.strip()
            ]
            if product.shopify_category.strip() and product.shopify_category != "Uncategorized":
                features.append(("taxonomy", "category", product.shopify_category))
            self._features[pid] = tuple(sorted(features))
            for feature in self._features[pid]:
                self._index[feature].append(pid)
            category = _alternative_category(product.shopify_category)
            if category is not None and math.isfinite(product.price) and product.price > 0:
                self._alternatives[product.product_type, product.section, category].append(
                    (product.price, pid)
                )
        for prices in self._alternatives.values():
            prices.sort()

        self._adjacency: dict[int, dict[int, tuple[int, float]]] = defaultdict(dict)
        columns = ["product_id", "other_id", "count", "lift"]
        if not co_purchases.empty:
            if not set(columns).issubset(co_purchases.columns):
                raise ValueError("Co-purchases must contain product_id, other_id, count and lift")
            for row in co_purchases[columns].itertuples(index=False, name=None):
                try:
                    left, right, count, lift = (float(value) for value in row)
                except (TypeError, ValueError, OverflowError):
                    continue
                if (
                    not all(math.isfinite(value) for value in (left, right, count, lift))
                    or not all(value.is_integer() for value in (left, right, count))
                    or left == right
                    or left not in self._products
                    or right not in self._products
                    or count < 1
                    or lift <= 0
                ):
                    continue
                a, b = int(left), int(right)
                # Reversed/duplicate rows represent the same edge; never double-count them.
                previous = self._adjacency[a].get(b)
                if previous is None or (lift, count) > (previous[1], previous[0]):
                    self._adjacency[a][b] = self._adjacency[b][a] = (int(count), lift)

        outside_catalog = set(customers.purchases["product_id"]) - self._products.keys()
        self._fallback = popular.top(len(self._products), exclude_ids=outside_catalog)
        self._fallback_ids = {p.product_id for p in self._fallback}

    def _history(self, user_id: int) -> set[int]:
        if self._customers.get_user(user_id) is None:
            raise UnknownUserError(user_id)
        return {p.product_id for p in self._customers.purchases_for(user_id)}

    def _scores(self, history: set[int]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        feature_counts: Counter[Feature] = Counter()
        for seed in sorted(history & self._products.keys()):
            for pid, (count, lift) in sorted(self._adjacency.get(seed, {}).items()):
                scores[pid] += _co_purchase_weight(count, lift)
            feature_counts.update(self._features[seed])
        for feature, count in sorted(feature_counts.items()):
            neighbors = self._index[feature]
            if len(neighbors) > 1:
                contribution = count / (len(neighbors) - 1)
                for pid in neighbors:
                    scores[pid] += contribution
        for pid in history:
            scores.pop(pid, None)
        return scores

    @staticmethod
    def _rank(scores: dict[int, float], top_k: int) -> list[ScoredProduct]:
        return [
            ScoredProduct(pid, score)
            for pid, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        ]

    def recommend(self, user_id: int, top_k: int = 10) -> list[ScoredProduct]:
        history = self._history(user_id)
        if top_k <= 0:
            return []
        ranked = self._rank(self._scores(history), top_k)
        excluded = history | {p.product_id for p in ranked}
        for item in self._fallback:
            if len(ranked) >= top_k:
                break
            if item.product_id not in excluded:
                ranked.append(ScoredProduct(item.product_id, 0.0))
        return ranked

    def reasons(self, user_id: int, product_id: int) -> list[str]:
        history = self._history(user_id)
        if product_id not in self._products or product_id in history:
            return []
        reasons = []
        seeds = sorted(history & self._products.keys())
        for seed in seeds:
            name = f"{self._products[seed].name} (product {seed})"
            if edge := self._adjacency.get(seed, {}).get(product_id):
                count, lift = edge
                reasons.append(f"Bought together with {name} in {count} baskets (lift {lift:g}).")
                if len(reasons) == 3:
                    return reasons
        for seed in seeds:
            name = f"{self._products[seed].name} (product {seed})"
            shared = set(self._features[seed]) & set(self._features[product_id])
            for kind, handle, value in sorted(shared):
                label = f"Shopify category {value}" if kind == "taxonomy" else f"{handle}={value}"
                reasons.append(f"Shares {label} with purchased {name}.")
                if len(reasons) == 3:
                    return reasons
        if not reasons and product_id in self._fallback_ids:
            reasons.append(
                "Popularity fallback from recent purchases; no graph evidence from history."
            )
        return reasons

    def related(
        self,
        product_id: int,
        kind: Literal["complement", "alternative"],
        top_k: int = 10,
    ) -> list[ScoredProduct]:
        if kind not in ("complement", "alternative"):
            raise ValueError(f"Unknown related-product kind: {kind}")
        product = self._products.get(product_id)
        if top_k <= 0 or product is None:
            return []
        if kind == "complement":
            return self._rank(
                {
                    pid: _co_purchase_weight(count, lift)
                    for pid, (count, lift) in self._adjacency.get(product_id, {}).items()
                    if self._products[pid].product_group != product.product_group
                },
                top_k,
            )
        category = _alternative_category(product.shopify_category)
        if category is None or not math.isfinite(product.price) or product.price <= 0:
            return []
        prices = self._alternatives[product.product_type, product.section, category]
        start = bisect_right(prices, (product.price, math.inf))
        ranked = []
        for index in range(start, len(prices)):
            price, pid = prices[index]
            candidate = self._products[pid]
            if any(
                (expected := product.shopify_attributes.get(handle, "").strip())
                and (actual := candidate.shopify_attributes.get(handle, "").strip())
                and expected != actual
                for handle in ("target-gender", "age-group")
            ):
                continue
            ranked.append(ScoredProduct(pid, product.price / price))
            if len(ranked) == top_k:
                break
        return ranked

    def neighborhood(self, product_id: int, top_k: int = 12) -> dict:
        """A deterministic star, capped at top_k total nodes including the focus.

        Focus metadata comes first, then co-purchased products ranked by lift.
        """
        product = self._products.get(product_id)
        if top_k <= 0 or product is None:
            return {"nodes": [], "edges": []}
        focus = f"product:{product_id}"
        nodes = [{"id": focus, "label": product.name, "kind": "product", "product_id": product_id}]
        edges = []
        for kind, handle, value in self._features[product_id]:
            if len(nodes) >= top_k:
                break
            node_id = f"{kind}:{json.dumps([handle, value], ensure_ascii=False)}"
            nodes.append(
                {
                    "id": node_id,
                    "label": value if kind == "taxonomy" else f"{handle}: {value}",
                    "kind": kind,
                    "product_id": None,
                }
            )
            edges.append(
                {
                    "source": focus,
                    "target": node_id,
                    "relation": "in_category" if kind == "taxonomy" else "has_attribute",
                    "weight": 1.0 / max(1, len(self._index[kind, handle, value]) - 1),
                }
            )
        neighbors = sorted(
            self._adjacency.get(product_id, {}).items(), key=lambda item: (-item[1][1], item[0])
        )
        for pid, (_, lift) in neighbors:
            if len(nodes) >= top_k:
                break
            node_id = f"product:{pid}"
            nodes.append(
                {
                    "id": node_id,
                    "label": self._products[pid].name,
                    "kind": "product",
                    "product_id": pid,
                }
            )
            edges.append(
                {"source": focus, "target": node_id, "relation": "co_purchased", "weight": lift}
            )
        return {"nodes": nodes, "edges": edges}
