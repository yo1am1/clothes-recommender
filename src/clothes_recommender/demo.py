"""Choose real, varied customer examples without changing the customer dataset."""

import math
from dataclasses import dataclass

from clothes_recommender.catalog import CustomerData, ProductCatalog
from clothes_recommender.recommenders.graph import GraphRecommender


@dataclass
class DemoCustomer:
    user_id: int
    label: str
    description: str
    focus_product_id: int


def select_demo_customers(
    customers: CustomerData,
    catalog: ProductCatalog,
    graph: GraphRecommender,
    limit: int = 10,
) -> list[DemoCustomer]:
    """Prefer supported, varied evidence, then novel groups/types/products and metadata."""
    if limit <= 0:
        return []
    products = {
        p.product_id: p
        for p in catalog
        if p.name.strip() and p.product_type.strip() and math.isfinite(p.price) and p.price > 0
    }
    evidence: dict[int, tuple[int, int, str]] = {}
    histories: dict[int, tuple[int, int]] = {}
    candidates: list[tuple[int, int]] = []
    for user_id in customers.user_ids():
        history = customers.purchases_for(user_id)
        purchased = {purchase.product_id for purchase in history}
        histories[user_id] = (len(history), len(purchased))
        for pid in sorted(purchased & products.keys()):
            if pid not in evidence:
                product = products[pid]
                richness = sum(bool(value.strip()) for value in product.shopify_attributes.values())
                richness += bool(
                    product.shopify_category.strip()
                    and product.shopify_category.strip() != "Uncategorized"
                )
                if graph.related(pid, "complement", top_k=1):
                    strength, detail = 3, "co-purchased complements"
                elif any(
                    edge["relation"] == "co_purchased"
                    for edge in graph.neighborhood(pid, top_k=len(product.shopify_attributes) + 3)[
                        "edges"
                    ]
                ):
                    strength, detail = 2, "co-purchase links"
                elif graph.related(pid, "alternative", top_k=1):
                    strength, detail = 1, "higher-priced alternatives"
                else:
                    strength = 0
                    detail = "product metadata" if richness else "purchase history"
                evidence[pid] = (strength, richness, detail)
            candidates.append((user_id, pid))

    selected: list[DemoCustomer] = []
    seen_evidence: set[str] = set()
    seen_products: set[int] = set()
    seen_types: set[str] = set()
    seen_groups: set[str] = set()
    while candidates and len(selected) < limit:
        user_id, pid = max(
            candidates,
            key=lambda pair: (
                evidence[pair[1]][0] > 0,
                evidence[pair[1]][2] not in seen_evidence,
                evidence[pair[1]][0],
                products[pair[1]].product_group not in seen_groups,
                products[pair[1]].product_type not in seen_types,
                pair[1] not in seen_products,
                evidence[pair[1]][1],
                -pair[0],
                -pair[1],
            ),
        )
        product = products[pid]
        detail = evidence[pid][2]
        records, distinct = histories[user_id]
        selected.append(
            DemoCustomer(
                user_id=user_id,
                label=f"{product.product_type}: {detail}",
                description=(
                    f"Inspect {detail} for {product.name}. "
                    f"Purchase records: {records}; distinct products: {distinct}."
                ),
                focus_product_id=pid,
            )
        )
        seen_evidence.add(detail)
        seen_products.add(pid)
        seen_types.add(product.product_type)
        seen_groups.add(product.product_group)
        candidates = [pair for pair in candidates if pair[0] != user_id]
    return selected
