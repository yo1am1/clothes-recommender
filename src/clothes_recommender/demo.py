"""Choose real, varied customer examples without changing the customer dataset.

An example must be a customer whose recommendations can be judged at a glance:
- a coherent history: 6-30 distinct products, at least 80% of them from one section;
- personal evidence in every strategy: a similar shopper for collaborative filtering and a
  purchased product with co-purchased complements for the knowledge graph;
- a hybrid list that stays in the sections the customer buys from and covers their main
  product group.

The checks only read the recommenders' output; nothing is tuned for the examples, and every
other customer stays available. The Evaluation tab scores all eligible customers instead.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from clothes_recommender.catalog import CustomerData, ProductCatalog
from clothes_recommender.domain import Product
from clothes_recommender.recommenders.base import Recommender
from clothes_recommender.recommenders.collaborative import CollaborativeRecommender
from clothes_recommender.recommenders.graph import GraphRecommender

MIN_DISTINCT_PRODUCTS = 6
MAX_DISTINCT_PRODUCTS = 30
MIN_SECTION_SHARE = 0.8
MIN_NEIGHBOUR_SIMILARITY = 0.1
CHECKED_RECOMMENDATIONS = 10


@dataclass
class DemoCustomer:
    user_id: int
    label: str
    description: str
    focus_product_id: int


@dataclass(frozen=True)
class _Candidate:
    user_id: int
    section: str
    main_group: str
    history: list[Product]  # distinct products, most recent purchase first
    records: int


def select_demo_customers(
    customers: CustomerData,
    catalog: ProductCatalog,
    graph: GraphRecommender,
    collaborative: CollaborativeRecommender,
    hybrid: Recommender,
    limit: int = 10,
) -> list[DemoCustomer]:
    """Up to `limit` examples, round-robin over sections so each audience is represented."""
    if limit <= 0:
        return []
    by_section: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in _coherent_customers(customers, catalog):
        by_section[candidate.section].append(candidate)
    # Most-populated sections first, so their second examples come before thin sections run dry.
    sections = sorted(by_section, key=lambda s: (-len(by_section[s]), s))

    selected: list[DemoCustomer] = []
    used_groups: dict[str, set[str]] = defaultdict(set)
    progress = True
    while len(selected) < limit and progress:
        progress = False
        for section in sections:
            if len(selected) == limit:
                break
            # Prefer a main product group this section has not shown yet.
            queue = sorted(
                by_section[section],
                key=lambda c: (c.main_group in used_groups[section], -len(c.history), c.user_id),
            )
            for candidate in queue:
                by_section[section].remove(candidate)
                example = _check(candidate, graph, collaborative, hybrid, catalog)
                if example is not None:
                    selected.append(example)
                    used_groups[section].add(candidate.main_group)
                    progress = True
                    break
    return selected


def _coherent_customers(customers: CustomerData, catalog: ProductCatalog) -> list[_Candidate]:
    candidates = []
    for user_id in customers.user_ids():
        purchases = customers.purchases_for(user_id)
        history = list(dict.fromkeys(p.product_id for p in purchases))
        if not MIN_DISTINCT_PRODUCTS <= len(history) <= MAX_DISTINCT_PRODUCTS:
            continue
        products = catalog.get_many(history)
        if len(products) != len(history):
            continue
        section, count = Counter(p.section for p in products).most_common(1)[0]
        if count / len(products) < MIN_SECTION_SHARE:
            continue
        candidates.append(
            _Candidate(
                user_id=user_id,
                section=section,
                main_group=_ranked(p.product_group for p in products)[0],
                history=products,
                records=len(purchases),
            )
        )
    return candidates


def _check(
    candidate: _Candidate,
    graph: GraphRecommender,
    collaborative: CollaborativeRecommender,
    hybrid: Recommender,
    catalog: ProductCatalog,
) -> DemoCustomer | None:
    neighbours = collaborative.similar_users(candidate.user_id, top_k=1)
    if not neighbours or neighbours[0].similarity < MIN_NEIGHBOUR_SIMILARITY:
        return None
    focus = next(
        (p for p in candidate.history if graph.related(p.product_id, "complement", top_k=1)),
        None,
    )
    if focus is None:
        return None
    recommended = catalog.get_many(
        item.product_id for item in hybrid.recommend(candidate.user_id, CHECKED_RECOMMENDATIONS)
    )
    sections = {p.section for p in candidate.history}
    if (
        len(recommended) < CHECKED_RECOMMENDATIONS
        or any(p.section not in sections for p in recommended)
        or all(p.product_group != candidate.main_group for p in recommended)
    ):
        return None

    types = _ranked(p.product_type for p in candidate.history)[:3]
    return DemoCustomer(
        user_id=candidate.user_id,
        label=f"{candidate.section}: {', '.join(types)}",
        description=(
            f"{candidate.records} purchases of {len(candidate.history)} products, mostly "
            f"{candidate.section}. Most similar shopper: {neighbours[0].similarity:.2f} "
            f"cosine similarity. {focus.name} has co-purchased complements to explore."
        ),
        focus_product_id=focus.product_id,
    )


def _ranked(values: Iterable[str]) -> list[str]:
    """Distinct values, most frequent first; ties alphabetical so results are deterministic."""
    counts = Counter(values)
    return sorted(counts, key=lambda value: (-counts[value], value))
