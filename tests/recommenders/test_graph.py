import json
import math
from datetime import date

import pandas as pd
import pytest

from clothes_recommender.catalog import ProductCatalog
from clothes_recommender.domain import ScoredProduct
from clothes_recommender.recommenders.base import UnknownUserError
from clothes_recommender.recommenders.graph import GraphRecommender
from clothes_recommender.recommenders.popularity import PopularityRecommender


def product(pid: int, **changes) -> dict:
    return {
        "product_id": pid,
        "name": f"Product {pid}",
        "product_type": "T-shirt",
        "product_group": "Garment Upper body",
        "section": "Ladieswear",
        "colour": "Black",
        "description": "",
        "price": 10.0,
        "shopify_category": "Uncategorized",
        "shopify_attributes": {},
    } | changes


@pytest.fixture
def make_graph(make_customers):
    def build(products, edges=(), purchases=(), users=(1, 2)):
        catalog = ProductCatalog(pd.DataFrame(products, columns=list(product(1))))
        customers = make_customers(users, [(u, p, date(2020, 9, 22)) for u, p in purchases])
        return GraphRecommender(
            customers,
            catalog,
            PopularityRecommender(customers.purchases),
            pd.DataFrame(edges, columns=["product_id", "other_id", "count", "lift"]),
        )

    return build


@pytest.fixture
def graph(make_graph):
    return make_graph(
        [
            product(1, shopify_category="Tops", shopify_attributes={"color": "Black"}),
            product(2, shopify_category="Tops", shopify_attributes={"color": "Black"}),
            product(3, shopify_category="Other", shopify_attributes={"color": "Black"}),
            product(4, product_group="Shoes"),
            product(5),
            product(6),
        ],
        edges=[(1, 4, 3, 2.0), (1, 3, 2, 1.0)],
        purchases=[(1, 1), (1, 1), (2, 5), (2, 5)],
        users=(1, 2, 3),
    )


def test_hand_computable_ranking_excludes_history_and_fills_with_popularity(graph) -> None:
    # Black has three products (1/2 each); Tops has two (1 each).
    # Repeated purchases of product 1 contribute once, and fallback always scores zero.
    assert graph.recommend(1) == [
        ScoredProduct(2, 1.5),
        ScoredProduct(3, 0.75),
        ScoredProduct(4, (2 / 3) * (3 / 5)),
        ScoredProduct(5, 0.0),
    ]
    assert graph.recommend(1, 2) == [ScoredProduct(2, 1.5), ScoredProduct(3, 0.75)]


def test_reasons_describe_only_actual_paths_or_popularity(graph) -> None:
    assert graph.reasons(1, 4) == [
        "Bought together with Product 1 (product 1) in 3 baskets (lift 2)."
    ]
    assert graph.reasons(1, 3) == [
        "Bought together with Product 1 (product 1) in 2 baskets (lift 1).",
        "Shares color=Black with purchased Product 1 (product 1).",
    ]
    assert graph.reasons(1, 2) == [
        "Shares color=Black with purchased Product 1 (product 1).",
        "Shares Shopify category Tops with purchased Product 1 (product 1).",
    ]
    assert graph.reasons(1, 5) == [
        "Popularity fallback from recent purchases; no graph evidence from history."
    ]
    assert graph.reasons(1, 1) == graph.reasons(1, 6) == graph.reasons(1, 999) == []


@pytest.mark.parametrize("co_seeds", [(), (2, 4), (1, 2, 3, 4)])
def test_reasons_cap_at_three_and_prioritize_co_purchases(make_graph, co_seeds) -> None:
    graph = make_graph(
        [
            product(i, shopify_category="Tops", shopify_attributes={"color": "Black"})
            for i in range(1, 6)
        ],
        edges=[(seed, 5, 2, seed) for seed in co_seeds],
        purchases=[(1, seed) for seed in range(1, 5)],
    )
    co_reasons = [
        f"Bought together with Product {seed} (product {seed}) in 2 baskets (lift {seed})."
        for seed in co_seeds
    ]
    metadata_reasons = [
        "Shares color=Black with purchased Product 1 (product 1).",
        "Shares Shopify category Tops with purchased Product 1 (product 1).",
        "Shares color=Black with purchased Product 2 (product 2).",
    ]
    assert graph.reasons(1, 5) == (co_reasons + metadata_reasons)[:3]
    assert graph.recommend(1) == [
        ScoredProduct(
            5, sum({1: 0.25, 2: 1 / 3, 3: 0.375, 4: 0.4}[seed] for seed in co_seeds) + 2.0
        )
    ]


def test_no_history_uses_popularity_without_inventing_preferences(graph) -> None:
    assert graph.recommend(3) == [ScoredProduct(1, 0.0), ScoredProduct(5, 0.0)]
    assert "Popularity fallback" in graph.reasons(3, 1)[0]


def test_out_of_catalog_history_and_popularity_are_ignored(make_graph) -> None:
    graph = make_graph([product(1), product(2)], purchases=[(1, 999), (2, 999), (2, 2)])
    assert graph.recommend(1) == [ScoredProduct(2, 0.0)]
    assert graph.reasons(1, 999) == []


def test_taxonomy_and_attributes_generate_candidates_without_co_purchase_edges(make_graph) -> None:
    graph = make_graph(
        [product(1, shopify_category="Tops"), product(2, shopify_category="Tops")],
        purchases=[(1, 1)],
    )
    assert graph.recommend(1) == [ScoredProduct(2, 1.0)]
    assert graph.related(1, "complement") == []


def test_shared_evidence_accumulates_from_distinct_history_products(make_graph) -> None:
    graph = make_graph(
        [product(i) for i in range(1, 4)],
        edges=[(1, 3, 2, 1.0), (2, 3, 4, 2.0)],
        purchases=[(1, 1), (1, 2)],
    )
    assert graph.recommend(1)[0].product_id == 3
    assert graph.recommend(1)[0].score == pytest.approx(0.25 + 4 / 9)
    assert len(graph.reasons(1, 3)) == 2


def test_repeated_seed_features_preserve_degree_weighted_scores(make_graph) -> None:
    graph = make_graph(
        [
            product(
                i,
                shopify_category="Tops" if i <= 3 else "Other",
                shopify_attributes={"color": "Black"},
            )
            for i in range(1, 5)
        ],
        edges=[(1, 3, 2, 1.5), (2, 3, 2, 2.5)],
        purchases=[(1, 1), (1, 2), (1, 1)],
    )
    ranked = graph.recommend(1)
    assert [p.product_id for p in ranked] == [3, 4]
    # Two distinct seeds: Black contributes 2/3, Tops 2/2, plus bounded edge evidence.
    assert [p.score for p in ranked] == pytest.approx([0.3 + 5 / 14 + 2 / 2 + 2 / 3, 2 / 3])


@pytest.mark.parametrize("size", [10, 20])
def test_common_attributes_are_degree_normalized_and_ties_use_product_id(make_graph, size) -> None:
    graph = make_graph(
        [
            product(
                i,
                shopify_category="Rare" if i <= 2 else "Uncategorized",
                shopify_attributes={"color": "Black"},
            )
            for i in reversed(range(1, size + 1))
        ],
        purchases=[(1, 1)],
    )
    ranked = graph.recommend(1, size)
    assert ranked[0] == ScoredProduct(2, 1 + 1 / (size - 1))
    assert [p.product_id for p in ranked[1:]] == list(range(3, size + 1))
    assert all(p.score == pytest.approx(1 / (size - 1)) for p in ranked[1:])


@pytest.mark.parametrize("top_k", [0, -1])
def test_nonpositive_limits_return_empty(graph, top_k) -> None:
    assert graph.recommend(1, top_k) == []
    assert graph.related(1, "complement", top_k) == []
    assert graph.related(1, "alternative", top_k) == []
    assert graph.neighborhood(1, top_k) == {"nodes": [], "edges": []}


def test_unknown_users_raise_including_when_limit_is_zero(graph) -> None:
    with pytest.raises(UnknownUserError):
        graph.recommend(999)
    with pytest.raises(UnknownUserError):
        graph.recommend(999, 0)
    with pytest.raises(UnknownUserError):
        graph.reasons(999, 1)


def test_complements_require_edges_and_different_broad_groups(make_graph) -> None:
    graph = make_graph(
        [
            product(1),
            product(2),
            product(3, product_group="Shoes"),
            product(4, product_group="Accessories"),
            product(5, product_group="Shoes"),
        ],
        edges=[(1, 2, 9, 9), (1, 3, 2, 2), (1, 4, 4, 2)],
    )
    assert graph.related(1, "complement") == [ScoredProduct(4, 4 / 9), ScoredProduct(3, 1 / 3)]
    assert graph.related(3, "complement") == [ScoredProduct(1, 1 / 3)]
    assert graph.related(1, "complement", 1) == [ScoredProduct(4, 4 / 9)]


def test_alternatives_are_same_specific_category_type_section_and_higher_priced(make_graph) -> None:
    products = [
        product(1),
        product(2, price=12),
        product(3, price=12),
        product(4, price=11, section="Menswear"),
        product(5, price=11, product_type="Blouse"),
        product(6, price=8),
        product(7, price=10),
        product(8, price=30),
        product(9, price=float("nan")),
        product(10, price=float("inf")),
        product(11, price=0),
    ]
    for row in products:
        row["shopify_category"] = "Apparel & Accessories > Clothing > Shirts & Tops"
    products.append(
        product(12, price=11, shopify_category="Apparel & Accessories > Clothing > Dresses")
    )
    graph = make_graph(products)

    assert graph.related(1, "alternative") == [
        ScoredProduct(2, 10 / 12),
        ScoredProduct(3, 10 / 12),
        ScoredProduct(8, 10 / 30),
    ]
    assert graph.related(1, "alternative", 1) == [ScoredProduct(2, 10 / 12)]
    assert graph.related(9, "alternative") == graph.related(11, "alternative") == []


@pytest.mark.parametrize(
    "category",
    [
        "Uncategorized",
        "",
        "  ",
        "Apparel & Accessories",
        "Clothing",
        "Clothing Accessories",
        "Apparel & Accessories > Clothing",
        "Apparel & Accessories > Clothing Accessories",
        "Apparel & Accessories > Clothing > Clothing Accessories",
        "Apparel & Accessories > > Shirts & Tops",
    ],
)
def test_other_accessories_without_specific_category_have_no_alternatives(make_graph, category):
    graph = make_graph(
        [
            product(
                pid,
                name=name,
                price=price,
                product_type="Other accessories",
                shopify_category=category,
            )
            for pid, name, price in [
                (344956023, "Waterbottle", 5),
                (2, "Necklace phone case", 10),
                (3, "Bag", 15),
            ]
        ]
    )
    assert graph.related(344956023, "alternative") == []


@pytest.mark.parametrize(
    "handle, known, conflicting",
    [
        ("target-gender", "Female", "Male"),
        ("age-group", "Adults", "Kids"),
    ],
)
@pytest.mark.parametrize("blank", [False, True])
def test_alternatives_skip_conflicting_known_audience_attributes(
    make_graph, handle, known, conflicting, blank
) -> None:
    unknown_attributes = {handle: " "} if blank else {}
    products = [
        product(1, shopify_attributes={handle: known}),
        product(2, price=11, shopify_attributes={handle: conflicting}),
        product(3, price=12, shopify_attributes={handle: known}),
        product(4, price=13, shopify_attributes=unknown_attributes),
        product(5, price=20, shopify_attributes={handle: conflicting}),
    ]
    for row in products:
        row["shopify_category"] = "Apparel & Accessories > Clothing > Shirts & Tops"
    graph = make_graph(products)

    assert graph.related(1, "alternative") == [ScoredProduct(3, 10 / 12), ScoredProduct(4, 10 / 13)]
    assert graph.related(1, "alternative", 1) == [ScoredProduct(3, 10 / 12)]
    assert graph.related(4, "alternative") == [ScoredProduct(5, 13 / 20)]


@pytest.mark.parametrize(
    "row",
    [
        (1, 1, 2, 2),
        (1, 999, 2, 2),
        (1.5, 3, 2, 2),
        (float("nan"), 3, 2, 2),
        (float("inf"), 3, 2, 2),
        (None, 3, 2, 2),
        ("bad", 3, 2, 2),
        (1, 3, 0, 2),
        (1, 3, -1, 2),
        (1, 3, 1.5, 2),
        (1, 3, float("nan"), 2),
        (1, 3, float("inf"), 2),
        (1, 3, "bad", 2),
        (1, 3, 2, 0),
        (1, 3, 2, -1),
        (1, 3, 2, float("nan")),
        (1, 3, 2, float("inf")),
        (1, 3, 2, "bad"),
    ],
)
def test_invalid_edges_are_filtered(make_graph, row) -> None:
    graph = make_graph(
        [product(1), product(2, product_group="Shoes"), product(3, product_group="Shoes")],
        edges=[(1, 2, 2, 1), row],
        purchases=[(1, 1)],
    )
    assert graph.recommend(1) == [ScoredProduct(2, 0.25)]
    assert graph.related(1, "complement") == [ScoredProduct(2, 0.25)]


def test_duplicate_and_reversed_edges_do_not_inflate_evidence(make_graph) -> None:
    graph = make_graph(
        [product(1), product(2)],
        edges=[(1, 2, 2, 1), (2, 1, 2, 1), (1, 2, 3, 2)],
        purchases=[(1, 1)],
    )
    assert graph.recommend(1)[0].product_id == 2
    assert graph.recommend(1)[0].score == pytest.approx(0.4)
    assert "3 baskets (lift 2)" in graph.reasons(1, 2)[0]


@pytest.mark.parametrize("top_k", [1, 2, 3, 4, 12])
def test_neighborhood_is_bounded_deterministic_and_has_no_missing_endpoints(graph, top_k) -> None:
    neighborhood = graph.neighborhood(1, top_k)
    assert neighborhood == graph.neighborhood(1, top_k)
    assert len(neighborhood["nodes"]) <= top_k
    ids = {n["id"] for n in neighborhood["nodes"]}
    assert len(ids) == len(neighborhood["nodes"])
    assert "product:1" in ids
    for node in neighborhood["nodes"]:
        assert set(node) == {"id", "label", "kind", "product_id"}
        assert node["id"].startswith(f"{node['kind']}:")
        assert node["product_id"] is None or isinstance(node["product_id"], int)
    for edge in neighborhood["edges"]:
        assert set(edge) == {"source", "target", "relation", "weight"}
        assert edge["source"] in ids and edge["target"] in ids
        assert isinstance(edge["weight"], float) and math.isfinite(edge["weight"])
    json.dumps(neighborhood, allow_nan=False)


def test_neighborhood_contains_taxonomy_attributes_and_real_co_purchase_edges(graph) -> None:
    neighborhood = graph.neighborhood(1)
    assert {n["kind"] for n in neighborhood["nodes"]} == {"product", "taxonomy", "attribute"}
    assert {n["product_id"] for n in neighborhood["nodes"] if n["kind"] == "product"} == {1, 3, 4}
    assert [e["weight"] for e in neighborhood["edges"] if e["relation"] == "co_purchased"] == [2, 1]


def test_catalog_is_iterated_once_at_construction_and_not_during_requests(make_graph, monkeypatch):
    original = ProductCatalog.__iter__
    calls = []

    def iterate(catalog):
        calls.append(1)
        assert len(calls) == 1
        return original(catalog)

    monkeypatch.setattr(ProductCatalog, "__iter__", iterate)
    graph = make_graph(
        [product(1, shopify_category="Tops"), product(2, shopify_category="Tops")],
        purchases=[(1, 1)],
    )
    graph.recommend(1)
    graph.reasons(1, 2)
    graph.related(1, "complement")
    graph.related(1, "alternative")
    graph.neighborhood(1)
    assert calls == [1]


def test_empty_catalog_and_purchase_history(make_graph) -> None:
    graph = make_graph([])
    assert graph.recommend(1) == []
    assert graph.reasons(1, 1) == []
    assert graph.related(1, "alternative") == []
    assert graph.neighborhood(1) == {"nodes": [], "edges": []}


def test_unknown_products_and_invalid_relation(graph) -> None:
    assert graph.related(999, "complement") == []
    assert graph.related(999, "alternative") == []
    assert graph.neighborhood(999) == {"nodes": [], "edges": []}
    with pytest.raises(ValueError, match="Unknown related-product kind"):
        graph.related(1, "quality")


def test_nonempty_co_purchase_table_requires_expected_columns(make_customers) -> None:
    customers = make_customers([1], [])
    catalog = ProductCatalog(pd.DataFrame([product(1)]))
    with pytest.raises(ValueError, match="Co-purchases must contain"):
        GraphRecommender(
            customers, catalog, PopularityRecommender(customers.purchases), pd.DataFrame({"x": [1]})
        )


def test_rare_pair_lift_is_bounded_and_support_matters(make_graph):
    graph = make_graph(
        [product(1), product(2), product(3), product(4)],
        edges=[(1, 2, 2, 146), (1, 3, 100, 146), (1, 4, 2, 2)],
        purchases=[(1, 1)],
    )
    hits = graph.recommend(1)
    assert [hit.product_id for hit in hits] == [3, 2, 4]
    assert all(0 < hit.score < 1 for hit in hits)
    assert hits[1].score == pytest.approx(146 / 147 * 0.5)
    assert "lift 146" in graph.reasons(1, 2)[0]
