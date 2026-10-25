from collections import Counter
from dataclasses import asdict

import pandas as pd
import pytest

from clothes_recommender.catalog import CustomerData, ProductCatalog
from clothes_recommender.demo import DemoCustomer, select_demo_customers
from clothes_recommender.domain import Product
from clothes_recommender.recommenders.graph import GraphRecommender
from clothes_recommender.recommenders.popularity import PopularityRecommender


def product(pid: int, **changes) -> dict:
    return Product.model_validate(
        {
            "product_id": pid,
            "name": f"Product {pid}",
            "product_type": "T-shirt",
            "product_group": "Garment Upper body",
            "section": "Ladieswear",
            "colour": "Black",
            "description": "",
            "price": 10,
        }
        | changes
    ).model_dump()


@pytest.fixture
def make_data():
    def build(products, purchases=(), users=(), edges=()):
        catalog = ProductCatalog(pd.DataFrame(products, columns=list(product(1))))
        customers = CustomerData(
            pd.DataFrame({"user_id": list(users), "age": 30}),
            pd.DataFrame(
                [(user, pid, "2020-09-22") for user, pid in purchases],
                columns=["user_id", "product_id", "purchase_date"],
            ),
        )
        graph = GraphRecommender(
            customers,
            catalog,
            PopularityRecommender(customers),
            pd.DataFrame(edges, columns=["product_id", "other_id", "count", "lift"]),
        )
        return customers, catalog, graph

    return build


def test_ten_unique_real_customers_are_deterministic_diverse_and_all_users_are_retained(make_data):
    products = [
        product(200 + i, product_type=f"Type {i}", product_group=f"Group {i % 3}")
        for i in range(12)
    ] + [product(900, product_group="Accessories")]
    users = [100 + i for i in range(12)] + [9999]
    purchases = [(100 + i, 200 + i) for i in range(12)]
    edges = [(200 + i, 900, 2, 1) for i in range(12)]
    customers, catalog, graph = make_data(products, purchases, users, edges)
    before = customers.purchases.copy(deep=True)
    catalog_before = list(catalog)

    selected = select_demo_customers(customers, catalog, graph)
    assert len(selected) == len({item.user_id for item in selected}) == 10
    assert all(isinstance(item, DemoCustomer) for item in selected)
    assert len({catalog.get(item.focus_product_id).product_type for item in selected}) == 10
    assert len({catalog.get(item.focus_product_id).product_group for item in selected[:3]}) == 3
    for item in selected:
        assert item.focus_product_id in {
            p.product_id for p in customers.purchases_for(item.user_id)
        }
        assert "co-purchased complements" in item.label
        assert "Purchase records: 1; distinct products: 1." in item.description
        assert graph.related(item.focus_product_id, "complement", top_k=1)
        assert set(asdict(item)) == {"user_id", "label", "description", "focus_product_id"}
    assert selected == select_demo_customers(customers, catalog, graph)
    shuffled = make_data(products[::-1], purchases[::-1], users[::-1], edges[::-1])
    assert selected == select_demo_customers(*shuffled)
    assert customers.user_ids() == sorted(users)
    assert customers.get_user(9999) is not None
    pd.testing.assert_frame_equal(customers.purchases, before)
    assert list(catalog) == catalog_before


def test_evidence_kinds_are_varied_and_purchase_graph_evidence_comes_first(make_data):
    category = "Apparel & Accessories > Clothing > Shirts & Tops"
    products = (
        [
            product(
                100 + i,
                product_type=f"Type {i}",
                product_group=f"Group {i}",
                shopify_category=category,
                shopify_attributes={"color": "Black", "pattern": "Solid"},
            )
            for i in range(12)
        ]
        + [
            product(200 + i, product_type=f"Type {i}", price=20, shopify_category=category)
            for i in range(12)
        ]
        + [
            product(900),
            product(901, product_group="Shoes"),
            product(910),
            product(911),
            product(800),
        ]
    )
    users = [*range(1, 13), 20, 30, 31, 9999]
    customers, catalog, graph = make_data(
        products,
        purchases=[(i + 1, 100 + i) for i in range(12)] + [(20, 800), (30, 900), (31, 910)],
        users=users,
        edges=[(900, 901, 2, 1), (910, 911, 2, 1)],
    )
    selected = select_demo_customers(customers, catalog, graph)
    assert len(selected) == len({item.user_id for item in selected}) == 10
    assert {item.label.split(": ", 1)[1] for item in selected[:3]} == {
        "co-purchased complements",
        "co-purchase links",
        "higher-priced alternatives",
    }
    assert selected[0].label.endswith(("co-purchased complements", "co-purchase links"))
    assert select_demo_customers(customers, catalog, graph, limit=1) == selected[:1]
    assert all(item.user_id != 20 for item in selected)
    assert customers.user_ids() == sorted(users)


def test_focus_prefers_real_links_over_richer_metadata_and_counts_actual_history(make_data):
    data = make_data(
        [
            product(
                1,
                shopify_category="Apparel & Accessories > Clothing > Shirts & Tops",
                shopify_attributes={"color": "Black", "pattern": "Solid"},
            ),
            product(9),
            product(10),
        ],
        purchases=[(11, 1), (11, 9), (11, 9)],
        users=[11],
        edges=[(9, 10, 2, 1)],
    )
    selected = select_demo_customers(*data)
    assert selected[0].focus_product_id == 9
    assert selected[0].label == "T-shirt: co-purchase links"
    assert "Purchase records: 3; distinct products: 2." in selected[0].description


def test_validated_alternatives_are_informative_without_co_purchase_edges(make_data):
    category = "Apparel & Accessories > Clothing > Shirts & Tops"
    data = make_data(
        [
            product(1),
            product(2, shopify_category=category),
            product(3, price=20, shopify_category=category),
        ],
        purchases=[(7, 1), (7, 2)],
        users=[7],
    )
    selected = select_demo_customers(*data)
    assert selected[0].focus_product_id == 2
    assert selected[0].label == "T-shirt: higher-priced alternatives"
    assert data[2].related(2, "alternative", top_k=1)


def test_metadata_richness_breaks_ties_without_inventing_evidence(make_data):
    data = make_data(
        [product(1), product(2, shopify_attributes={"color": "Black", "pattern": "Solid"})],
        purchases=[(5, 1), (5, 2)],
        users=[5],
    )
    selected = select_demo_customers(*data)
    assert selected[0].focus_product_id == 2
    assert selected[0].label == "T-shirt: product metadata"
    assert "co-purchase" not in selected[0].description
    assert "alternative" not in selected[0].description


def test_diversity_can_choose_another_focus_from_the_same_users_history(make_data):
    data = make_data(
        [
            product(1, shopify_attributes={"color": "Black"}),
            product(2, product_type="Dress", product_group="Garment Full body"),
        ],
        purchases=[(1, 1), (2, 1), (2, 2)],
        users=[1, 2],
    )
    selected = select_demo_customers(*data)
    assert [(item.user_id, item.focus_product_id) for item in selected] == [(1, 1), (2, 2)]


def test_evidence_is_cached_per_product_and_whole_recommendations_are_never_run(
    make_data, monkeypatch
):
    customers, catalog, graph = make_data(
        [product(1)], purchases=[(user, 1) for user in range(1, 16)], users=range(1, 16)
    )
    calls = Counter()
    related, neighborhood = graph.related, graph.neighborhood

    def counted_related(pid, kind, top_k):
        calls[pid, kind] += 1
        return related(pid, kind, top_k)

    def counted_neighborhood(pid, top_k):
        calls[pid, "neighborhood"] += 1
        return neighborhood(pid, top_k)

    def no_recommendations(*args, **kwargs):
        pytest.fail("Demo curation must not run per-user recommendations")

    monkeypatch.setattr(graph, "related", counted_related)
    monkeypatch.setattr(graph, "neighborhood", counted_neighborhood)
    monkeypatch.setattr(graph, "recommend", no_recommendations)
    assert len(select_demo_customers(customers, catalog, graph)) == 10
    assert calls == {(1, "complement"): 1, (1, "alternative"): 1, (1, "neighborhood"): 1}


def test_fewer_eligible_customers_are_returned_without_removing_other_users(make_data):
    data = make_data(
        [product(1), product(2, price=0)],
        purchases=[(2, 1), (3, 999), (4, 2), (9999, 1)],
        users=[1, 2, 3, 4],
    )
    selected = select_demo_customers(*data)
    assert [item.user_id for item in selected] == [2]
    assert selected[0].label == "T-shirt: purchase history"
    assert data[0].user_ids() == [1, 2, 3, 4]


@pytest.mark.parametrize("limit", [0, -1])
def test_nonpositive_limit_is_empty(make_data, limit):
    data = make_data([product(1)], purchases=[(1, 1)], users=[1])
    assert select_demo_customers(*data, limit=limit) == []


def test_positive_limit_is_respected(make_data):
    data = make_data([product(1)], purchases=[(1, 1), (2, 1)], users=[1, 2])
    assert len(select_demo_customers(*data, limit=1)) == 1


@pytest.mark.parametrize(
    "products, purchases, users",
    [
        ([], [], []),
        ([product(1)], [], [1]),
        ([], [(1, 999)], [1]),
    ],
)
def test_empty_or_no_usable_histories(make_data, products, purchases, users):
    assert select_demo_customers(*make_data(products, purchases, users)) == []
