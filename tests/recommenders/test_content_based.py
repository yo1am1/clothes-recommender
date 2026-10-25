from datetime import date
from math import sqrt

import numpy as np
import pandas as pd
import pytest
from qdrant_client import QdrantClient

from clothes_recommender.catalog import ProductCatalog
from clothes_recommender.recommenders.base import UnknownUserError
from clothes_recommender.recommenders.content_based import ContentBasedRecommender
from clothes_recommender.recommenders.popularity import PopularityRecommender
from clothes_recommender.vector_store import ProductVectorStore

LOWER, SHOES, UNDERWEAR, UPPER = (
    "Garment Lower body",
    "Shoes",
    "Underwear",
    "Garment Upper body",
)
AXIS = np.eye(4, dtype=np.float32)

# product_id -> (product group, stored vector or None when the product is not indexed)
PRODUCTS = {
    1: (LOWER, AXIS[0]),
    2: (LOWER, AXIS[1]),
    5: (SHOES, None),
    6: (UPPER, AXIS[2]),
    7: (SHOES, AXIS[3]),
    8: (UNDERWEAR, AXIS[3]),
    10: (LOWER, AXIS[0]),
    11: (LOWER, AXIS[1]),
    12: (SHOES, AXIS[1]),
    13: (UPPER, AXIS[2]),
    14: (UNDERWEAR, AXIS[3]),
}
PURCHASES = [
    (1, 1, date(2025, 1, 1)),
    (1, 2, date(2025, 4, 1)),  # 90 days = 3 half-lives later, so product 1 weighs 1/8
    (3, 5, date(2025, 2, 1)),
    (4, 1, date(2025, 2, 1)),
    (4, 2, date(2025, 2, 1)),
    (4, 6, date(2025, 2, 1)),
    (4, 7, date(2025, 2, 1)),
    (4, 8, date(2025, 2, 1)),
]


@pytest.fixture
def catalog(make_catalog):
    return make_catalog([(pid, group) for pid, (group, _) in PRODUCTS.items()])


@pytest.fixture
def customers(make_customers):
    return make_customers([1, 2, 3, 4], PURCHASES)


@pytest.fixture
def store(catalog):
    client = QdrantClient(":memory:")
    store = ProductVectorStore(client, "products")
    store.ensure_collection(dimension=4)
    indexed = {pid: vector for pid, (_, vector) in PRODUCTS.items() if vector is not None}
    store.upsert(catalog.get_many(indexed), np.stack(list(indexed.values())))
    yield store
    client.close()


@pytest.fixture
def popular(customers) -> PopularityRecommender:
    return PopularityRecommender(customers)


@pytest.fixture
def recommender(customers, catalog, store, popular) -> ContentBasedRecommender:
    return ContentBasedRecommender(customers, catalog, store, popular)


def test_recent_purchases_dominate_the_taste_vector(recommender) -> None:
    recommendations = recommender.recommend(1)

    # Shoes (12) are outside the user's product groups; 1 and 2 are already bought.
    assert [r.product_id for r in recommendations] == [11, 10]
    norm = sqrt(1 + (1 / 8) ** 2)
    assert recommendations[0].score == pytest.approx(1 / norm, rel=1e-5)
    assert recommendations[1].score == pytest.approx((1 / 8) / norm, rel=1e-5)


def test_search_is_limited_to_the_top_three_product_groups(recommender, catalog) -> None:
    # User 4: two lower-body garments, then one each of shoes, underwear and upper body.
    # The tie is broken alphabetically, so underwear (14) is dropped.
    recommended = catalog.get_many(r.product_id for r in recommender.recommend(4))

    assert {p.product_id for p in recommended} == {10, 11, 12, 13}


def test_user_without_purchases_falls_back_to_popularity(recommender, popular) -> None:
    assert recommender.recommend(2, top_k=3) == popular.top(3)


def test_user_without_indexed_purchases_falls_back_to_popularity(recommender, popular) -> None:
    recommendations = recommender.recommend(3, top_k=3)

    assert recommendations == popular.top(3, exclude_ids={5})


def test_unknown_user_raises(recommender) -> None:
    with pytest.raises(UnknownUserError):
        recommender.recommend(404)


# Interests: product id -> (product group, section, stored vector). Dresses and leggings
# point in different directions, so one averaged taste vector would match neither well.
DRESS, LEGGINGS, SHIRT = AXIS[0], AXIS[1], AXIS[2]
INTEREST_PRODUCTS = {
    # Bought by user 1.
    101: ("Garment Full body", "Ladieswear", DRESS),
    102: ("Garment Full body", "Ladieswear", DRESS),
    103: ("Garment Lower body", "Ladieswear", LEGGINGS),
    # Candidates.
    201: ("Garment Full body", "Ladieswear", DRESS),
    202: ("Garment Full body", "Ladieswear", DRESS * 0.9 + LEGGINGS * 0.1),
    203: ("Garment Lower body", "Ladieswear", LEGGINGS),
    204: ("Garment Lower body", "Menswear", LEGGINGS),  # same look, a section never bought
    205: ("Garment Upper body", "Ladieswear", SHIRT),  # a group never bought
}


@pytest.fixture
def interest_recommender(make_customers):
    rows = [
        {
            "product_id": pid,
            "name": f"product {pid}",
            "product_type": group,
            "product_group": group,
            "section": section,
            "colour": "Black",
            "description": "",
            "price": 10.0,
        }
        for pid, (group, section, _) in INTEREST_PRODUCTS.items()
    ]
    catalog = ProductCatalog(pd.DataFrame(rows))
    # User 2 makes 204 and 205 the only popular products user 1 has not bought.
    customers = make_customers(
        [1, 2],
        [(1, pid, date(2025, 3, 1)) for pid in (101, 102, 103)]
        + [(2, pid, date(2025, 3, 1)) for pid in (204, 205)],
    )
    client = QdrantClient(":memory:")
    store = ProductVectorStore(client, "products")
    store.ensure_collection(dimension=4)
    vectors = np.stack([v / np.linalg.norm(v) for _, _, v in INTEREST_PRODUCTS.values()])
    store.upsert(catalog.get_many(INTEREST_PRODUCTS), vectors.astype(np.float32))
    yield ContentBasedRecommender(customers, catalog, store, PopularityRecommender(customers))
    client.close()


def test_each_interest_gets_its_own_vector_and_slots_by_weight(interest_recommender) -> None:
    # Two dresses and one pair of leggings: 3 slots split 2 dresses, 1 leggings, heaviest first.
    ids = [r.product_id for r in interest_recommender.recommend(1, top_k=3)]
    assert ids == [201, 203, 202]
    scores = {r.product_id: r.score for r in interest_recommender.recommend(1, top_k=3)}
    # Leggings match the leggings vector exactly, not a blend with the dresses.
    assert scores[203] == pytest.approx(1.0, rel=1e-5)


def test_interests_stay_in_the_sections_they_were_bought_from(interest_recommender) -> None:
    ids = [r.product_id for r in interest_recommender.recommend(1, top_k=10)]
    assert 204 not in ids[:3]  # menswear leggings only arrive as a popularity top-up
    assert ids[:3] == [201, 203, 202]


def test_short_lists_are_topped_up_with_popular_products(interest_recommender) -> None:
    recommendations = interest_recommender.recommend(1, top_k=6)
    # Three look-alikes in the bought groups and sections, then the two popular products.
    assert [r.product_id for r in recommendations] == [201, 203, 202, 204, 205]
    assert [r.score for r in recommendations[3:]] == [0.0, 0.0]
