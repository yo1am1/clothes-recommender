from datetime import date
from math import sqrt

import numpy as np
import pytest
from qdrant_client import QdrantClient

from clothes_recommender.recommenders.base import UnknownUserError
from clothes_recommender.recommenders.content_based import ContentBasedRecommender
from clothes_recommender.recommenders.popularity import PopularityRecommender
from clothes_recommender.vector_store import ProductVectorStore

TROUSERS, SNEAKERS, BRA, TSHIRT = "Trousers", "Sneakers", "Bra", "T-shirt"
AXIS = np.eye(4, dtype=np.float32)

# product_id -> (product type, stored vector or None when the product is not indexed)
PRODUCTS = {
    1: (TROUSERS, AXIS[0]),
    2: (TROUSERS, AXIS[1]),
    5: (SNEAKERS, None),
    6: (TSHIRT, AXIS[2]),
    7: (SNEAKERS, AXIS[3]),
    8: (BRA, AXIS[3]),
    10: (TROUSERS, AXIS[0]),
    11: (TROUSERS, AXIS[1]),
    12: (SNEAKERS, AXIS[1]),
    13: (TSHIRT, AXIS[2]),
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
    return make_catalog([(pid, product_type) for pid, (product_type, _) in PRODUCTS.items()])


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

    # Sneakers (12) are outside the user's product types; 1 and 2 are already bought.
    assert [r.product_id for r in recommendations] == [11, 10]
    norm = sqrt(1 + (1 / 8) ** 2)
    assert recommendations[0].score == pytest.approx(1 / norm, rel=1e-5)
    assert recommendations[1].score == pytest.approx((1 / 8) / norm, rel=1e-5)


def test_search_is_limited_to_the_top_three_product_types(recommender, catalog) -> None:
    # User 4: two trousers, then one each of sneakers, bra and T-shirt. The tie is
    # broken alphabetically, so the T-shirt is dropped.
    recommended = catalog.get_many(r.product_id for r in recommender.recommend(4))

    assert {p.product_id for p in recommended} == {10, 11, 12}


def test_user_without_purchases_falls_back_to_popularity(recommender, popular) -> None:
    assert recommender.recommend(2, top_k=3) == popular.top(3)


def test_user_without_indexed_purchases_falls_back_to_popularity(recommender, popular) -> None:
    recommendations = recommender.recommend(3, top_k=3)

    assert recommendations == popular.top(3, exclude_ids={5})


def test_unknown_user_raises(recommender) -> None:
    with pytest.raises(UnknownUserError):
        recommender.recommend(404)
