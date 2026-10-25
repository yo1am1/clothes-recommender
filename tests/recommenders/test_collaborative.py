from datetime import date, timedelta

import pandas as pd
import pytest

from clothes_recommender.domain import ScoredProduct
from clothes_recommender.recommenders.base import UnknownUserError
from clothes_recommender.recommenders.collaborative import (
    POPULARITY_DAMPING,
    CollaborativeRecommender,
)

DAY = date(2025, 1, 15)
WEEK_EARLIER = DAY - timedelta(days=7)

# The crowd: (shopper, product). Buyers: 10 -> 3, 11 -> 2, 12 -> 2, 13 -> 1, 99 -> 1.
# Bought together: 10+11 by one shopper, 10+12 by two, 11+13 by one.
CROWD = [
    (1, 10),
    (1, 11),
    (2, 10),
    (2, 12),
    (2, 12),  # buying twice is not a stronger preference
    (3, 10),
    (3, 12),
    (4, 11),
    (4, 13),
    (5, 99),
]
# Sampled customers: 1 bought 10; 2 bought 11 now and 10 a week earlier; 3 nothing;
# 4 only 99, which nobody bought with anything else.
CUSTOMERS = [(1, 10, DAY), (2, 11, DAY), (2, 10, WEEK_EARLIER), (4, 99, DAY)]


def damped(together: float, buyers: int) -> float:
    return together * buyers**-POPULARITY_DAMPING


class FixedPopularity:
    """Stands in for PopularityRecommender with a known ranking."""

    def top(self, top_k: int, exclude_ids=()) -> list[ScoredProduct]:
        ranking = [ScoredProduct(pid, 1.0) for pid in (99, 50, 51)]
        return [s for s in ranking if s.product_id not in exclude_ids][:top_k]


@pytest.fixture
def recommender(make_customers, make_catalog) -> CollaborativeRecommender:
    crowd = pd.DataFrame(CROWD, columns=["shopper_id", "product_id"]).assign(purchase_date=DAY)
    return CollaborativeRecommender(
        make_customers([1, 2, 3, 4], CUSTOMERS),
        make_catalog([(pid, "Garment Upper body") for pid in (10, 11, 12, 13, 50, 51, 99)]),
        crowd,
        FixedPopularity(),
    )


def test_ranks_by_damped_co_purchases_then_fills_with_popular(recommender) -> None:
    recommendations = recommender.recommend(1, top_k=4)

    assert [r.product_id for r in recommendations] == [12, 11, 99, 50]
    assert recommendations[0].score == pytest.approx(damped(2, buyers=2))
    assert recommendations[1].score == pytest.approx(damped(1, buyers=2))
    assert [r.score for r in recommendations[2:]] == [0.0, 0.0]


def test_recent_purchases_outweigh_older_ones(recommender) -> None:
    # Unweighted, 12 (two shoppers with 10) would beat 13 (one shopper with 11). But 10 was
    # bought a week, one half-life, before 11, so its evidence counts half.
    recommendations = recommender.recommend(2, top_k=2)

    assert [r.product_id for r in recommendations] == [13, 12]
    assert recommendations[0].score == pytest.approx(damped(1, buyers=1))
    assert recommendations[1].score == pytest.approx(damped(0.5 * 2, buyers=2))


def test_never_recommends_purchases(recommender) -> None:
    ids = [r.product_id for r in recommender.recommend(2, top_k=10)]
    assert 10 not in ids and 11 not in ids


def test_customers_without_co_purchase_evidence_get_popular_products(recommender) -> None:
    assert [r.product_id for r in recommender.recommend(3, top_k=2)] == [99, 50]
    assert [r.product_id for r in recommender.recommend(4, top_k=2)] == [50, 51]


def test_reasons_name_the_purchases_that_point_to_a_product(recommender) -> None:
    assert recommender.reasons(1, 12) == [
        "2 shoppers who bought product 10 (black) also bought this."
    ]
    assert recommender.reasons(2, 12) == [
        "2 shoppers who bought product 10 (black) also bought this."
    ]
    assert recommender.reasons(1, 99)[0].startswith("Popular in the last week")


def test_unknown_user_raises(recommender) -> None:
    with pytest.raises(UnknownUserError):
        recommender.recommend(404)
    with pytest.raises(UnknownUserError):
        recommender.reasons(404, 12)
