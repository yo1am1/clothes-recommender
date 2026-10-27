from datetime import date

import pytest

from clothes_recommender.domain import ScoredProduct
from clothes_recommender.recommenders.base import UnknownUserError
from clothes_recommender.recommenders.popularity import PopularityRecommender
from clothes_recommender.recommenders.repeat import RepeatPurchaseRecommender


def test_recent_unique_purchases_then_popularity_with_deterministic_ties(make_customers):
    customers = make_customers(
        [1, 2],
        [
            (1, 2, date(2020, 1, 2)),
            (1, 1, date(2020, 1, 2)),
            (1, 1, date(2020, 1, 1)),
            (2, 3, date(2020, 1, 2)),
        ],
    )
    recommender = RepeatPurchaseRecommender(customers, PopularityRecommender(customers.purchases))
    assert recommender.recommend(1, 3) == [
        ScoredProduct(1, 1),
        ScoredProduct(2, 0.5),
        ScoredProduct(3, 1 / 3),
    ]
    assert len(recommender.recommend(1, 1)) == 1
    assert recommender.recommend(1, 0) == []
    assert recommender.reasons(1, 1) == ["Buy again: last purchased on 2020-01-02."]
    assert recommender.reasons(1, 3)[0].startswith("Popular in the last week")
    with pytest.raises(UnknownUserError):
        recommender.recommend(99)


def test_known_customer_without_history_gets_best_sellers(make_customers):
    customers = make_customers([1, 2], [(1, 3, date(2020, 1, 2))])
    recommender = RepeatPurchaseRecommender(customers, PopularityRecommender(customers.purchases))
    assert recommender.recommend(2, 12) == [ScoredProduct(3, 1)]
