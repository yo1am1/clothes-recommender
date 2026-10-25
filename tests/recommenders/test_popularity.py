from datetime import date

import pandas as pd
import pytest

from clothes_recommender.recommenders.popularity import PopularityRecommender

LAST_DAY = date(2020, 9, 22)


def purchases(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["product_id", "purchase_date"])


@pytest.fixture
def recommender() -> PopularityRecommender:
    return PopularityRecommender(
        purchases(
            [
                *[(1, LAST_DAY)] * 3,
                *[(2, date(2020, 9, 20))] * 2,
                *[(3, date(2020, 9, 16))] * 2,  # ties with product 2
                *[(4, date(2020, 9, 1))] * 5,  # best-seller, but three weeks ago
                (5, date(2020, 9, 15)),  # exactly 7 days before the last day: outside
            ]
        )
    )


def test_ranks_by_purchases_in_the_last_week(recommender) -> None:
    assert [p.product_id for p in recommender.top(10)] == [1, 2, 3]
    assert [p.score for p in recommender.top(10)] == [3.0, 2.0, 2.0]


def test_top_skips_excluded_ids_and_respects_top_k(recommender) -> None:
    assert [p.product_id for p in recommender.top(1, exclude_ids={1})] == [2]


def test_no_purchases_means_no_ranking() -> None:
    assert PopularityRecommender(purchases([])).top(5) == []
