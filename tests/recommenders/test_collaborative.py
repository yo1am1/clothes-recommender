from datetime import date
from math import sqrt

import pytest

from clothes_recommender.domain import ScoredProduct
from clothes_recommender.recommenders.base import UnknownUserError
from clothes_recommender.recommenders.collaborative import CollaborativeRecommender, SimilarUser

DAY = date(2025, 1, 1)
PURCHASES = [
    (1, 10, DAY),
    (1, 10, DAY),  # a repeat purchase must not count as a stronger preference
    (1, 11, DAY),
    (2, 10, DAY),  # 2 and 6 share both of user 1's products...
    (2, 11, DAY),
    (2, 12, DAY),
    (6, 10, DAY),
    (6, 11, DAY),
    (6, 14, DAY),
    (3, 10, DAY),  # ...3 shares only one...
    (3, 13, DAY),
    (4, 99, DAY),  # ...and 4 shares nothing. User 5 bought nothing.
]


class FixedPopularity:
    """Stands in for PopularityRecommender with a known ranking."""

    def top(self, top_k: int, exclude_ids=()) -> list[ScoredProduct]:
        ranking = [ScoredProduct(pid, 1.0) for pid in (99, 50, 51)]
        return [s for s in ranking if s.product_id not in exclude_ids][:top_k]


@pytest.fixture
def recommender(make_customers) -> CollaborativeRecommender:
    return CollaborativeRecommender(
        make_customers([1, 2, 3, 4, 5, 6], PURCHASES), FixedPopularity()
    )


def test_similar_users_ranks_by_overlap_and_breaks_ties_by_user_id(recommender) -> None:
    assert recommender.similar_users(1) == [
        SimilarUser(user_id=2, similarity=pytest.approx(2 / sqrt(2 * 3))),
        SimilarUser(user_id=6, similarity=pytest.approx(2 / sqrt(2 * 3))),
        SimilarUser(user_id=3, similarity=pytest.approx(1 / sqrt(2 * 2))),
    ]


def test_similar_users_respects_top_k(recommender) -> None:
    assert [u.user_id for u in recommender.similar_users(1, top_k=1)] == [2]


def test_recommend_suggests_neighbours_items_but_not_own_purchases(recommender) -> None:
    recommendations = recommender.recommend(1, top_k=3)

    assert [r.product_id for r in recommendations] == [12, 14, 13]
    assert recommendations[0].score == pytest.approx(2 / sqrt(6))
    assert recommendations[2].score == pytest.approx(0.5)


def test_short_neighbourhood_is_topped_up_with_popular_items(recommender) -> None:
    recommendations = recommender.recommend(1, top_k=5)

    assert [r.product_id for r in recommendations] == [12, 14, 13, 99, 50]
    assert [r.score for r in recommendations[3:]] == [0.0, 0.0]


def test_recommend_respects_top_k(recommender) -> None:
    assert [r.product_id for r in recommender.recommend(1, top_k=2)] == [12, 14]


def test_user_without_purchases_falls_back_to_popularity(recommender) -> None:
    assert recommender.similar_users(5) == []
    assert [r.product_id for r in recommender.recommend(5)] == [99, 50, 51]


def test_user_without_overlap_gets_popular_items_they_did_not_buy(recommender) -> None:
    assert [r.product_id for r in recommender.recommend(4)] == [50, 51]


def test_unknown_user_raises(recommender) -> None:
    with pytest.raises(UnknownUserError):
        recommender.similar_users(404)
    with pytest.raises(UnknownUserError):
        recommender.recommend(404)
