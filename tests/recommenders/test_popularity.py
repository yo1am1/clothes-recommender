import pytest

from clothes_recommender.recommenders.popularity import PopularityRecommender


@pytest.fixture
def recommender(make_catalog) -> PopularityRecommender:
    return PopularityRecommender(
        make_catalog(
            [
                (1, "jeans", 5.0, 1),  # perfect but unproven
                (2, "jeans", 4.6, 1000),
                (3, "shoes", 4.0, 1000),
                (4, "shoes", 0.0, 0),  # never reviewed
                (5, "shoes", 4.0, 1000),  # identical to 3
            ]
        )
    )


def test_well_reviewed_products_beat_a_single_perfect_review(recommender) -> None:
    assert [p.product_id for p in recommender.top(5)] == [2, 3, 5, 1, 4]


def test_ties_are_broken_by_product_id(recommender) -> None:
    top = recommender.top(5)
    assert top[1].score == top[2].score
    assert (top[1].product_id, top[2].product_id) == (3, 5)


def test_unreviewed_product_scores_zero(recommender) -> None:
    assert recommender.top(5)[-1].score == 0.0


def test_top_skips_excluded_ids_and_respects_top_k(recommender) -> None:
    assert [p.product_id for p in recommender.top(2, exclude_ids={2, 3})] == [5, 1]
