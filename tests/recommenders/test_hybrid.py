import pytest

from clothes_recommender.domain import ScoredProduct
from clothes_recommender.recommenders.base import UnknownUserError
from clothes_recommender.recommenders.hybrid import RRF_K, HybridRecommender


class FixedRecommender:
    def __init__(self, product_ids: list[int]) -> None:
        self._ids = product_ids
        self.requested_top_k: int | None = None

    def recommend(self, user_id: int, top_k: int = 10) -> list[ScoredProduct]:
        if user_id != 1:
            raise UnknownUserError(user_id)
        self.requested_top_k = top_k
        return [ScoredProduct(pid, 100.0 - i) for i, pid in enumerate(self._ids[:top_k])]


def test_products_listed_by_both_components_outrank_single_top_picks():
    hybrid = HybridRecommender({"a": FixedRecommender([1, 2, 3]), "b": FixedRecommender([4, 2])})
    ranked = hybrid.recommend(1, top_k=4)
    assert [item.product_id for item in ranked] == [2, 1, 4, 3]
    assert ranked[0].score == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 2))


def test_ties_go_to_the_first_component_and_top_k_truncates():
    hybrid = HybridRecommender({"a": FixedRecommender([9]), "b": FixedRecommender([5])})
    assert [item.product_id for item in hybrid.recommend(1, top_k=2)] == [9, 5]
    assert [item.product_id for item in hybrid.recommend(1, top_k=1)] == [9]


def test_equal_scores_from_mirrored_lists_follow_the_first_component():
    hybrid = HybridRecommender({"a": FixedRecommender([9, 5]), "b": FixedRecommender([5, 9])})
    assert [item.product_id for item in hybrid.recommend(1, top_k=2)] == [9, 5]


def test_explain_reports_each_components_rank():
    hybrid = HybridRecommender(
        {"collaborative": FixedRecommender([1, 2]), "content-based": FixedRecommender([2])}
    )
    top = hybrid.explain(1, top_k=1)[0]
    assert top.product_id == 2
    assert top.ranks == {"collaborative": 2, "content-based": 1}
    assert top.reasons() == ["Ranked #2 by collaborative.", "Ranked #1 by content-based."]


def test_components_are_read_deeper_than_top_k():
    component = FixedRecommender(list(range(1, 100)))
    HybridRecommender({"a": component}).recommend(1, top_k=3)
    assert component.requested_top_k == 50


def test_unknown_users_propagate_and_empty_hybrids_are_rejected():
    with pytest.raises(UnknownUserError):
        HybridRecommender({"a": FixedRecommender([1])}).recommend(2)
    with pytest.raises(ValueError, match="at least one"):
        HybridRecommender({})
