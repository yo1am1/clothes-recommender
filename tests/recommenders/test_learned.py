from datetime import date

import numpy as np
import pandas as pd
import pytest

from clothes_recommender.catalog import CustomerData
from clothes_recommender.recommenders.base import UnknownUserError
from clothes_recommender.recommenders.collaborative import CollaborativeRecommender
from clothes_recommender.recommenders.learned import (
    FEATURES,
    LearnedRecommender,
    RankerModel,
    fit,
    load_model,
)
from clothes_recommender.recommenders.popularity import PopularityRecommender
from scripts.train_ranker import train


def artifact(**updates):
    values = dict(
        weights=[0.0] * len(FEATURES),
        trained_through=date(2020, 1, 22),
        training_window_ends=[date(2020, 1, 8), date(2020, 1, 15), date(2020, 1, 22)],
        training_rows=20,
        training_positives=3,
    )
    return RankerModel(**(values | updates))


def test_json_model_rejects_wrong_features_nonfinite_weights_and_inconsistent_provenance(tmp_path):
    path = tmp_path / "ranker.json"
    path.write_text(artifact().model_dump_json())
    assert load_model(path) == artifact()
    for changes in (
        {"weights": [0]},
        {"weights": [float("inf")] * len(FEATURES)},
        {"feature_version": 2},
        {"training_positives": 20},
        {"training_window_ends": [date(2020, 1, 22)] * 3},
    ):
        with pytest.raises(ValueError):
            artifact(**changes)
    path.write_bytes(b" " * (64 * 1024 + 1))
    with pytest.raises(ValueError, match="64 KiB"):
        load_model(path)


def test_fixed_candidate_features_use_global_as_of_and_rank_deterministically(sample_catalog):
    rows = [(1, 1, "2020-01-01"), (1, 1, "2020-01-01"), (2, 1, "2020-01-01"), (2, 3, "2020-01-01")]
    purchases = pd.DataFrame(rows, columns=["user_id", "product_id", "purchase_date"])
    customers = CustomerData(pd.DataFrame({"user_id": [1, 2, 3], "age": [30] * 3}), purchases)
    popular = PopularityRecommender(purchases)
    collaborative = CollaborativeRecommender(
        customers, sample_catalog, purchases.rename(columns={"user_id": "shopper_id"}), popular
    )
    weights = np.zeros(len(FEATURES))
    weights[FEATURES.index("repeat")] = 1
    ranker = LearnedRecommender(
        customers, sample_catalog, collaborative, popular, weights, date(2020, 1, 8)
    )
    ids, features = ranker.candidates(1)
    row = features[ids.index(1)]
    assert row[FEATURES.index("frequency")] == pytest.approx(np.log(3))
    assert row[FEATURES.index("recency_7d")] == pytest.approx(np.exp(-1))
    assert len(ids) == len(set(ids))
    assert ranker.recommend(1)[0].product_id == 1
    assert ranker.recommend(1, 0) == []
    assert ranker.reasons(1, 1)[0] == "Buy again: last purchased on 2020-01-01."
    assert ranker.recommend(3)
    with pytest.raises(UnknownUserError):
        ranker.recommend(99)
    with pytest.raises(ValueError, match="weights"):
        LearnedRecommender(customers, sample_catalog, collaborative, popular, [0], date(2020, 1, 8))


def test_logistic_fit_learns_signal_and_rejects_missing_label_classes():
    features = np.zeros((4, len(FEATURES)))
    features[:, 0] = 1
    features[2:, 1] = 1
    weights = fit(features, [0, 0, 1, 1])
    assert np.isfinite(weights).all()
    assert (features[2] @ weights) > (features[0] @ weights)
    for labels in ([0] * 4, [1] * 4, [[0], [0], [1], [1]]):
        with pytest.raises(ValueError, match="binary label"):
            fit(features, labels)


def test_training_explicit_windows_ignore_every_later_purchase(tmp_path, sample_products):
    pd.DataFrame([p.model_dump() for p in sample_products]).to_parquet(
        tmp_path / "products.parquet"
    )
    pd.DataFrame({"user_id": [1, 2, 3], "age": [30] * 3}).to_parquet(tmp_path / "users.parquet")
    rows = [
        (uid, pid, f"2020-01-{day:02d}")
        for day in (1, 8, 15, 22, 29)
        for uid, pid in ((1, 1), (2, 2), (2, 3), (3, 4))
    ]
    purchases = pd.DataFrame(rows, columns=["user_id", "product_id", "purchase_date"])
    purchases.to_parquet(tmp_path / "purchases.parquet")
    first = train(tmp_path, date(2020, 1, 29))
    future = pd.DataFrame([(1, 6, "2020-03-01")] * 20, columns=purchases.columns)
    pd.concat([purchases, future]).to_parquet(tmp_path / "purchases.parquet")
    second = train(tmp_path, date(2020, 1, 29))
    assert first.model_dump() == second.model_dump()
    assert first.training_window_ends == [date(2020, 1, 15), date(2020, 1, 22), date(2020, 1, 29)]
    assert 0 < first.training_positives < first.training_rows
