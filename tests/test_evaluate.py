import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from qdrant_client import QdrantClient

from clothes_recommender.domain import ScoredProduct
from clothes_recommender.vector_store import ProductVectorStore
from scripts import evaluate


def test_average_precision_hand_calculated_and_deduplicated():
    # Unique ranks [9, 1, 2, 3]: hits at ranks 2, 3, 4; denominator is three.
    assert evaluate.average_precision_at_k([1, 1, 2, 3], [9, 9, 1, 2, 3], 4) == pytest.approx(
        (1 / 2 + 2 / 3 + 3 / 4) / 3
    )
    assert evaluate.average_precision_at_k([1, 2, 3], [1, 1, 2, 3], 2) == 1
    assert evaluate.average_precision_at_k([], [1]) == 0
    assert evaluate.average_precision_at_k([1], []) == 0
    with pytest.raises(ValueError, match="positive"):
        evaluate.average_precision_at_k([1], [1], 0)


@pytest.fixture
def prepared_evaluation(tmp_path, sample_products, monkeypatch):
    pd.DataFrame([p.model_dump() for p in sample_products]).to_parquet(
        tmp_path / "products.parquet"
    )
    pd.DataFrame({"user_id": [1, 2, 3], "age": [25, 30, 40]}).to_parquet(tmp_path / "users.parquet")
    rows = [(u, p, "2020-01-01") for u in (1, 2) for p in (1, 2)]
    rows += [(3, 4, "2020-01-01")]
    rows += [(u, p, "2020-01-08") for u in (1, 2) for p in (1, 3, 3)]
    pd.DataFrame(rows, columns=["user_id", "product_id", "purchase_date"]).to_parquet(
        tmp_path / "purchases.parquet"
    )
    # The evaluator must never read these precomputed links, even if they exist.
    (tmp_path / "co_purchases.parquet").write_text("holdout-contaminated; do not read")
    seen = {}

    class Graph:
        def __init__(self, customers, catalog, popular, co_purchases):
            seen["customers"] = customers
            seen["links"] = co_purchases

        def recommend(self, user_id, top_k):
            assert top_k == 12
            return [ScoredProduct(3, 1.0), ScoredProduct(3, 0.5), ScoredProduct(4, 0.1)]

    monkeypatch.setitem(
        sys.modules,
        "clothes_recommender.recommenders.graph",
        SimpleNamespace(GraphRecommender=Graph),
    )
    return tmp_path, seen


def test_future_purchases_never_reach_training_or_co_purchase_links(
    prepared_evaluation, monkeypatch
):
    data_dir, seen = prepared_evaluation

    def unexpected_index(*args, **kwargs):
        pytest.fail("An index must only be opened with explicit --qdrant-path")

    monkeypatch.setattr(evaluate, "QdrantClient", unexpected_index)
    result = evaluate.evaluate(data_dir)
    assert result["cutoff"] == "2020-01-02"
    assert set(seen["customers"].purchases["product_id"]) == {1, 2, 4}
    assert seen["customers"].purchases["purchase_date"].max().isoformat() == "2020-01-01"
    assert seen["links"][["product_id", "other_id", "count"]].to_dict("records") == [
        {"product_id": 1, "other_id": 2, "count": 2}
    ]
    assert result["users_evaluated"] == 2
    graph = result["results"]["graph"]
    assert graph["full_holdout"] == {
        "users_evaluated": 2,
        "map_at_12": 0.5,
        "hit_rate": 1.0,
        "recall_at_12": 0.5,
    }
    assert graph["novel_item_holdout"] == {
        "users_evaluated": 2,
        "map_at_12": 1.0,
        "hit_rate": 1.0,
        "recall_at_12": 1.0,
    }
    assert graph["catalog_coverage"] == pytest.approx(2 / 6)
    assert graph["recommendation_fill"] == pytest.approx(2 / 12)
    assert result["content_index"] is None
    assert result["repeat_purchase_limitation"]["full_holdout_unique_user_items"] == 4
    assert result["repeat_purchase_limitation"]["novel_holdout_unique_user_items"] == 2
    for metrics in result["results"].values():
        assert 0 <= metrics["full_holdout"]["map_at_12"] <= 1
        assert 0 <= metrics["novel_item_holdout"]["map_at_12"] <= 1
        assert 0 <= metrics["catalog_coverage"] <= 1
        assert 0 <= metrics["recommendation_fill"] <= 1


def test_tiny_cli_prints_reproducible_json(prepared_evaluation, capsys):
    data_dir, _ = prepared_evaluation
    evaluate.main(["--data-dir", str(data_dir)])
    result = json.loads(capsys.readouterr().out)
    assert result == evaluate.evaluate(data_dir)
    assert set(result["results"]) == {
        "popularity",
        "popularity_novel",
        "repeat",
        "collaborative",
        "graph",
    }
    assert result["sample_scope"]["training_purchases"] == 5
    assert result["sample_scope"]["holdout_purchases"] == 6


def test_cli_output_file_matches_printed_json(prepared_evaluation, capsys, tmp_path):
    data_dir, _ = prepared_evaluation
    output = tmp_path / "evaluation.json"
    evaluate.main(["--data-dir", str(data_dir), "--output", str(output)])
    assert json.loads(output.read_text()) == json.loads(capsys.readouterr().out)


def test_repeat_only_holdout_has_no_novel_map(prepared_evaluation):
    data_dir, _ = prepared_evaluation
    path = data_dir / "purchases.parquet"
    purchases = pd.read_parquet(path)
    purchases.loc[purchases["product_id"] == 3, "product_id"] = 1
    purchases.to_parquet(path)
    result = evaluate.evaluate(data_dir)
    for metrics in result["results"].values():
        assert metrics["novel_item_holdout"] == {
            "users_evaluated": 0,
            "map_at_12": None,
            "hit_rate": None,
            "recall_at_12": None,
        }


@pytest.mark.parametrize("case", ["empty", "no_train", "no_overlap"])
def test_no_eligible_users_is_actionable(prepared_evaluation, case):
    data_dir, _ = prepared_evaluation
    path = data_dir / "purchases.parquet"
    purchases = pd.read_parquet(path)
    if case == "empty":
        purchases = purchases.iloc[:0]
    elif case == "no_train":
        purchases["purchase_date"] = "2020-01-08"
    else:
        purchases.loc[purchases["purchase_date"] == "2020-01-08", "user_id"] = 3
        purchases = purchases[~((purchases["user_id"] == 3) & (purchases["product_id"] == 4))]
    purchases.to_parquet(path)
    with pytest.raises(ValueError, match="prepare"):
        evaluate.evaluate(data_dir)


@pytest.mark.parametrize("case", ["empty", "foreign", "holdout_only"])
def test_explicit_index_must_be_compatible(prepared_evaluation, tmp_path, sample_products, case):
    data_dir, _ = prepared_evaluation
    with pytest.raises(ValueError, match="existing"):
        evaluate.evaluate(data_dir, tmp_path / "missing")
    index = tmp_path / "index"
    client = QdrantClient(path=str(index))
    store = ProductVectorStore(client, evaluate.get_settings().qdrant_collection)
    store.ensure_collection(3)
    if case != "empty":
        product = sample_products[2]
        if case == "foreign":
            product = product.model_copy(update={"product_id": 999})
        store.upsert([product], np.array([[1.0, 0.0, 0.0]]))
    store.close()
    message = "training purchase vectors" if case == "holdout_only" else "prepared catalog IDs"
    with pytest.raises(ValueError, match=message):
        evaluate.evaluate(data_dir, index)


def test_explicit_compatible_index_enables_content(prepared_evaluation, sample_products, tmp_path):
    data_dir, _ = prepared_evaluation
    index = tmp_path / "index"
    store = ProductVectorStore(
        QdrantClient(path=str(index)), evaluate.get_settings().qdrant_collection
    )
    store.ensure_collection(3)
    store.upsert(sample_products, np.tile([1.0, 0.0, 0.0], (len(sample_products), 1)))
    store.close()
    result = evaluate.evaluate(data_dir, index)
    assert "content" in result["results"]
    assert result["content_index"]["indexed_products"] == 6
    # A second open also verifies that evaluation released the local index lock.
    assert evaluate.evaluate(data_dir, index) == result


def test_split_excludes_cutoff_day_from_training():
    purchases = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "product_id": [1, 2, 3],
            "purchase_date": ["2020-01-01", "2020-01-02", "2020-01-08"],
        }
    )
    cutoff, train, holdout = evaluate.chronological_split(purchases)
    assert cutoff == pd.Timestamp("2020-01-02")
    assert train["product_id"].tolist() == [1]
    assert holdout["product_id"].tolist() == [2, 3]
    assert (train["purchase_date"] < cutoff).all()
    assert (holdout["purchase_date"] >= cutoff).all()


def test_crowd_signals_use_every_shopper_but_only_before_the_cutoff(prepared_evaluation):
    data_dir, seen = prepared_evaluation
    # Two unsampled shoppers buy 1 and 4 together before the cutoff (a link needs two
    # baskets); two more buy 2 and 4 together in the holdout week, which must stay unseen.
    crowd = [(s, p, "2020-01-01") for s in (100, 101) for p in (1, 4)]
    crowd += [(s, p, "2020-01-08") for s in (102, 103) for p in (2, 4)]
    pd.DataFrame(crowd, columns=["shopper_id", "product_id", "purchase_date"]).to_parquet(
        data_dir / "shopper_purchases.parquet"
    )
    result = evaluate.evaluate(data_dir)

    assert result["sample_scope"]["training_shopper_purchases"] == 4
    assert result["sample_scope"]["training_shoppers"] == 2
    assert seen["links"][["product_id", "other_id"]].values.tolist() == [[1, 4]]


def test_hit_rate_and_recall_have_independent_hand_calculated_denominators():
    metrics = evaluate.ranking_metrics({1: {1, 2}, 2: {3, 4, 5}}, {1: [1, 2], 2: [9]}, [1, 2])
    assert metrics == {"users_evaluated": 2, "map_at_12": 0.5, "hit_rate": 0.5, "recall_at_12": 0.5}


def test_repeat_baseline_can_recommend_known_purchases(prepared_evaluation):
    data_dir, _ = prepared_evaluation
    result = evaluate.evaluate(data_dir)
    assert result["results"]["repeat"]["full_holdout"]["hit_rate"] == 1
    assert result["results"]["repeat"]["full_holdout"]["recall_at_12"] == 0.5
    assert result["results"]["repeat"]["novel_item_holdout"]["recall_at_12"] == 0


def test_explicit_validation_window_ignores_later_purchases():
    from datetime import date

    purchases = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "product_id": [1, 2, 3],
            "purchase_date": ["2020-01-01", "2020-01-08", "2020-01-15"],
        }
    )
    cutoff, train, holdout = evaluate.chronological_split(purchases, date(2020, 1, 8))
    assert cutoff == pd.Timestamp("2020-01-02")
    assert train["product_id"].tolist() == [1]
    assert holdout["product_id"].tolist() == [2]


def test_ranker_artifact_cannot_have_seen_holdout_labels(prepared_evaluation):
    from datetime import date

    from clothes_recommender.recommenders.learned import FEATURES, RankerModel

    data_dir, _ = prepared_evaluation
    model = RankerModel(
        weights=[0] * len(FEATURES),
        trained_through=date(2019, 12, 31),
        training_window_ends=[date(2019, 12, 17), date(2019, 12, 24), date(2019, 12, 31)],
        training_rows=10,
        training_positives=1,
    )
    path = data_dir / "ranker.json"
    path.write_text(model.model_dump_json())
    result = evaluate.evaluate(data_dir, ranker_path=path)
    assert "learned" in result["results"]
    assert result["ranker"]["trained_through"] == "2019-12-31"
    assert "weights" not in result["ranker"]
    data = model.model_dump(mode="json")
    data["trained_through"] = "2020-01-02"
    data["training_window_ends"] = ["2019-12-19", "2019-12-26", "2020-01-02"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="strictly before"):
        evaluate.evaluate(data_dir, ranker_path=path)


def test_paired_bootstrap_preserves_identical_and_known_customer_gains():
    truth = {1: {1}, 2: {2}}
    baseline = {1: [], 2: []}
    candidate = {1: [1], 2: [2]}
    zero = evaluate.paired_map_gain(truth, candidate, candidate, [1, 2])
    assert (zero["mean_delta"], zero["lower_95"], zero["upper_95"]) == (0, 0, 0)
    gain = evaluate.paired_map_gain(truth, candidate, baseline, [1, 2])
    assert (gain["mean_delta"], gain["lower_95"], gain["upper_95"]) == (1, 1, 1)
