from datetime import date
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from clothes_recommender.domain import ScoredProduct
from scripts import compare_retrieval as experiment


def small_window():
    first = np.zeros((4, 24))
    first[2, 15] = 1  # family recalls product 2
    first[3, 21] = 1  # latent recalls product 5
    return experiment.Window(
        {
            1: experiment.CandidateRows([1, 4, 2, 5], first, 2),
            2: experiment.CandidateRows([7], np.zeros((1, 24)), 1),
        },
        {1: {1, 2, 3}, 2: {9}},
        {},
    )


def test_reach_oracle_and_incremental_counts_keep_unreachable_users_and_truth():
    window = small_window()
    result = experiment.retrieval_diagnostics(window.rows, window.truth)
    assert result["users"] == 2
    assert result["truth_pairs"] == 4
    base, union = (result["sources"][name] for name in ("base", "union"))
    assert base["micro_recall"] == 0.25
    assert union["micro_recall"] == 0.5
    assert base["macro_recall"] == pytest.approx(1 / 6)
    assert union["oracle_map_at_12"] == pytest.approx(1 / 3)
    assert union["additional_pairs_over_base"] == 1
    assert union["users_with_reachable_truth"] == 1
    assert result["sources"]["family"]["additional_pairs_over_base"] == 1
    assert result["sources"]["latent"]["additional_pairs_over_base"] == 0
    assert union["candidates_min_median_max"] == [1, 2.5, 4]


def test_arrays_labels_and_prediction_offsets_match_both_candidate_pools():
    window = small_window()
    _, labels = experiment.arrays([window], True)
    assert labels.tolist() == [1, 0, 1, 0, 0]
    assert experiment.predictions(window, np.array([0.1, 0.2, 0.2, 0.3, 0.5]), True) == {
        1: [5, 2, 4, 1],
        2: [7],
    }
    _, labels = experiment.arrays([window], False)
    assert labels.tolist() == [1, 0, 0]
    assert experiment.predictions(window, np.array([0.1, 0.2, 0.9]), False) == {1: [4, 1], 2: [7]}
    with pytest.raises(ValueError, match="align"):
        experiment.predictions(window, np.ones(6), True)


def test_source_features_keep_overlapping_sources_and_native_scores():
    ranker = Mock()

    def candidates(uid, extra_ids=()):
        ids = list(dict.fromkeys([1, *extra_ids]))
        return ids, np.zeros((len(ids), 15))

    ranker.candidates.side_effect = candidates
    channels = Mock()
    channels.channels.return_value = {
        "family": [ScoredProduct(2, 2.5)],
        "recent_basket": [],
        "latent": [ScoredProduct(2, 0.2)],
    }
    channels.scores.return_value = {
        "family": [0, 2.5],
        "recent_basket": [0, 0],
        "latent": [0, 0.2],
    }
    rows = experiment.feature_rows(ranker, channels, 1)
    assert rows.ids == [1, 2]
    assert rows.base_count == 1
    np.testing.assert_allclose(rows.features[1, 15:], [1, 1, 2.5, 0, 0, 0, 1, 1, 0.2])
    channels.scores.return_value["latent"] = [0, np.nan]
    with pytest.raises(ValueError, match="finite"):
        experiment.feature_rows(ranker, channels, 1)


def test_cli_rejects_report_inside_dataset_before_reading_anything(tmp_path):
    with pytest.raises(SystemExit):
        experiment.main(
            [
                "--data-dir",
                str(tmp_path),
                "--holdout-end",
                "2020-09-01",
                "--output",
                str(tmp_path / "users.parquet"),
            ]
        )
    assert not list(tmp_path.iterdir())


def test_comparison_uses_only_earlier_history_and_crowd(tmp_path, sample_products, monkeypatch):
    products = pd.DataFrame(
        [p.model_dump() | {"product_code": p.product_id} for p in sample_products]
    )
    products.to_parquet(tmp_path / "products.parquet")
    pd.DataFrame({"user_id": [1, 2], "age": [30, 30]}).to_parquet(tmp_path / "users.parquet")
    events = pd.DataFrame(
        [(1, 1, "2020-01-01"), (1, 2, "2020-01-08"), (2, 3, "2020-03-01")],
        columns=["user_id", "product_id", "purchase_date"],
    )
    events.to_parquet(tmp_path / "purchases.parquet")
    events.rename(columns={"user_id": "shopper_id"}).to_parquet(
        tmp_path / "shopper_purchases.parquet"
    )

    class Channels:
        def __init__(self, products, customers, crowd, as_of):
            assert as_of == date(2020, 1, 1)
            assert set(customers.purchases["product_id"]) == {1}
            assert set(crowd["product_id"]) == {1}

        def channels(self, user_id, top_k):
            return {name: [] for name in experiment.CHANNELS}

        def scores(self, user_id, product_ids):
            return {name: np.zeros(len(product_ids)) for name in experiment.CHANNELS}

    monkeypatch.setattr(experiment, "ComplementaryRetrieval", Channels)
    result = experiment.compare(tmp_path, date(2020, 1, 8))
    assert result["retrieval"]["users"] == 1
    assert result["retrieval"]["truth_pairs"] == 1
    assert result["retrieval"]["sources"]["union"]["micro_recall"] == 0
    assert result["ranking"] is None


def test_tree_selection_uses_only_prior_weeks_then_refits_before_target(monkeypatch):
    import sys
    from types import SimpleNamespace

    def window(marker):
        features = np.zeros((2, 24))
        features[:, 0] = marker
        features[0, 1] = 1
        return experiment.Window({1: experiment.CandidateRows([1, 2], features, 2)}, {1: {1}}, {})

    calls = []

    def train(params, dataset, num_boost_round, **kwargs):
        calls.append((set(dataset.data[:, 0]), num_boost_round))
        if "valid_sets" in kwargs:
            assert set(kwargs["valid_sets"][0].data[:, 0]) == {3}
            name, value, larger = kwargs["feval"](np.array([0.9, 0.1]), None)
            assert (name, value, larger) == ("full_truth_map12", 1.0, True)
        return SimpleNamespace(best_iteration=7, predict=lambda x: x[:, 1])

    fake = SimpleNamespace(
        Dataset=lambda data, label: SimpleNamespace(data=data, label=label),
        train=train,
        early_stopping=lambda *a, **k: "callback",
    )
    monkeypatch.setitem(sys.modules, "lightgbm", fake)
    pred, rounds = experiment.tree_predictions([window(1), window(2), window(3)], window(4), True)
    assert calls == [({1, 2}, 400), ({1, 2, 3}, 7)]
    assert rounds == 7
    assert pred == {1: [1, 2]}


def test_rank_comparison_reports_buy_again_with_same_full_truth(
    tmp_path, sample_products, monkeypatch
):
    pd.DataFrame(
        [p.model_dump() | {"product_code": p.product_id} for p in sample_products]
    ).to_parquet(tmp_path / "products.parquet")
    pd.DataFrame({"user_id": [1, 2], "age": [30, 30]}).to_parquet(tmp_path / "users.parquet")
    rows = [(u, u, f"2020-01-{d:02d}") for d in (1, 8, 15, 22, 29) for u in (1, 2)]
    events = pd.DataFrame(rows, columns=["user_id", "product_id", "purchase_date"])
    events.to_parquet(tmp_path / "purchases.parquet")
    events.rename(columns={"user_id": "shopper_id"}).to_parquet(
        tmp_path / "shopper_purchases.parquet"
    )
    monkeypatch.setattr(experiment, "_lightgbm", lambda: None)
    monkeypatch.setattr(
        experiment,
        "tree_predictions",
        lambda training, target, expanded: ({uid: [uid] for uid in target.rows}, 1),
    )
    result = experiment.compare(tmp_path, date(2020, 1, 29), rank=True)
    repeat = result["ranking"]["metrics"]["repeat"]
    assert repeat == {"users_evaluated": 2, "map_at_12": 1.0, "hit_rate": 1.0, "recall_at_12": 1.0}


def test_missing_optional_ranker_fails_before_loading_data(tmp_path, monkeypatch):
    def missing():
        raise ValueError("optional LightGBM unavailable")

    monkeypatch.setattr(experiment, "_lightgbm", missing)
    with pytest.raises(ValueError, match="optional LightGBM"):
        experiment.compare(tmp_path / "absent", date(2020, 9, 1), rank=True)
