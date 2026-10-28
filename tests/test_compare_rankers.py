from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import compare_rankers as experiment
from scripts.compare_retrieval import CandidateRows, Window


def window(marker=1, width=45):
    features = np.zeros((2, width))
    features[:, 0] = marker
    features[:, 1] = [0, 1]
    return Window(
        {100: CandidateRows([2, 1], features, 1), 999: CandidateRows([], np.empty((0, width)), 0)},
        {100: {1}, 999: {99}},
        {},
    )


def test_enrichment_keeps_ids_original_features_and_normalizes_sources():
    original = window(width=24)
    original.rows[100].features[:, 17] = [0, 3]
    original.rows[100].features[:, 23] = [0.1, 0.2]
    before = original.rows[100].features.copy()
    context = SimpleNamespace(
        features=lambda uid, ids: np.ones((len(ids), len(experiment.FEATURE_NAMES)))
    )
    enriched = experiment.enrich(original, context)
    assert enriched.rows[100].ids == [2, 1]
    assert enriched.rows[100].base_count == 1
    np.testing.assert_array_equal(enriched.rows[100].features[:, :24], before)
    np.testing.assert_array_equal(original.rows[100].features, before)
    np.testing.assert_allclose(
        enriched.rows[100].features[:, 24:27], [[0, 0, np.log1p(0.1) / np.log1p(0.2)], [1, 0, 1]]
    )
    assert enriched.rows[999].features.shape == (0, 24 + 3 + len(experiment.FEATURE_NAMES))
    context.features = lambda uid, ids: np.full((len(ids), len(experiment.FEATURE_NAMES)), np.nan)
    with pytest.raises(ValueError, match="finite"):
        experiment.enrich(original, context)


@pytest.mark.parametrize("variant", experiment.VARIANTS)
def test_fit_uses_prior_validation_groups_and_ignores_target_truth(monkeypatch, tmp_path, variant):
    calls = []
    ranking = variant == "context_lambdarank"
    width = 24 if variant == "loss_selection" else 45

    def dataset(data, label, **kwargs):
        if ranking:
            assert sum(kwargs["group"]) == len(label)
        else:
            assert not kwargs
        return SimpleNamespace(data=data, label=label, groups=kwargs.get("group"))

    def train(params, data, num_boost_round, **kwargs):
        calls.append((set(data.data[:, 0]), num_boost_round, data.groups))
        if "valid_sets" in kwargs:
            assert set(kwargs["valid_sets"][0].data[:, 0]) == {3}
            if ranking:
                assert params["lambdarank_truncation_level"] == 15
                assert kwargs["feval"](np.array([0.1, 0.9]), None) == (
                    "full_truth_map12",
                    0.5,
                    True,
                )
            else:
                assert params["metric"] == "binary_logloss"
                assert kwargs["feval"] is None
        return SimpleNamespace(
            best_iteration=7,
            predict=lambda x: x[:, 1],
            save_model=lambda path: Path(path).write_text("synthetic model"),
        )

    fake = SimpleNamespace(Dataset=dataset, train=train, early_stopping=lambda *a, **k: "callback")
    monkeypatch.setattr(experiment, "_lightgbm", lambda: fake)
    history = [window(i, width) for i in (1, 2, 3)]
    target = window(4, width)
    first, metadata = experiment.fit_predict(
        history, target, variant, model_path=tmp_path / "model.txt"
    )
    target.truth = {100: {2, 88}, 999: {1, 2}}
    second, other = experiment.fit_predict(history, target, variant)
    assert first == second == {100: [1, 2], 999: []}
    assert metadata == other
    assert metadata["selected_rounds"] == 7
    assert (tmp_path / "model.txt").read_text() == "synthetic model"
    assert [c[:2] for c in calls] == [({1, 2}, 400), ({1, 2, 3}, 7)] * 2
    assert [c[2] for c in calls[:2]] == ([[2, 2], [2, 2, 2]] if ranking else [None, None])


def test_unknown_variant_rejected_before_model_or_arrays():
    with pytest.raises(ValueError, match="Unknown ranking"):
        experiment.fit_predict([], window(), "unknown")


def test_cli_refuses_report_over_input_directory(tmp_path):
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


def test_confirmation_tests_only_frozen_arm_with_prior_training_and_full_truth(
    tmp_path, sample_products, monkeypatch
):
    from datetime import date

    import pandas as pd

    pd.DataFrame([p.model_dump() for p in sample_products]).to_parquet(
        tmp_path / "products.parquet"
    )
    pd.DataFrame({"user_id": [100, 999], "age": [None, None]}).to_parquet(
        tmp_path / "users.parquet"
    )
    events = pd.DataFrame(
        [(u, p, f"2020-01-{d:02d}") for d in (1, 8, 15, 22, 29) for u, p in [(100, 1), (999, 2)]],
        columns=["user_id", "product_id", "purchase_date"],
    )
    events.to_parquet(tmp_path / "purchases.parquet")
    events.rename(columns={"user_id": "shopper_id"}).to_parquet(
        tmp_path / "shopper_purchases.parquet"
    )

    def built(products, users, purchases, crowd, end):
        w = window(end.day, width=24)
        w.truth[999] = {2}
        w.rows[100].base_count = 2
        w.diagnostics = {"holdout_end": end.isoformat()}
        return w

    calls = []

    def fitted(training, target, variant, model_path):
        calls.append((variant, [w.diagnostics["holdout_end"] for w in training]))
        return {100: [1, 2], 999: []}, {"selected_rounds": 7}

    monkeypatch.setattr(experiment, "_lightgbm", lambda: None)
    monkeypatch.setattr(experiment, "build_window", built)
    monkeypatch.setattr(experiment, "fit_predict", fitted)
    result = experiment.confirm(tmp_path, date(2020, 1, 29))
    assert calls == [("loss_selection", ["2020-01-08", "2020-01-15", "2020-01-22"])]
    assert set(result["metrics"]) == {"v1", "challenger", "repeat"}
    assert result["metrics"]["challenger"]["users_evaluated"] == 2
    assert result["metrics"]["challenger"]["map_at_12"] == 0.5
    assert result["relative_map_gain"] == 0
    assert result["practical_gate_passed"] is False
    assert result["positive_paired_interval"] is False
    assert result["material_slice_warnings"] == []


def test_confirmation_cli_refuses_multiple_target_weeks(tmp_path):
    with pytest.raises(SystemExit):
        experiment.main(
            [
                "--data-dir",
                str(tmp_path),
                "--holdout-end",
                "2020-09-01",
                "--holdout-end",
                "2020-09-08",
                "--confirm",
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
    assert not list(tmp_path.iterdir())
