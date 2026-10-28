from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import compare_bpr as study
from scripts.compare_retrieval import CandidateRows, Window


def window(end=date(2020, 9, 1)):
    return Window(
        {
            7: CandidateRows([2, 1], np.zeros((2, 24)), 1),
            8: CandidateRows([], np.empty((0, 24)), 0),
        },
        {7: {1, 3}, 8: {2}},
        {"holdout_end": end.isoformat()},
    )


def test_bpr_append_preserves_pool_truth_and_original_features():
    w = window()
    model = SimpleNamespace(features=lambda uid, ids: np.tile([-3.0, 0.5, 1], (len(ids), 1)))
    enriched = study.append_bpr(w, model)
    assert enriched.rows[7].ids == w.rows[7].ids
    assert enriched.rows[7].base_count == 1
    assert enriched.truth == w.truth
    assert w.rows[7].features.shape == (2, 24)
    assert enriched.rows[8].features.shape == (0, 27)
    np.testing.assert_array_equal(enriched.rows[7].features[:, :24], w.rows[7].features)
    np.testing.assert_allclose(enriched.rows[7].features[:, 24:], [[-3, 0.5, 1], [-3, 0.5, 1]])
    model.features = lambda uid, ids: np.full((len(ids), 3), np.nan)
    with pytest.raises(ValueError, match="finite"):
        study.append_bpr(w, model)


def test_compare_separates_sampling_and_selection_controls_and_keeps_truth(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "dependencies", lambda: None)
    monkeypatch.setattr(study.pd, "read_parquet", lambda p: pd.DataFrame())
    built = []

    def build(products, users, purchases, crowd, end):
        built.append(end)
        return window(end)

    monkeypatch.setattr(study, "build_window", build)
    asofs = []

    class BPR:
        def __init__(self, products, users, crowd, as_of):
            asofs.append(as_of)

        def features(self, uid, ids):
            return np.zeros((len(ids), 3))

    monkeypatch.setattr(study, "BPRFeatures", BPR)
    calls = []

    def fitted(training, target, variant):
        assert [w.diagnostics["holdout_end"] for w in training] == [
            "2020-08-11",
            "2020-08-18",
            "2020-08-25",
        ]
        calls.append((target.rows[7].features.shape[1], variant))
        return {7: [1, 2], 8: []}, {"selected_rounds": 7}

    monkeypatch.setattr(study, "fit_predict", fitted)
    tree = []

    def old_tree(training, target, expanded):
        assert expanded
        tree.append(target.rows[7].features.shape[1])
        return {7: [1, 2], 8: []}, 5

    monkeypatch.setattr(study, "tree_predictions", old_tree)
    r = study.compare(tmp_path, [date(2020, 9, 1)])["results"][0]
    assert calls == [
        (24, "loss_selection"),
        (24, "sampled_binary"),
        (27, "loss_selection"),
        (27, "sampled_binary"),
    ]
    assert tree == [24, 27]
    assert built == [date(2020, 8, d) for d in (11, 18, 25)] + [date(2020, 9, 1)]
    assert asofs == [date(2020, 8, d) for d in (4, 11, 18, 25)]
    assert set(r["metrics"]) == {
        "previous_loss",
        "full_map",
        "sampled",
        "bpr_loss",
        "bpr_map",
        "bpr_sampled",
    }
    assert all(v["map_at_12"] == 0.25 and v["users_evaluated"] == 2 for v in r["metrics"].values())


def test_dependencies_fail_before_reading_inputs(tmp_path, monkeypatch):
    def absent():
        raise ValueError("optional packages unavailable")

    monkeypatch.setattr(study, "dependencies", absent)
    with pytest.raises(ValueError, match="optional"):
        study.compare(tmp_path, [date(2020, 9, 1)])


def test_cli_refuses_output_under_input_directory(tmp_path):
    with pytest.raises(SystemExit):
        study.main(
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
