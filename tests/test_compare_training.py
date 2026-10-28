from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from scripts import compare_training as study
from scripts.compare_retrieval import CandidateRows, Window


def snapshot(ids, end):
    return Window(
        {uid: CandidateRows([1, 2], np.array([[1.0, 0], [0.0, 1]]), 2) for uid in ids},
        {uid: {1, 999} for uid in ids},
        {"holdout_end": end.isoformat()},
    )


def shared(tmp_path):
    training, evaluation = tmp_path / "train", tmp_path / "evaluate"
    training.mkdir()
    evaluation.mkdir()
    for directory in (training, evaluation):
        for name in ("products", "shopper_purchases"):
            (directory / f"{name}.parquet").write_bytes(b"same evidence")
    original = pd.DataFrame({"user_id": [1, 2], "shopper_id": [81, 82]})
    users = pd.concat([original, pd.DataFrame({"user_id": [3], "shopper_id": [83]})])
    events = pd.DataFrame(
        [(1, 1, "2020-08-01"), (2, 2, "2020-09-08")],
        columns=["user_id", "product_id", "purchase_date"],
    )
    return training, evaluation, users, original, events.copy(), events.copy(), date(2020, 9, 1)


def test_shared_inputs_preserve_prior_history_but_do_not_require_future_training_labels(tmp_path):
    args = list(shared(tmp_path))
    args[4] = args[4][args[4].purchase_date < "2020-09-02"]
    study.validate_shared(*args)
    args[4].loc[0, "product_id"] = 7
    with pytest.raises(ValueError, match="prior events"):
        study.validate_shared(*args)


@pytest.mark.parametrize("change", ["identity", "duplicate", "missing", "evidence"])
def test_mismatched_cohort_or_retrieval_evidence_rejected(tmp_path, change):
    args = list(shared(tmp_path))
    if change == "identity":
        args[2].loc[args[2].user_id == 1, "shopper_id"] = 80
    elif change == "duplicate":
        args[2].loc[args[2].user_id == 3, "shopper_id"] = 81
    elif change == "missing":
        args[2] = args[2][args[2].user_id != 1]
    else:
        (args[0] / "products.parquet").write_bytes(b"different catalog")
    with pytest.raises(ValueError):
        study.validate_shared(*args)


def test_training_arms_keep_common_validation_and_unchanged_full_truth(monkeypatch, tmp_path):
    training, evaluation = tmp_path / "train", tmp_path / "eval"
    original = pd.DataFrame({"user_id": [1, 2]})
    users = pd.DataFrame({"user_id": [1, 2, 3]})
    frames = {
        evaluation / "products.parquet": pd.DataFrame(),
        evaluation / "users.parquet": original,
        evaluation / "purchases.parquet": pd.DataFrame(),
        evaluation / "shopper_purchases.parquet": pd.DataFrame(),
        training / "users.parquet": users,
        training / "purchases.parquet": pd.DataFrame(),
    }
    monkeypatch.setattr(study.pd, "read_parquet", frames.__getitem__)
    monkeypatch.setattr(study, "validate_shared", lambda *a: None)
    monkeypatch.setattr(study, "_lightgbm", lambda: None)
    monkeypatch.setattr(study, "build_window", lambda p, u, h, c, end: snapshot(u.user_id, end))
    end = date(2020, 9, 8)
    calls = []

    def fit(history, target, variant, validation_window):
        assert set(target.rows) == set(validation_window.rows) == {1, 2}
        assert target.truth == {1: {1, 999}, 2: {1, 999}}
        assert validation_window.diagnostics["holdout_end"] == "2020-09-01"
        count = len(history)
        assert [w.diagnostics["holdout_end"] for w in history] == [
            (end - timedelta(days=7 * i)).isoformat() for i in range(count, 0, -1)
        ]
        calls.append((count, set(history[0].rows)))
        return {1: [1, 2], 2: [1, 2]}, {}

    monkeypatch.setattr(study, "fit_predict", fit)
    result = study.compare(training, evaluation, [end])
    assert calls == [(3, {1, 2}), (5, {1, 2}), (3, {1, 2, 3}), (5, {1, 2, 3})]
    for metrics in result["results"][0]["metrics"].values():
        assert metrics["users_evaluated"] == 2
        assert metrics["map_at_12"] == 0.5
        assert metrics["recall_at_12"] == 0.5


def test_cli_protects_both_inputs_and_checks_dependency_before_reads(monkeypatch, tmp_path):
    base = [
        "--training-data-dir",
        str(tmp_path / "train"),
        "--evaluation-data-dir",
        str(tmp_path / "eval"),
        "--holdout-end",
        "2020-09-01",
        "--output",
    ]
    for name in ("train", "eval"):
        with pytest.raises(SystemExit):
            study.main([*base, str(tmp_path / name / "users.parquet")])

    def dependency():
        raise ValueError("install optional LightGBM")

    monkeypatch.setattr(study, "_lightgbm", dependency)
    monkeypatch.setattr(
        study, "digest", lambda p: pytest.fail("Input read before dependency check")
    )
    with pytest.raises(SystemExit):
        study.main([*base, str(tmp_path / "report.json")])
    assert not list(tmp_path.iterdir())
