"""Offline supervision-scale ablation; evaluation customers and candidate pools stay fixed."""

import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from scripts.compare_rankers import fit_predict
from scripts.compare_retrieval import Window, _lightgbm, arrays, build_window
from scripts.evaluate import paired_map_gain, ranking_metrics
from scripts.fetch_photos import atomic_write


def digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def original_rows(window: Window, original_ids: set[int]) -> Window:
    rows = {uid: row for uid, row in window.rows.items() if uid in original_ids}
    return Window(rows, {uid: window.truth[uid] for uid in rows}, window.diagnostics)


def validate_shared(training_dir, evaluation_dir, users, original, purchases, evaluated, last):
    for name, frame in [("training", users), ("evaluation", original)]:
        if not {"user_id", "shopper_id"}.issubset(frame.columns):
            raise ValueError(f"{name} users need an explicit shopper crosswalk")
        if (
            frame[["user_id", "shopper_id"]].isna().any().any()
            or frame.user_id.duplicated().any()
            or frame.shopper_id.duplicated().any()
        ):
            raise ValueError(f"{name} crosswalk must be unique and nonmissing")
    if not set(original.user_id).issubset(users.user_id):
        raise ValueError("Training profiles must preserve all original users")
    a = users.set_index("user_id").loc[original.user_id, "shopper_id"].to_numpy()
    if not (a == original.shopper_id.to_numpy()).all():
        raise ValueError("Original user IDs must preserve the same shopper identities")
    for name in ("products.parquet", "shopper_purchases.parquet"):
        if digest(training_dir / name) != digest(evaluation_dir / name):
            raise ValueError("Catalog and crowd retrieval evidence must remain identical")
    columns = list(evaluated.columns)
    stop = pd.Timestamp(last) + pd.Timedelta(days=1)
    left = purchases[purchases.user_id.isin(original.user_id)].assign(
        purchase_date=pd.to_datetime(purchases.purchase_date)
    )
    right = evaluated.assign(purchase_date=pd.to_datetime(evaluated.purchase_date))
    left = left[left.purchase_date < stop][columns].sort_values(columns).reset_index(drop=True)
    right = right[right.purchase_date < stop][columns].sort_values(columns).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
    except AssertionError as error:
        raise ValueError("Original users' prior events must remain identical") from error


def compare(training_dir: Path, evaluation_dir: Path, ends: list[date]) -> dict:
    _lightgbm()
    products = pd.read_parquet(evaluation_dir / "products.parquet")
    original = pd.read_parquet(evaluation_dir / "users.parquet")
    evaluated = pd.read_parquet(evaluation_dir / "purchases.parquet")
    crowd = pd.read_parquet(evaluation_dir / "shopper_purchases.parquet")
    users = pd.read_parquet(training_dir / "users.parquet")
    purchases = pd.read_parquet(training_dir / "purchases.parquet")
    validate_shared(
        training_dir,
        evaluation_dir,
        users,
        original,
        purchases,
        evaluated,
        max(ends) - timedelta(days=7),
    )
    required = sorted({end - timedelta(days=d) for end in ends for d in (7, 14, 21, 28, 35)})
    expanded, small = {}, {}
    for end in required:
        expanded[end] = build_window(products, users, purchases, crowd, end)
        small[end] = original_rows(expanded[end], set(original.user_id))
    results = []
    for end in ends:
        target = build_window(products, original, evaluated, crowd, end)
        validation = small[end - timedelta(days=7)]
        pred, models = {}, {}
        for name, source, weeks in (
            ("previous3", small, 3),
            ("original5", small, 5),
            ("expanded3", expanded, 3),
            ("expanded5", expanded, 5),
        ):
            history = [source[end - timedelta(days=7 * i)] for i in range(weeks, 0, -1)]
            pred[name], models[name] = fit_predict(
                history, target, "loss_selection", validation_window=validation
            )
            x, y = arrays(history, True)
            models[name].update(
                {
                    "training_rows": len(y),
                    "training_positives": int(y.sum()),
                    "user_windows": sum(len(w.rows) for w in history),
                    "validation_users": len(validation.rows),
                }
            )
            del x, y
        eligible = sorted(target.rows)
        results.append(
            {
                "holdout_end": end.isoformat(),
                "retrieval": target.diagnostics,
                "metrics": {
                    name: ranking_metrics(target.truth, values, eligible)
                    for name, values in pred.items()
                },
                "paired_vs_previous3": {
                    name: paired_map_gain(target.truth, pred[name], pred["previous3"], eligible)
                    for name in ("original5", "expanded3", "expanded5")
                },
                "models": models,
            }
        )
    return {
        "scope": "Training examples change only; evaluation customers/full truth/candidate pools "
        "and latest validation cohort remain original and fixed; no promotion",
        "original_users": len(original),
        "training_users": len(users),
        "results": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-data-dir", type=Path, required=True)
    parser.add_argument("--evaluation-data-dir", type=Path, required=True)
    parser.add_argument("--holdout-end", action="append", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(
        args.output.resolve().is_relative_to(p.resolve())
        for p in (args.training_data_dir, args.evaluation_data_dir)
    ):
        parser.error("Write reports outside both input datasets")

    def hashes():
        return {
            f"{role}/{name}": digest(folder / f"{name}.parquet")
            for role, folder in [
                ("training", args.training_data_dir),
                ("evaluation", args.evaluation_data_dir),
            ]
            for name in ("products", "users", "purchases", "shopper_purchases")
        }

    try:
        _lightgbm()
        before = hashes()
        result = compare(
            args.training_data_dir, args.evaluation_data_dir, sorted(set(args.holdout_end))
        )
        if before != hashes():
            raise ValueError("Input datasets changed during comparison")
        result["input_sha256"] = before
        args.output.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(result, indent=2)
        atomic_write(args.output, content.encode())
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(content)


if __name__ == "__main__":
    main()
