"""Bounded offline ranking ablation on unchanged expanded candidate pools.

Needs optional LightGBM. No serving artifacts or evaluation files are updated.
"""

import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from clothes_recommender.catalog import CustomerData
from clothes_recommender.recommenders.learned import FEATURES, PROVENANCE_FEATURES, fit
from clothes_recommender.recommenders.popularity import PopularityRecommender
from clothes_recommender.recommenders.repeat import RepeatPurchaseRecommender
from scripts.compare_retrieval import (
    CHANNELS,
    CandidateRows,
    Window,
    _lightgbm,
    arrays,
    build_window,
    predictions,
    tree_predictions,
)
from scripts.evaluate import chronological_split, paired_map_gain, ranking_metrics
from scripts.fetch_photos import atomic_write
from scripts.ranking_features import FEATURE_NAMES, TemporalFeatures

VARIANTS = ("loss_selection", "context_binary", "context_lambdarank")


def enrich(window: Window, context: TemporalFeatures) -> Window:
    rows = {}
    for uid, row in window.rows.items():
        normalized = []
        for i in range(len(CHANNELS)):
            values = row.features[:, len(PROVENANCE_FEATURES) + 2 + 3 * i]
            scale = np.log1p(values.max(initial=0)) or 1.0
            normalized.append(np.log1p(values) / scale)
        extra = context.features(uid, row.ids)
        if extra.shape != (len(row.ids), len(FEATURE_NAMES)) or not np.isfinite(extra).all():
            raise ValueError("Context features must be finite and align with candidate IDs")
        features = np.column_stack([row.features, *normalized, extra])
        if not np.isfinite(features).all():
            raise ValueError("Ranking features must be finite")
        rows[uid] = CandidateRows(list(row.ids), features, row.base_count)
    return Window(rows, window.truth, window.diagnostics)


def groups(windows: list[Window]) -> list[int]:
    return [len(row.ids) for w in windows for row in w.rows.values() if row.ids]


def fit_predict(
    training: list[Window], target: Window, variant: str, *, model_path: Path | None = None
) -> tuple[dict, dict]:
    if variant not in VARIANTS:
        raise ValueError("Unknown ranking variant")
    lgb = _lightgbm()
    x, y = arrays(training[:2], True)
    vx, vy = arrays(training[2:], True)
    if set(np.unique(y)) != {0, 1} or set(np.unique(vy)) != {0, 1}:
        raise ValueError("Training and earlier validation require both label classes")
    ranking = variant == "context_lambdarank"
    params = {
        "objective": "lambdarank" if ranking else "binary",
        "metric": "None" if ranking else "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 50,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "num_threads": 2,
        "seed": 42,
        "deterministic": True,
        "force_col_wise": True,
    }
    if ranking:
        params["lambdarank_truncation_level"] = 15
    train_data = lgb.Dataset(x, label=y, **({"group": groups(training[:2])} if ranking else {}))
    valid_data = lgb.Dataset(vx, label=vy, **({"group": groups(training[2:])} if ranking else {}))
    validation = training[2]

    def metric(scores, _):
        pred = predictions(validation, scores, True)
        return (
            "full_truth_map12",
            ranking_metrics(validation.truth, pred, sorted(pred))["map_at_12"],
            True,
        )

    model = lgb.train(
        params,
        train_data,
        num_boost_round=400,
        valid_sets=[valid_data],
        feval=metric if ranking else None,
        callbacks=[lgb.early_stopping(30, first_metric_only=True, verbose=False)],
    )
    rounds = model.best_iteration
    del x, y, vx, vy, train_data, valid_data, model
    x, y = arrays(training, True)
    model = lgb.train(
        params,
        lgb.Dataset(x, label=y, **({"group": groups(training)} if ranking else {})),
        num_boost_round=rounds,
    )
    tx, _ = arrays([target], True)
    pred = predictions(target, model.predict(tx), True)
    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_path))
    return pred, {
        "selected_rounds": rounds,
        "features": x.shape[1],
        "objective": params["objective"],
        "selection_metric": "full_truth_map12" if ranking else "binary_logloss",
    }


def compare(data_dir: Path, ends: list[date]) -> dict:
    _lightgbm()
    products = pd.read_parquet(data_dir / "products.parquet")
    users = pd.read_parquet(data_dir / "users.parquet")
    purchases = pd.read_parquet(data_dir / "purchases.parquet")
    crowd = pd.read_parquet(data_dir / "shopper_purchases.parquet")
    required = sorted({end - timedelta(days=d) for end in ends for d in (0, 7, 14, 21)})
    plain, rich = {}, {}
    for end in required:
        plain[end] = build_window(products, users, purchases, crowd, end)
        context = TemporalFeatures(products, purchases, crowd, end - timedelta(days=7))
        rich[end] = enrich(plain[end], context)
    report = {
        "scope": (
            "Previously inspected development weeks; fixed users/candidates/full truth; "
            "no fresh confirmation or serving promotion"
        ),
        "protocol": (
            "Three prior labeled weeks; first two fit, third selects rounds, all three refit. "
            "Fixed 400-round cap, patience30, seed42, 31leaves/depth6/minleaf50/lr.05/L2=1. "
            "No target labels fit or select models."
        ),
        "context_features": list(FEATURE_NAMES),
        "results": [],
    }
    for end in ends:
        keys = [end - timedelta(days=d) for d in (21, 14, 7)]
        target = plain[end]
        baseline, old_rounds = tree_predictions([plain[k] for k in keys], target, True)
        pred, details = (
            {"previous_expanded": baseline},
            {"previous_expanded": {"selected_rounds": old_rounds}},
        )
        for variant in VARIANTS:
            data = plain if variant == "loss_selection" else rich
            pred[variant], details[variant] = fit_predict(
                [data[k] for k in keys], data[end], variant
            )
        eligible = sorted(target.rows)
        report["results"].append(
            {
                "holdout_end": end.isoformat(),
                "retrieval": target.diagnostics,
                "metrics": {
                    name: ranking_metrics(target.truth, values, eligible)
                    for name, values in pred.items()
                },
                "paired_vs_previous_expanded": {
                    name: paired_map_gain(target.truth, pred[name], baseline, eligible)
                    for name in VARIANTS
                },
                "models": details,
            }
        )
    return report


def confirm(data_dir: Path, end: date, *, model_path: Path | None = None) -> dict:
    """Score only the frozen simpler challenger; caller must reserve the cohort before use."""
    _lightgbm()
    products = pd.read_parquet(data_dir / "products.parquet")
    users = pd.read_parquet(data_dir / "users.parquet")
    purchases = pd.read_parquet(data_dir / "purchases.parquet")
    crowd = pd.read_parquet(data_dir / "shopper_purchases.parquet")
    training = [
        build_window(products, users, purchases, crowd, end - timedelta(days=d))
        for d in (21, 14, 7)
    ]
    target = build_window(products, users, purchases, crowd, end)
    x, y = arrays(training, False)

    weights = fit(x[:, : len(FEATURES)], y)
    tx, _ = arrays([target], False)
    pred = {"v1": predictions(target, tx[:, : len(FEATURES)] @ weights, False)}
    pred["challenger"], details = fit_predict(
        training, target, "loss_selection", model_path=model_path
    )
    eligible = sorted(target.rows)

    cutoff, history, _ = chronological_split(purchases, end)
    popular = PopularityRecommender(crowd[pd.to_datetime(crowd["purchase_date"]) < cutoff])
    repeat = RepeatPurchaseRecommender(CustomerData(users, history), popular)
    pred["repeat"] = {uid: [p.product_id for p in repeat.recommend(uid, 12)] for uid in eligible}
    metrics = {
        name: ranking_metrics(target.truth, values, eligible) for name, values in pred.items()
    }
    paired = paired_map_gain(target.truth, pred["challenger"], pred["v1"], eligible)
    base, challenger = metrics["v1"], metrics["challenger"]
    relative = challenger["map_at_12"] / base["map_at_12"] - 1 if base["map_at_12"] else None
    cohort_slices = {}
    sections = products.set_index("product_id")["section"]
    ranked = (
        history.assign(section=history.product_id.map(sections))
        .groupby(["user_id", "section"])
        .size()
        .reset_index(name="count")
        .sort_values(["user_id", "count", "section"], ascending=[True, False, True])
        .drop_duplicates("user_id")
        .set_index("user_id")["section"]
        .to_dict()
    )
    counts = history.groupby("user_id").size()
    last = history.groupby("user_id")["purchase_date"].max()
    for uid in eligible:
        labels = [
            f"section:{ranked.get(uid, 'unknown')}",
            "history:5-9" if counts[uid] < 10 else "history:10+",
            "activity:within7days"
            if (cutoff - pd.Timedelta(days=1) - last[uid]).days <= 7
            else "activity:older",
        ]
        for label in labels:
            cohort_slices.setdefault(label, []).append(uid)
    slices, flags = {}, []
    for label, members in sorted(cohort_slices.items()):
        a = ranking_metrics(target.truth, pred["v1"], members)
        b = ranking_metrics(target.truth, pred["challenger"], members)
        change = b["map_at_12"] / a["map_at_12"] - 1 if a["map_at_12"] else None
        slices[label] = {
            "users": len(members),
            "v1": a,
            "challenger": b,
            "relative_map_gain": change,
        }
        if len(members) >= 50 and change is not None and change < -0.1:
            flags.append(label)
    practical = (
        relative is not None
        and relative >= 0.05
        and challenger["hit_rate"] >= base["hit_rate"]
        and challenger["recall_at_12"] >= base["recall_at_12"]
    )
    return {
        "scope": "Frozen single challenger. Independent confirmation only if the caller "
        "reserved this cohort before outcome inspection; "
        "fixed historical catalog/time scope.",
        "holdout_end": end.isoformat(),
        "prepared_users": len(users),
        "retrieval": target.diagnostics,
        "model": details,
        "metrics": metrics,
        "paired_vs_v1": paired,
        "relative_map_gain": relative,
        "practical_gate_passed": bool(practical),
        "positive_paired_interval": paired["lower_95"] > 0,
        "material_slice_warnings": flags,
        "slices": slices,
        "serving_status": "Not deployed; integration and operational validation are separate.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--holdout-end", action="append", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Test only the frozen loss-selection challenger on a reserved cohort.",
    )
    args = parser.parse_args(argv)
    if args.confirm and len(args.holdout_end) != 1:
        parser.error("--confirm requires exactly one reserved target week")
    if args.output.resolve().is_relative_to(args.data_dir.resolve()):
        parser.error("Write reports outside the prepared dataset directory")

    def hashes():
        result = {}
        for name in ("products", "users", "purchases", "shopper_purchases"):
            with (args.data_dir / f"{name}.parquet").open("rb") as f:
                result[name] = hashlib.file_digest(f, "sha256").hexdigest()
        return result

    try:
        _lightgbm()
        before = hashes()
        result = (
            confirm(args.data_dir, args.holdout_end[0])
            if args.confirm
            else compare(args.data_dir, sorted(set(args.holdout_end)))
        )
        if before != hashes():
            raise ValueError("Prepared inputs changed during comparison")
        result["input_sha256"] = before
        args.output.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(result, indent=2)
        atomic_write(args.output, content.encode())
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(content)


if __name__ == "__main__":
    main()
