"""Offline candidate ablation; never promotes a model or opens the serving index.

Use --rank to compare freshly fitted v1 and matched-capacity tree rankers.
Tree fitting needs optional LightGBM (e.g. uv run --with lightgbm ...).
All reported weeks are retrospective unless independently reserved beforehand.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from clothes_recommender.catalog import CustomerData, ProductCatalog
from clothes_recommender.recommenders.collaborative import CollaborativeRecommender
from clothes_recommender.recommenders.learned import (
    FEATURES,
    PROVENANCE_FEATURES,
    LearnedRecommender,
    fit,
)
from clothes_recommender.recommenders.popularity import PopularityRecommender
from clothes_recommender.recommenders.repeat import RepeatPurchaseRecommender
from scripts.evaluate import TOP_K, chronological_split, paired_map_gain, ranking_metrics
from scripts.fetch_photos import atomic_write
from scripts.retrieval_channels import ComplementaryRetrieval

CHANNELS = ("family", "recent_basket", "latent")


@dataclass
class CandidateRows:
    ids: list[int]
    features: np.ndarray
    base_count: int


@dataclass
class Window:
    rows: dict[int, CandidateRows]
    truth: dict[int, set[int]]
    diagnostics: dict


def feature_rows(ranker, channels, user_id: int) -> CandidateRows:
    base_ids, _ = ranker.candidates(user_id)
    sources = channels.channels(user_id, top_k=50)
    extras = [item.product_id for name in CHANNELS for item in sources[name]]
    ids, base = ranker.candidates(user_id, extra_ids=extras)
    scores = channels.scores(user_id, ids)
    columns = [base]
    for name in CHANNELS:
        ranks = {item.product_id: i for i, item in enumerate(sources[name], 1)}
        values = np.asarray(scores[name], dtype=float)
        if values.shape != (len(ids),) or not np.isfinite(values).all():
            raise ValueError("Channel scores must be finite and align with candidate IDs")
        columns.append(
            np.array(
                [
                    [float(pid in ranks), 1 / ranks[pid] if pid in ranks else 0, values[i]]
                    for i, pid in enumerate(ids)
                ]
            ).reshape(-1, 3)
        )
    assert ids[: len(base_ids)] == base_ids
    return CandidateRows(ids, np.hstack(columns), len(base_ids))


def retrieval_diagnostics(rows: dict, truth: dict) -> dict:
    users = sorted(rows)
    total = sum(len(truth[u]) for u in users)
    result = {"users": len(users), "truth_pairs": total, "sources": {}}
    for name, column in [
        ("base", None),
        *[(n, len(PROVENANCE_FEATURES) + 3 * i) for i, n in enumerate(CHANNELS)],
        ("union", -1),
    ]:
        reachable = additional = customer_hits = 0
        oracle = macro_recall = 0.0
        sizes = []
        for uid in users:
            row = rows[uid]
            base = set(row.ids[: row.base_count])
            selected = (
                base
                if column is None
                else (
                    set(row.ids)
                    if column == -1
                    else {
                        pid
                        for pid, values in zip(row.ids, row.features, strict=True)
                        if values[column]
                    }
                )
            )
            hits = truth[uid] & selected
            reachable += len(hits)
            additional += len(hits - base)
            customer_hits += bool(hits)
            macro_recall += len(hits) / len(truth[uid])
            oracle += min(TOP_K, len(hits)) / min(TOP_K, len(truth[uid]))
            sizes.append(len(selected))
        result["sources"][name] = {
            "reachable_pairs": reachable,
            "additional_pairs_over_base": additional,
            "micro_recall": reachable / total if total else None,
            "macro_recall": macro_recall / len(users) if users else None,
            "users_with_reachable_truth": customer_hits,
            "oracle_map_at_12": oracle / len(users) if users else None,
            "candidates_min_median_max": [min(sizes), float(np.median(sizes)), max(sizes)]
            if sizes
            else [],
        }
    return result


def build_window(products, users, purchases, crowd, end: date) -> Window:
    cutoff, history, holdout = chronological_split(purchases, end)
    truth = holdout.groupby("user_id")["product_id"].agg(set).to_dict()
    eligible = sorted(set(history["user_id"]) & truth.keys())
    if not eligible:
        raise ValueError("Each comparison window needs past and holdout purchases")
    customers = CustomerData(users, history)
    past_crowd = crowd[pd.to_datetime(crowd["purchase_date"]) < cutoff]
    catalog = ProductCatalog(products)
    popular = PopularityRecommender(past_crowd)
    cf = CollaborativeRecommender(customers, catalog, past_crowd, popular)
    as_of = (cutoff - pd.Timedelta(days=1)).date()
    ranker = LearnedRecommender(
        customers,
        catalog,
        cf,
        popular,
        np.zeros(len(PROVENANCE_FEATURES)),
        as_of,
        feature_version=2,
    )
    started = perf_counter()
    channels = ComplementaryRetrieval(products, customers, past_crowd, as_of)
    fit_seconds = perf_counter() - started
    rows, durations = {}, []
    for uid in eligible:
        started = perf_counter()
        rows[uid] = feature_rows(ranker, channels, uid)
        durations.append(perf_counter() - started)
    diagnostics = retrieval_diagnostics(rows, truth)
    diagnostics.update(
        {
            "holdout_end": end.isoformat(),
            "cutoff": cutoff.date().isoformat(),
            "channel_fit_seconds": fit_seconds,
            "candidate_seconds_p50_p95": np.quantile(durations, [0.5, 0.95]).tolist(),
        }
    )
    return Window(rows, truth, diagnostics)


def arrays(windows: list[Window], expanded: bool) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for window in windows:
        for uid, row in window.rows.items():
            size = len(row.ids) if expanded else row.base_count
            xs.append(row.features[:size])
            ys.extend(int(pid in window.truth[uid]) for pid in row.ids[:size])
    return np.vstack(xs), np.asarray(ys)


def predictions(window: Window, scores: np.ndarray, expanded: bool) -> dict:
    result, start = {}, 0
    for uid, row in window.rows.items():
        size = len(row.ids) if expanded else row.base_count
        ids = row.ids[:size]
        values = scores[start : start + size]
        result[uid] = [ids[i] for i in np.lexsort((ids, -values))[:TOP_K]]
        start += size
    if start != len(scores):
        raise ValueError("Prediction scores do not align with candidate rows")
    return result


def _lightgbm():
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise ValueError("--rank needs optional LightGBM; use uv run --with lightgbm") from error
    return lgb


def tree_predictions(training: list[Window], target: Window, expanded: bool) -> tuple[dict, int]:
    lgb = _lightgbm()
    x, y = arrays(training[:2], expanded)
    vx, vy = arrays(training[2:], expanded)
    if set(np.unique(y)) != {0, 1} or set(np.unique(vy)) != {0, 1}:
        raise ValueError("Tree training and temporal validation need both label classes")
    params = {
        "objective": "binary",
        "metric": "None",
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
    validation = training[2]

    def metric(scores, _):
        pred = predictions(validation, scores, expanded)
        return (
            "full_truth_map12",
            ranking_metrics(validation.truth, pred, sorted(pred))["map_at_12"],
            True,
        )

    model = lgb.train(
        params,
        lgb.Dataset(x, label=y),
        num_boost_round=400,
        valid_sets=[lgb.Dataset(vx, label=vy)],
        feval=metric,
        callbacks=[lgb.early_stopping(30, first_metric_only=True, verbose=False)],
    )
    rounds = model.best_iteration
    del x, y, vx, vy, model
    x, y = arrays(training, expanded)
    model = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=rounds)
    tx, _ = arrays([target], expanded)
    return predictions(target, model.predict(tx), expanded), rounds


def compare(data_dir: Path, holdout_end: date, *, rank: bool = False) -> dict:
    if rank:
        _lightgbm()  # Fail before expensive data/channel work if the optional package is absent.
    products = pd.read_parquet(data_dir / "products.parquet")
    if "product_code" not in products:
        raise ValueError("Prepare enriched products with product_code before this experiment")
    users = pd.read_parquet(data_dir / "users.parquet")
    purchases = pd.read_parquet(data_dir / "purchases.parquet")
    crowd = pd.read_parquet(data_dir / "shopper_purchases.parquet")
    target = build_window(products, users, purchases, crowd, holdout_end)
    result = {
        "scope": "Retrospective experimental comparison; no serving promotion. "
        "Prepared population/catalog selection may use the full observation window.",
        "retrieval": target.diagnostics,
        "ranking": None,
    }
    if rank:
        training = [
            build_window(products, users, purchases, crowd, holdout_end - timedelta(days=d))
            for d in (21, 14, 7)
        ]
        x, y = arrays(training, False)
        weights = fit(x[:, : len(FEATURES)], y)
        tx, _ = arrays([target], False)
        pred = {"v1": predictions(target, tx[:, : len(FEATURES)] @ weights, False)}
        del x, tx
        rounds = {}
        for name, expanded in (("tree_base", False), ("tree_expanded", True)):
            pred[name], rounds[name] = tree_predictions(training, target, expanded)
        eligible = sorted(target.rows)
        cutoff, history, _ = chronological_split(purchases, holdout_end)
        popular = PopularityRecommender(crowd[pd.to_datetime(crowd["purchase_date"]) < cutoff])
        repeat = RepeatPurchaseRecommender(CustomerData(users, history), popular)
        pred["repeat"] = {
            uid: [item.product_id for item in repeat.recommend(uid, TOP_K)] for uid in eligible
        }
        result["ranking"] = {
            "metrics": {
                name: ranking_metrics(target.truth, values, eligible)
                for name, values in pred.items()
            },
            "paired_vs_v1": {
                name: paired_map_gain(target.truth, pred[name], pred["v1"], eligible)
                for name in rounds
            },
            "paired_expanded_vs_base": paired_map_gain(
                target.truth, pred["tree_expanded"], pred["tree_base"], eligible
            ),
            "selected_rounds": rounds,
            "protocol": "400-round cap; temporal early stopping on third prior training week, "
            "then refit all three prior weeks. Same tree capacity/search budget; "
            "iteration counts selected separately. No target labels used for fitting.",
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--holdout-end", type=date.fromisoformat, required=True)
    parser.add_argument("--rank", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.resolve().is_relative_to(args.data_dir.resolve()):
        parser.error("Write experiment reports outside the prepared dataset directory")
    files = [
        args.data_dir / f"{name}.parquet"
        for name in ("products", "users", "purchases", "shopper_purchases")
    ]

    def hashes():
        result = {}
        for path in files:
            with path.open("rb") as file:
                result[path.name] = hashlib.file_digest(file, "sha256").hexdigest()
        return result

    try:
        before = hashes()
        result = compare(args.data_dir, args.holdout_end, rank=args.rank)
        if before != hashes():
            raise ValueError("Prepared inputs changed during experiment")
        result["input_sha256"] = before
        args.output.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(result, indent=2)
        atomic_write(args.output, content.encode())
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(content)


if __name__ == "__main__":
    main()
