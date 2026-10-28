"""Evaluate prepared purchases with a global last-seven-days holdout; print JSON.

Usage: uv run python -m scripts.evaluate --data-dir data [--qdrant-path .qdrant]
    [--output data/evaluation.json]

With --output the JSON is also saved; the API serves that file at GET /evaluation.
"""

import argparse
import json
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient

from clothes_recommender.catalog import CustomerData, ProductCatalog, load_shopper_purchases
from clothes_recommender.config import get_settings
from clothes_recommender.recommenders.collaborative import CollaborativeRecommender
from clothes_recommender.recommenders.content_based import ContentBasedRecommender
from clothes_recommender.recommenders.learned import LearnedRecommender, load_model
from clothes_recommender.recommenders.popularity import PopularityRecommender
from clothes_recommender.recommenders.repeat import RepeatPurchaseRecommender
from clothes_recommender.vector_store import ProductVectorStore
from scripts.prepare_data import co_purchases

TOP_K = 12


def average_precision_at_k(
    relevant: Iterable[int], predictions: Iterable[int], k: int = TOP_K
) -> float:
    """AP over unique relevant items and unique ranked predictions; empty truth scores zero."""
    if k <= 0:
        raise ValueError("k must be positive")
    truth = set(relevant)
    ranked = list(dict.fromkeys(predictions))[:k]
    hits = 0
    total = 0.0
    for rank, product_id in enumerate(ranked, 1):
        if product_id in truth:
            hits += 1
            total += hits / rank
    return total / min(k, len(truth)) if truth else 0.0


def chronological_split(
    purchases: pd.DataFrame, holdout_end: date | None = None
) -> tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    """Hold out the latest seven calendar days, including the dataset's last day."""
    if purchases.empty:
        raise ValueError(
            "No purchases to evaluate; prepare data with more than seven days of history"
        )
    if purchases[["user_id", "product_id", "purchase_date"]].isna().any().any():
        raise ValueError("Purchases must have user_id, product_id and purchase_date")
    purchases = purchases.assign(purchase_date=pd.to_datetime(purchases["purchase_date"]))
    end = pd.Timestamp(holdout_end) if holdout_end else purchases["purchase_date"].max().normalize()
    purchases = purchases[purchases["purchase_date"] < end + pd.Timedelta(days=1)]
    cutoff = end - pd.Timedelta(days=6)
    train = purchases[purchases["purchase_date"] < cutoff].copy()
    holdout = purchases[purchases["purchase_date"] >= cutoff].copy()
    if train.empty:
        raise ValueError(
            "No training purchases before cutoff; prepare more than seven days of history"
        )
    return cutoff, train, holdout


def ranking_metrics(truth: dict, predictions: dict, user_ids: list) -> dict:
    """Macro customer metrics; recall uses unique relevant products, not transaction counts."""
    size = len(user_ids)
    return {
        "users_evaluated": size,
        "map_at_12": sum(average_precision_at_k(truth[u], predictions[u]) for u in user_ids) / size
        if size
        else None,
        "hit_rate": sum(bool(truth[u] & set(predictions[u])) for u in user_ids) / size
        if size
        else None,
        "recall_at_12": sum(len(truth[u] & set(predictions[u])) / len(truth[u]) for u in user_ids)
        / size
        if size
        else None,
    }


def paired_map_gain(truth: dict, candidate: dict, baseline: dict, user_ids: list) -> dict:
    """Paired customer bootstrap over this selected cohort, not population-level uncertainty."""
    differences = np.array(
        [
            average_precision_at_k(truth[u], candidate[u])
            - average_precision_at_k(truth[u], baseline[u])
            for u in user_ids
        ]
    )
    random = np.random.default_rng(42)
    samples = [
        differences[random.integers(0, len(differences), len(differences))].mean()
        for _ in range(2000)
    ]
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "mean_delta": float(differences.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "paired_customers": len(user_ids),
        "bootstrap_resamples": 2000,
        "scope": "Selected-customer resampling; excludes selection and model-choice uncertainty",
    }


def evaluate(
    data_dir: Path,
    qdrant_path: Path | None = None,
    *,
    holdout_end: date | None = None,
    ranker_path: Path | None = None,
) -> dict:
    """Evaluate prepared users with training and holdout purchases; download no models."""
    from clothes_recommender.recommenders.graph import GraphRecommender

    data_dir = Path(data_dir)
    catalog = ProductCatalog.from_parquet(data_dir / "products.parquet")
    users = pd.read_parquet(data_dir / "users.parquet")
    purchases = pd.read_parquet(data_dir / "purchases.parquet")
    cutoff, train, holdout = chronological_split(purchases, holdout_end)
    catalog_ids = {product.product_id for product in catalog}
    if not catalog_ids or not set(purchases["product_id"]).issubset(catalog_ids):
        raise ValueError("Purchases must reference a nonempty prepared product catalog")
    if not set(purchases["user_id"]).issubset(set(users["user_id"])):
        raise ValueError(
            "Purchases reference users missing from users.parquet; prepare matching files"
        )
    eligible = sorted(set(train["user_id"]) & set(holdout["user_id"]))
    if not eligible:
        raise ValueError(
            "No users have both training and holdout purchases; prepare a longer history "
            "or a larger customer sample"
        )
    customers = CustomerData(users, train)
    # Crowd signals (popularity, collaborative filtering, co-purchase links) come from every
    # shopper's purchases before the cutoff, as in production; older prepared data without
    # shopper_purchases.parquet falls back to the sampled customers.
    shoppers = load_shopper_purchases(data_dir / "shopper_purchases.parquet", customers)
    shoppers = shoppers[pd.to_datetime(shoppers["purchase_date"]) < cutoff]
    bought = train.groupby("user_id")["product_id"].agg(set).to_dict()
    truth = holdout.groupby("user_id")["product_id"].agg(set).to_dict()
    novel = {uid: truth[uid] - bought[uid] for uid in eligible}
    novel_users = [uid for uid in eligible if novel[uid]]
    links = co_purchases(
        shoppers.rename(
            columns={
                "shopper_id": "customer_id",
                "purchase_date": "t_dat",
                "product_id": "article_id",
            }
        ),
        sorted(catalog_ids),
    )
    popular = PopularityRecommender(shoppers)
    recommenders = {
        "repeat": RepeatPurchaseRecommender(customers, popular),
        "collaborative": CollaborativeRecommender(customers, catalog, shoppers, popular),
        "graph": GraphRecommender(customers, catalog, popular, links),
    }
    ranker_metadata = None
    if ranker_path is not None:
        model = load_model(ranker_path)
        if model.trained_through >= cutoff.date():
            raise ValueError("Ranker training must end strictly before the holdout cutoff")
        recommenders["learned"] = LearnedRecommender(
            customers,
            catalog,
            recommenders["collaborative"],
            popular,
            model.weights,
            (cutoff - pd.Timedelta(days=1)).date(),
            feature_version=model.feature_version,
        )
        ranker_metadata = model.model_dump(mode="json", exclude={"weights"})
    result = {
        "cutoff": cutoff.date().isoformat(),
        "holdout_end": (
            holdout_end or pd.to_datetime(holdout["purchase_date"]).max().date()
        ).isoformat(),
        "holdout_days": 7,
        "top_k": TOP_K,
        "users_evaluated": len(eligible),
        "sample_scope": {
            "data_dir": str(data_dir.resolve()),
            "selection": "all users with training and holdout purchases, sorted by user_id",
            "limitation": "Prepared user/catalog selection may use the full observation window; "
            "results describe this sample, not the full customer population",
            "prepared_users": len(users),
            "catalog_products": len(catalog_ids),
            "training_purchases": len(train),
            "holdout_purchases": len(holdout),
            "training_shopper_purchases": len(shoppers),
            "training_shoppers": int(shoppers["shopper_id"].nunique()),
            "training_co_purchase_links": len(links),
            "crowd_signal_source": "every shopper's catalog purchases before the cutoff; "
            "co-purchase lift counts only baskets with catalog products",
        },
        "repeat_purchase_limitation": {
            "description": "Full holdout, including repeat purchases, is the primary comparison. "
            "Popularity and repeat strategies allow past purchases; discovery strategies "
            "exclude them. Novel-item metrics use the original ranked lists; they are not "
            "a like-for-like task for repeat strategies.",
            "full_holdout_unique_user_items": sum(len(truth[uid]) for uid in eligible),
            "novel_holdout_unique_user_items": sum(len(novel[uid]) for uid in eligible),
        },
        "content_index": None,
        "ranker": ranker_metadata,
        "results": {},
    }
    store = None
    try:
        if qdrant_path is not None:
            qdrant_path = Path(qdrant_path)
            if not qdrant_path.is_dir():
                raise ValueError("--qdrant-path must name an existing local index directory")
            collection = get_settings().qdrant_collection
            store = ProductVectorStore(QdrantClient(path=str(qdrant_path)), collection)
            if not store.exists():
                raise ValueError(f"Explicit index has no '{collection}' collection; build it first")
            indexed = store.indexed_ids()
            if not indexed or not indexed.issubset(catalog_ids):
                raise ValueError(
                    "Explicit index must be nonempty and contain only prepared catalog IDs"
                )
            if not indexed.intersection(train["product_id"]):
                raise ValueError(
                    "Explicit index has no training purchase vectors; index training products"
                )
            recommenders["content"] = ContentBasedRecommender(customers, catalog, store, popular)
            result["content_index"] = {
                "path": str(qdrant_path.resolve()),
                "collection": collection,
                "indexed_products": len(indexed),
                "limitation": "The index does not record vector provenance",
            }
        all_predictions = {}
        for name in ["popularity", "popularity_novel", *recommenders]:
            predictions = {}
            for uid in eligible:
                recommendations = (
                    popular.top(
                        TOP_K, exclude_ids=bought[uid] if name == "popularity_novel" else ()
                    )
                    if name in ("popularity", "popularity_novel")
                    else recommenders[name].recommend(uid, TOP_K)
                )
                ranked = list(dict.fromkeys(item.product_id for item in recommendations))[:TOP_K]
                if not set(ranked).issubset(catalog_ids):
                    raise ValueError(f"{name} recommended products outside the prepared catalog")
                predictions[uid] = ranked
            all_predictions[name] = predictions
            result["results"][name] = {
                "full_holdout": ranking_metrics(truth, predictions, eligible),
                "novel_item_holdout": ranking_metrics(novel, predictions, novel_users),
                "catalog_coverage": len({p for row in predictions.values() for p in row})
                / len(catalog_ids),
                "recommendation_fill": sum(map(len, predictions.values()))
                / (len(eligible) * TOP_K),
            }
        if "learned" in all_predictions:
            result["learned_vs_repeat"] = paired_map_gain(
                truth, all_predictions["learned"], all_predictions["repeat"], eligible
            )
    finally:
        if store is not None:
            store.close()
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=get_settings().data_dir)
    parser.add_argument(
        "--qdrant-path", type=Path, help="Explicit existing compatible local image index"
    )
    parser.add_argument("--output", type=Path, help="Also write the JSON results to this file")
    parser.add_argument(
        "--holdout-end",
        type=date.fromisoformat,
        help="Last date of the seven-day holdout; permits earlier validation weeks",
    )
    parser.add_argument("--ranker", type=Path, help="Explicit past-trained JSON ranker artifact")
    args = parser.parse_args(argv)
    try:
        result = evaluate(
            args.data_dir, args.qdrant_path, holdout_end=args.holdout_end, ranker_path=args.ranker
        )
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
