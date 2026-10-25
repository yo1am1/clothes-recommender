"""Evaluate prepared purchases with a global last-seven-days holdout; print JSON.

Usage: uv run python -m scripts.evaluate --data-dir data [--qdrant-path .qdrant]
    [--output data/evaluation.json]

With --output the JSON is also saved; the API serves that file at GET /evaluation.
"""

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient

from clothes_recommender.catalog import CustomerData, ProductCatalog
from clothes_recommender.config import get_settings
from clothes_recommender.recommenders.collaborative import CollaborativeRecommender
from clothes_recommender.recommenders.content_based import ContentBasedRecommender
from clothes_recommender.recommenders.popularity import PopularityRecommender
from clothes_recommender.services import hybrid_recommender
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


def chronological_split(purchases: pd.DataFrame) -> tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    """Hold out the latest seven calendar days, including the dataset's last day."""
    if purchases.empty:
        raise ValueError(
            "No purchases to evaluate; prepare data with more than seven days of history"
        )
    if purchases[["user_id", "product_id", "purchase_date"]].isna().any().any():
        raise ValueError("Purchases must have user_id, product_id and purchase_date")
    purchases = purchases.assign(purchase_date=pd.to_datetime(purchases["purchase_date"]))
    cutoff = purchases["purchase_date"].max().normalize() - pd.Timedelta(days=6)
    train = purchases[purchases["purchase_date"] < cutoff].copy()
    holdout = purchases[purchases["purchase_date"] >= cutoff].copy()
    if train.empty:
        raise ValueError(
            "No training purchases before cutoff; prepare more than seven days of history"
        )
    return cutoff, train, holdout


def evaluate(data_dir: Path, qdrant_path: Path | None = None) -> dict:
    """Evaluate prepared users with training and holdout purchases; download no models."""
    from clothes_recommender.recommenders.graph import GraphRecommender

    data_dir = Path(data_dir)
    catalog = ProductCatalog.from_parquet(data_dir / "products.parquet")
    users = pd.read_parquet(data_dir / "users.parquet")
    purchases = pd.read_parquet(data_dir / "purchases.parquet")
    cutoff, train, holdout = chronological_split(purchases)
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
    bought = train.groupby("user_id")["product_id"].agg(set).to_dict()
    truth = holdout.groupby("user_id")["product_id"].agg(set).to_dict()
    novel = {uid: truth[uid] - bought[uid] for uid in eligible}
    novel_users = [uid for uid in eligible if novel[uid]]
    links = co_purchases(
        train.rename(
            columns={"user_id": "customer_id", "purchase_date": "t_dat", "product_id": "article_id"}
        ),
        sorted(catalog_ids),
    )
    popular = PopularityRecommender(customers)
    recommenders = {
        "collaborative": CollaborativeRecommender(customers, popular),
        "graph": GraphRecommender(customers, catalog, popular, links),
    }
    result = {
        "cutoff": cutoff.date().isoformat(),
        "holdout_end": pd.to_datetime(holdout["purchase_date"]).max().date().isoformat(),
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
            "training_co_purchase_links": len(links),
            "co_purchase_source": "training prepared purchases only; sampled users only",
        },
        "repeat_purchase_limitation": {
            "description": "Training purchases are excluded from recommendations. "
            "Full holdout includes repeats; "
            "novel-item MAP excludes repeats and users with no novel holdout items.",
            "full_holdout_unique_user_items": sum(len(truth[uid]) for uid in eligible),
            "novel_holdout_unique_user_items": sum(len(novel[uid]) for uid in eligible),
        },
        "content_index": None,
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
            recommenders["hybrid"] = hybrid_recommender(
                recommenders["collaborative"], recommenders["content"]
            )
            result["content_index"] = {
                "path": str(qdrant_path.resolve()),
                "collection": collection,
                "indexed_products": len(indexed),
                "limitation": "The index does not record vector provenance",
            }
        for name in ["popularity", *recommenders]:
            predictions = {}
            for uid in eligible:
                recommendations = (
                    popular.top(TOP_K, exclude_ids=bought[uid])
                    if name == "popularity"
                    else recommenders[name].recommend(uid, TOP_K)
                )
                ranked = list(dict.fromkeys(item.product_id for item in recommendations))[:TOP_K]
                if not set(ranked).issubset(catalog_ids):
                    raise ValueError(f"{name} recommended products outside the prepared catalog")
                predictions[uid] = ranked
            result["results"][name] = {
                "full_holdout": {
                    "users_evaluated": len(eligible),
                    "map_at_12": sum(
                        average_precision_at_k(truth[u], predictions[u]) for u in eligible
                    )
                    / len(eligible),
                },
                "novel_item_holdout": {
                    "users_evaluated": len(novel_users),
                    "map_at_12": (
                        sum(average_precision_at_k(novel[u], predictions[u]) for u in novel_users)
                        / len(novel_users)
                        if novel_users
                        else None
                    ),
                },
                "catalog_coverage": len({p for row in predictions.values() for p in row})
                / len(catalog_ids),
                "recommendation_fill": sum(map(len, predictions.values()))
                / (len(eligible) * TOP_K),
            }
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
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.data_dir, args.qdrant_path)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
