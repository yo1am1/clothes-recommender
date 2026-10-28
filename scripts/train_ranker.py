"""Train the fixed linear ranker on three earlier seven-day windows, using no future signals.

Usage: uv run python -m scripts.train_ranker --data-dir data --train-through 2020-09-15
The default leaves the dataset's last seven days out. Saves JSON, never executable pickle.
"""

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from clothes_recommender.catalog import CustomerData, ProductCatalog, load_shopper_purchases
from clothes_recommender.recommenders.collaborative import CollaborativeRecommender
from clothes_recommender.recommenders.learned import (
    LearnedRecommender,
    RankerModel,
    feature_names,
    fit,
)
from clothes_recommender.recommenders.popularity import PopularityRecommender
from scripts.evaluate import chronological_split
from scripts.fetch_photos import atomic_write


def train(data_dir: Path, through: date | None = None, *, feature_version: int = 1) -> RankerModel:
    names = feature_names(feature_version)
    catalog = ProductCatalog.from_parquet(data_dir / "products.parquet")
    users = pd.read_parquet(data_dir / "users.parquet")
    purchases = pd.read_parquet(data_dir / "purchases.parquet")
    last = pd.to_datetime(purchases["purchase_date"]).max()
    if pd.isna(last):
        raise ValueError("Prepare at least four weeks of purchases to train the ranker")
    through = through or (last.date() - timedelta(days=7))
    if through > last.date():
        raise ValueError("Training end exceeds the observed purchase history")
    crowd = load_shopper_purchases(
        data_dir / "shopper_purchases.parquet", CustomerData(users, purchases)
    )
    crowd["purchase_date"] = pd.to_datetime(crowd["purchase_date"])
    feature_rows, labels, ends = [], [], []
    for offset in (14, 7, 0):
        end = through - timedelta(days=offset)
        cutoff, history, holdout = chronological_split(purchases, end)
        customers = CustomerData(users, history)
        shoppers = crowd[crowd["purchase_date"] < cutoff]
        popular = PopularityRecommender(shoppers)
        collaborative = CollaborativeRecommender(customers, catalog, shoppers, popular)
        ranker = LearnedRecommender(
            customers,
            catalog,
            collaborative,
            popular,
            np.zeros(len(names)),
            (cutoff - pd.Timedelta(days=1)).date(),
            feature_version=feature_version,
        )
        truth = holdout.groupby("user_id")["product_id"].agg(set).to_dict()
        eligible = sorted(set(history["user_id"]) & truth.keys())
        if not eligible:
            raise ValueError(
                "Each ranker training window needs customers with past and future purchases"
            )
        for user_id in eligible:
            ids, features = ranker.candidates(user_id)
            feature_rows.append(features)
            labels.extend(int(pid in truth[user_id]) for pid in ids)
        ends.append(end)
    labels = np.asarray(labels)
    weights = fit(np.vstack(feature_rows), labels, feature_version=feature_version)
    return RankerModel(
        feature_version=feature_version,
        weights=weights.tolist(),
        trained_through=through,
        training_window_ends=ends,
        training_rows=len(labels),
        training_positives=int(labels.sum()),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-through", type=date.fromisoformat)
    parser.add_argument("--output", type=Path, help="Defaults to <data-dir>/ranker.json")
    parser.add_argument("--feature-version", type=int, choices=(1, 2), default=1)
    args = parser.parse_args(argv)
    if args.feature_version == 2 and args.output is None:
        parser.error("--output is required for experimental feature version 2")
    try:
        model = train(args.data_dir, args.train_through, feature_version=args.feature_version)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    output = args.output or args.data_dir / "ranker.json"
    atomic_write(output, model.model_dump_json(indent=2).encode())
    print(
        f"Ranker saved to {output}; trained through {model.trained_through}, "
        f"{model.training_positives} positive candidates / {model.training_rows} rows."
    )


if __name__ == "__main__":
    main()
