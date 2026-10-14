"""Build the demo dataset from the H&M Personalized Fashion Recommendations Kaggle archive.

Download the archive once (needs a Kaggle account and accepted competition rules):
    uvx kaggle competitions download -c h-and-m-personalized-fashion-recommendations -p data/raw

Then:
    uv run python -m scripts.prepare_data

Keeps the most recent weeks of transactions, samples active customers, and writes
products/users/purchases parquet files plus downsized photos of every product they bought.
The archive is read in place: nothing else is extracted.
"""

import argparse
import logging
import zipfile
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from PIL import Image
from pyarrow import csv
from tqdm import tqdm

from clothes_recommender.config import get_settings

# Prices in the dataset are normalised; x590 recovers real euro price points (0.0508 -> 29.99).
# The factor comes from the Kaggle competition discussion and is an approximation.
HM_PRICE_SCALE = 590
DATASET_LAST_DAY = date(2020, 9, 22)
# Enough history per customer for both the taste vector and collaborative overlap.
MIN_PURCHASES = 5
# CLIP looks at 224 px, so larger photos would only cost disk space.
PHOTO_MAX_SIZE = (384, 576)

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/h-and-m-personalized-fashion-recommendations.zip"),
        help="Kaggle competition zip (the download command above writes it here).",
    )
    parser.add_argument("--weeks", type=int, default=12, help="Weeks of transactions to keep.")
    parser.add_argument("--customers", type=int, default=5000, help="Customers to sample.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def read_recent_transactions(archive: zipfile.ZipFile, since: date) -> pd.DataFrame:
    """Stream the 31M-row transactions file and keep rows from `since` onwards."""
    options = csv.ConvertOptions(
        include_columns=["t_dat", "customer_id", "article_id", "price"],
        column_types={"t_dat": pa.date32(), "article_id": pa.int64(), "price": pa.float64()},
    )
    batches = []
    with archive.open("transactions_train.csv") as raw:
        for batch in csv.open_csv(raw, convert_options=options):
            batches.append(batch.filter(pc.greater_equal(batch["t_dat"], pa.scalar(since))))
    table = pa.Table.from_batches(batches)
    return table.to_pandas().astype({"customer_id": "category"})


def sample_customers(transactions: pd.DataFrame, size: int, seed: int) -> list[str]:
    counts = transactions["customer_id"].value_counts()
    eligible = np.sort(counts[counts >= MIN_PURCHASES].index.to_numpy(dtype=str))
    chosen = np.random.default_rng(seed).choice(
        eligible, size=min(size, len(eligible)), replace=False
    )
    return sorted(chosen.tolist())


def photo_member(article_id: int) -> str:
    padded = f"{article_id:010d}"
    return f"images/{padded[:3]}/{padded}.jpg"


def save_photo(archive: zipfile.ZipFile, article_id: int, images_dir: Path) -> None:
    with archive.open(photo_member(article_id)) as raw, Image.open(raw) as image:
        photo = image.convert("RGB")
        photo.thumbnail(PHOTO_MAX_SIZE)
        photo.save(images_dir / f"{article_id}.jpg", quality=90)


def prepare(archive_path: Path, data_dir: Path, weeks: int, customer_count: int, seed: int) -> None:
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        since = DATASET_LAST_DAY - timedelta(weeks=weeks) + timedelta(days=1)
        transactions = read_recent_transactions(archive, since)
        logger.info("Kept %d transactions since %s", len(transactions), since)

        customer_ids = sample_customers(transactions, customer_count, seed)
        user_id_of = {customer_id: user_id for user_id, customer_id in enumerate(customer_ids, 1)}
        bought = transactions[transactions["customer_id"].isin(customer_ids)]

        # Some articles have no photo in the archive; they can't be embedded, so they're dropped.
        photos = set(archive.namelist())
        article_ids = sorted(
            a for a in bought["article_id"].unique().tolist() if photo_member(a) in photos
        )
        bought = bought[bought["article_id"].isin(article_ids)]

        with archive.open("articles.csv") as raw:
            articles = pd.read_csv(raw, dtype={"article_id": "int64"})
        median_price = transactions.groupby("article_id")["price"].median()
        products = (
            articles[articles["article_id"].isin(article_ids)]
            .assign(
                description=lambda df: df["detail_desc"].fillna(""),
                price=lambda df: (df["article_id"].map(median_price) * HM_PRICE_SCALE).round(2),
            )
            .rename(
                columns={
                    "article_id": "product_id",
                    "prod_name": "name",
                    "product_type_name": "product_type",
                    "product_group_name": "product_group",
                    "index_group_name": "section",
                    "colour_group_name": "colour",
                }
            )[
                [
                    "product_id",
                    "name",
                    "product_type",
                    "product_group",
                    "section",
                    "colour",
                    "description",
                    "price",
                ]
            ]
        )

        with archive.open("customers.csv") as raw:
            customers = pd.read_csv(raw, usecols=["customer_id", "age"])
        users = (
            customers[customers["customer_id"].isin(user_id_of)]
            .assign(
                user_id=lambda df: df["customer_id"].map(user_id_of),
                age=lambda df: df["age"].astype("Int64"),
            )
            .sort_values("user_id")[["user_id", "age"]]
        )
        purchases = pd.DataFrame(
            {
                "user_id": bought["customer_id"].astype(str).map(user_id_of),
                "product_id": bought["article_id"],
                "purchase_date": bought["t_dat"],
            }
        )

        for article_id in tqdm(article_ids, unit="photo", desc="Extracting photos"):
            save_photo(archive, article_id, images_dir)

    products.to_parquet(data_dir / "products.parquet", index=False)
    users.to_parquet(data_dir / "users.parquet", index=False)
    purchases.to_parquet(data_dir / "purchases.parquet", index=False)
    logger.info(
        "Wrote %d products, %d users, %d purchases to %s",
        len(products),
        len(users),
        len(purchases),
        data_dir,
    )


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    prepare(args.archive, get_settings().data_dir, args.weeks, args.customers, args.seed)


if __name__ == "__main__":
    main()
