"""Build the demo dataset from the H&M Personalized Fashion Recommendations Kaggle archive.

Download the archive once (needs a Kaggle account and accepted competition rules):
    uvx kaggle competitions download -c h-and-m-personalized-fashion-recommendations -p data/raw

Then:
    uv run python -m scripts.prepare_data

`--archive` also accepts a directory holding the unzipped CSVs, such as the files of the
Hugging Face mirror that scripts/fetch_photos.py downloads photos from. Without an
`images/` folder next to the CSVs, photos are not checked or extracted here: run
`scripts.fetch_photos` afterwards.

Keeps the most recent weeks of transactions, samples active customers, and writes
products/users/purchases parquet files plus downsized photos of every product they bought.
Co-purchase links between those products are counted over all shoppers, not just the
sample, so they stay dense; shopper_purchases.parquet keeps every shopper's purchases of
catalog products, under anonymous integer ids, for popularity and collaborative filtering.
Users retain the matching crowd shopper_id; event prices remain in original normalized
units and sales channels are preserved for past-only experiments.
The archive is read in place: nothing else is extracted.
"""

import argparse
import json
import logging
import zipfile
from collections.abc import Iterable, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from PIL import Image
from pyarrow import csv
from scipy.sparse import csr_array, triu
from tqdm import tqdm

from clothes_recommender.config import get_settings
from scripts.shopify_enrichment import shopify_attributes, shopify_categories

# Prices in the dataset are normalised; x590 recovers real euro price points (0.0508 -> 29.99).
# The factor comes from the Kaggle competition discussion and is an approximation.
HM_PRICE_SCALE = 590
DATASET_LAST_DAY = date(2020, 9, 22)
# Enough history per customer for both the taste vector and collaborative overlap.
MIN_PURCHASES = 5
# Pairs bought together only once are mostly noise.
MIN_CO_PURCHASES = 2
# CLIP looks at 224 px, so larger photos would only cost disk space.
PHOTO_MAX_SIZE = (384, 576)

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/h-and-m-personalized-fashion-recommendations.zip"),
        help="Kaggle competition zip (the download command above writes it here), "
        "or a directory with the unzipped CSVs.",
    )
    parser.add_argument(
        "--data-dir", type=Path, help="Output directory; defaults to configured data_dir."
    )
    parser.add_argument("--weeks", type=int, default=12, help="Weeks of transactions to keep.")
    parser.add_argument("--customers", type=int, default=5000, help="Customers to sample.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


class RawSource:
    """The Kaggle zip, or a directory with the same layout (CSVs plus optional images/)."""

    def __init__(self, path: Path) -> None:
        self._dir = path if path.is_dir() else None
        self._zip = None if self._dir else zipfile.ZipFile(path)
        self.has_photos = (
            (self._dir / "images").is_dir()
            if self._dir
            else any(name.startswith("images/") for name in self._zip.namelist())
        )

    def __enter__(self) -> "RawSource":
        return self

    def __exit__(self, *_: object) -> None:
        if self._zip is not None:
            self._zip.close()

    def open(self, member: str) -> IO[bytes]:
        return (self._dir / member).open("rb") if self._dir else self._zip.open(member)

    def members(self, names: Iterable[str]) -> set[str]:
        """The subset of `names` present in the source."""
        if self._dir:
            return {name for name in names if (self._dir / name).is_file()}
        return set(names) & set(self._zip.namelist())


def read_recent_transactions(archive: RawSource, since: date) -> pd.DataFrame:
    """Stream the 31M-row transactions file and keep rows from `since` onwards."""
    options = csv.ConvertOptions(
        include_columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
        strings_can_be_null=True,
        column_types={
            "t_dat": pa.date32(),
            "customer_id": pa.string(),
            "article_id": pa.int64(),
            "price": pa.float64(),
            "sales_channel_id": pa.int8(),
        },
    )
    batches = []
    with archive.open("transactions_train.csv") as raw:
        reader = csv.open_csv(raw, convert_options=options)
        for batch in reader:
            batches.append(batch.filter(pc.greater_equal(batch["t_dat"], pa.scalar(since))))
        table = pa.Table.from_batches(batches, schema=reader.schema)
    return table.to_pandas().astype({"customer_id": "category"})


def sample_customers(transactions: pd.DataFrame, size: int, seed: int) -> list[str]:
    counts = transactions["customer_id"].value_counts()
    eligible = np.sort(counts[counts >= MIN_PURCHASES].index.to_numpy(dtype=str))
    chosen = np.random.default_rng(seed).choice(
        eligible, size=min(size, len(eligible)), replace=False
    )
    return sorted(chosen.tolist())


def co_purchases(transactions: pd.DataFrame, article_ids: Sequence[int]) -> pd.DataFrame:
    """Product pairs bought in the same basket (one customer, one day), with count and lift.

    Lift is how much more often the pair is bought together than chance predicts, so
    best-sellers that land in every basket don't dominate the links. The denominator
    includes all recent baskets, including those containing only non-catalog products.
    """
    baskets = transactions.groupby(["customer_id", "t_dat"], observed=True).ngroup()
    in_catalog = transactions["article_id"].isin(article_ids)
    columns = pd.Categorical(
        transactions.loc[in_catalog, "article_id"], categories=article_ids
    ).codes
    contains = csr_array(
        (np.ones(in_catalog.sum()), (baskets[in_catalog].to_numpy(), columns)),
        shape=(baskets.nunique(), len(article_ids)),
    )
    contains = (contains > 0).astype(np.float64)  # buying two of an item is still one basket
    pairs = triu(contains.T @ contains, k=1).tocoo()
    keep = pairs.data >= MIN_CO_PURCHASES
    rows, cols, together = pairs.row[keep], pairs.col[keep], pairs.data[keep]
    basket_count = np.asarray(contains.sum(axis=0)).ravel()
    lift = together * contains.shape[0] / (basket_count[rows] * basket_count[cols])
    ids = np.asarray(article_ids, dtype=np.int64)
    return pd.DataFrame(
        {
            "product_id": ids[rows],
            "other_id": ids[cols],
            "count": together.astype(int),
            "lift": lift.round(3),
        }
    )


def photo_member(article_id: int) -> str:
    padded = f"{article_id:010d}"
    return f"images/{padded[:3]}/{padded}.jpg"


def save_photo(archive: RawSource, article_id: int, images_dir: Path) -> None:
    with archive.open(photo_member(article_id)) as raw, Image.open(raw) as image:
        photo = image.convert("RGB")
        photo.thumbnail(PHOTO_MAX_SIZE)
        photo.save(images_dir / f"{article_id}.jpg", quality=90)


def prepare(archive_path: Path, data_dir: Path, weeks: int, customer_count: int, seed: int) -> None:
    if weeks <= 0 or customer_count <= 0:
        raise ValueError("weeks and customer_count must be positive")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    images_dir = data_dir / "images"
    with RawSource(archive_path) as archive:
        since = DATASET_LAST_DAY - timedelta(weeks=weeks) + timedelta(days=1)
        transactions = read_recent_transactions(archive, since)
        logger.info("Kept %d transactions since %s", len(transactions), since)

        if (
            transactions[["customer_id", "article_id", "price"]].isna().any().any()
            or not np.isfinite(transactions["price"]).all()
            or (transactions["price"] <= 0).any()
        ):
            raise ValueError(
                "Recent transactions must have customer/article IDs and finite positive prices"
            )
        if not transactions["sales_channel_id"].isin([1, 2]).all():
            raise ValueError("Sales channel must be 1 (store) or 2 (online)")
        customer_ids = sample_customers(transactions, customer_count, seed)
        if not customer_ids:
            raise ValueError("No eligible customers in the requested transaction window")
        bought = transactions[transactions["customer_id"].isin(customer_ids)]

        # Some articles have no photo in the archive; they can't be embedded, so they're dropped.
        # A source without photos keeps every article; fetch_photos reports the missing ones.
        article_ids = sorted(bought["article_id"].unique().tolist())
        if archive.has_photos:
            photos = archive.members(photo_member(a) for a in article_ids)
            article_ids = [a for a in article_ids if photo_member(a) in photos]
        if not article_ids:
            raise ValueError("No photos found for products bought by sampled customers")
        bought = bought[bought["article_id"].isin(article_ids)]
        customers_with_photos = set(bought["customer_id"])
        customer_ids = [c for c in customer_ids if c in customers_with_photos]
        user_id_of = {customer_id: user_id for user_id, customer_id in enumerate(customer_ids, 1)}

        with archive.open("articles.csv") as raw:
            articles = pd.read_csv(raw, dtype={"article_id": "int64"})
        articles = articles[articles["article_id"].isin(article_ids)]
        if articles["article_id"].duplicated().any() or set(articles["article_id"]) != set(
            article_ids
        ):
            raise ValueError("articles.csv must contain exactly one row per selected product")
        median_price = transactions.groupby("article_id")["price"].median()
        products = articles.assign(
            description=lambda df: df["detail_desc"].fillna(""),
            price=lambda df: (df["article_id"].map(median_price) * HM_PRICE_SCALE).round(2),
            shopify_category=lambda df: shopify_categories(df["product_type_name"]),
            shopify_attributes=lambda df: df.apply(
                lambda row: json.dumps(shopify_attributes(row)), axis=1
            ),
        ).rename(
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
                "product_code",
                "department_name",
                "index_name",
                "name",
                "product_type",
                "product_group",
                "section",
                "colour",
                "description",
                "price",
                "shopify_category",
                "shopify_attributes",
            ]
        ]

        with archive.open("customers.csv") as raw:
            customers = pd.read_csv(raw, usecols=["customer_id", "age"], dtype={"customer_id": str})
        customers = customers[customers["customer_id"].isin(user_id_of)]
        if customers["customer_id"].duplicated().any() or set(customers["customer_id"]) != set(
            user_id_of
        ):
            raise ValueError("customers.csv must contain exactly one row per selected customer")
        # Use the same categorical codes as crowd events; sampled IDs are a separate space.
        shopper_id_of = {
            customer_id: shopper_id
            for shopper_id, customer_id in enumerate(transactions["customer_id"].cat.categories)
        }
        users = customers.assign(
            user_id=lambda df: df["customer_id"].map(user_id_of),
            age=lambda df: df["age"].astype("Int64"),
            shopper_id=lambda df: df["customer_id"].map(shopper_id_of).astype("int32"),
        ).sort_values("user_id")[["user_id", "age", "shopper_id"]]
        purchases = pd.DataFrame(
            {
                "user_id": bought["customer_id"].astype(str).map(user_id_of),
                "product_id": bought["article_id"],
                "purchase_date": bought["t_dat"],
                "price_normalized": bought["price"],
                "sales_channel_id": bought["sales_channel_id"],
            }
        )

        links = co_purchases(transactions, article_ids)
        # Every shopper's purchases of catalog products, not just the sample's: popularity
        # and item-to-item collaborative filtering need the whole crowd to be reliable.
        in_catalog = transactions[transactions["article_id"].isin(article_ids)]
        shopper_purchases = pd.DataFrame(
            {
                "shopper_id": in_catalog["customer_id"].cat.codes.astype("int32"),
                "product_id": in_catalog["article_id"],
                "purchase_date": in_catalog["t_dat"],
                "price_normalized": in_catalog["price"],
                "sales_channel_id": in_catalog["sales_channel_id"],
            }
        )
        if archive.has_photos:
            images_dir.mkdir(parents=True, exist_ok=True)
            for article_id in tqdm(article_ids, unit="photo", desc="Extracting photos"):
                save_photo(archive, article_id, images_dir)
        else:
            logger.warning("No images/ in %s: run scripts.fetch_photos next.", archive_path)

    data_dir.mkdir(parents=True, exist_ok=True)
    products.to_parquet(data_dir / "products.parquet", index=False)
    users.to_parquet(data_dir / "users.parquet", index=False)
    purchases.to_parquet(data_dir / "purchases.parquet", index=False)
    links.to_parquet(data_dir / "co_purchases.parquet", index=False)
    shopper_purchases.to_parquet(data_dir / "shopper_purchases.parquet", index=False)
    linked = len(set(links["product_id"]) | set(links["other_id"]))
    logger.info(
        "Wrote %d products, %d users, %d purchases and %d co-purchase links "
        "(%.0f%% of products have at least one) to %s",
        len(products),
        len(users),
        len(purchases),
        len(links),
        100 * linked / max(len(products), 1),
        data_dir,
    )


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    prepare(
        args.archive,
        args.data_dir or get_settings().data_dir,
        args.weeks,
        args.customers,
        args.seed,
    )


if __name__ == "__main__":
    main()
