"""Fetch photos of H&M garments that are not in the prepared catalog, to try image search.

Usage: uv run python -m scripts.fetch_query_photos --articles data/raw/articles.csv
    [--data-dir data] [--count 12] [--seed 7]

The photos come from the same pinned mirror as scripts/fetch_photos.py, but none of them
is indexed, so every image-search result is a different product that merely looks alike.
Writes data/query_photos/{article_id}.jpg plus query_photos.json with each photo's label.
"""

import argparse
import json
import logging
from collections.abc import Sequence
from itertools import zip_longest
from pathlib import Path

import httpx
import pandas as pd

from clothes_recommender.config import get_settings
from scripts.fetch_photos import atomic_write, download_photo

# Product groups whose photos show one recognisable garment; accessories and
# "Unknown" are mostly flat-lay close-ups that make poor queries.
QUERY_GROUPS = (
    "Garment Upper body",
    "Garment Lower body",
    "Garment Full body",
    "Shoes",
    "Swimwear",
    "Underwear",
    "Nightwear",
)

logger = logging.getLogger(__name__)


def candidate_articles(articles: pd.DataFrame, catalog_ids: set[int], seed: int) -> list[dict]:
    """Articles outside the catalog, shuffled, interleaved across QUERY_GROUPS."""
    outside = articles[
        ~articles["article_id"].isin(catalog_ids)
        & articles["product_group_name"].isin(QUERY_GROUPS)
    ]
    shuffled = outside.sample(frac=1.0, random_state=seed)
    groups = [
        shuffled[shuffled["product_group_name"] == group].to_dict(orient="records")
        for group in QUERY_GROUPS
    ]
    return [row for round_ in zip_longest(*groups) for row in round_ if row is not None]


def fetch_query_photos(articles_path: Path, data_dir: Path, count: int, seed: int) -> list[dict]:
    if count <= 0:
        raise ValueError("count must be positive")
    articles = pd.read_csv(
        articles_path,
        usecols=[
            "article_id",
            "prod_name",
            "product_type_name",
            "product_group_name",
            "colour_group_name",
        ],
        dtype={"article_id": "int64"},
    )
    catalog_ids = set(
        pd.read_parquet(data_dir / "products.parquet", columns=["product_id"])["product_id"]
    )
    out_dir = data_dir / "query_photos"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[dict] = []
    with httpx.Client(timeout=httpx.Timeout(30, connect=10), follow_redirects=True) as client:
        # Some articles have no photo in the mirror; keep trying until `count` succeed.
        for row in candidate_articles(articles, catalog_ids, seed):
            if len(saved) == count:
                break
            result = download_photo(client, int(row["article_id"]), out_dir)
            if result["status"] == "failed":
                logger.info("Skipping %s: %s", row["article_id"], result["error"])
                continue
            saved.append(
                {
                    "photo_id": int(row["article_id"]),
                    "name": row["prod_name"],
                    "product_type": row["product_type_name"],
                    "colour": row["colour_group_name"],
                }
            )
    atomic_write(out_dir / "query_photos.json", json.dumps(saved, indent=2).encode())
    return saved


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--articles", type=Path, required=True, help="H&M articles.csv")
    parser.add_argument("--data-dir", type=Path, default=get_settings().data_dir)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    logger.setLevel(logging.INFO)
    saved = fetch_query_photos(args.articles, args.data_dir, args.count, args.seed)
    print(f"Saved {len(saved)} query photos to {args.data_dir / 'query_photos'}")


if __name__ == "__main__":
    main()
