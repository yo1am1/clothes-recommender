"""Download product photos, embed them with CLIP and upsert them into the vector store.

Usage: uv run python -m scripts.build_index [--limit N] [--batch-size B] [--recreate]

Re-running resumes: products already in the collection are skipped unless --recreate.
"""

import argparse
import asyncio
import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from itertools import batched, zip_longest

import httpx
from PIL import Image
from tqdm import tqdm

from clothes_recommender.catalog import ProductCatalog
from clothes_recommender.config import get_settings
from clothes_recommender.domain import Encoder, Product
from clothes_recommender.embeddings import ClipEncoder
from clothes_recommender.vector_store import ProductVectorStore

DOWNLOAD_CONCURRENCY = 32
DOWNLOAD_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.0
DOWNLOAD_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
USER_AGENT = "clothes-recommender-index-builder/0.2"

logger = logging.getLogger(__name__)


@dataclass
class BuildSummary:
    indexed: int = 0
    failed: int = 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--limit",
        type=int,
        help="Index only N products, spread evenly across categories (quick demo build).",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Images per encode batch.")
    parser.add_argument(
        "--recreate", action="store_true", help="Drop the collection and index from scratch."
    )
    return parser.parse_args(argv)


def select_demo_subset(products: Sequence[Product], limit: int) -> list[Product]:
    """Pick products round-robin across categories so even a tiny index demos every category."""
    by_category: dict[str, list[Product]] = defaultdict(list)
    for product in sorted(products, key=lambda p: p.product_id):
        by_category[product.category].append(product)
    groups = [by_category[category] for category in sorted(by_category)]
    interleaved = [p for round_ in zip_longest(*groups) for p in round_ if p is not None]
    return interleaved[:limit]


async def _fetch_bytes(client: httpx.AsyncClient, url: str) -> bytes:
    for attempt in range(DOWNLOAD_RETRIES + 1):
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.content
        except httpx.HTTPError as exc:
            if attempt == DOWNLOAD_RETRIES:
                raise
            logger.debug("Attempt %d for %s failed: %s", attempt + 1, url, exc)
            await asyncio.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise AssertionError("unreachable")


async def _download_image(
    client: httpx.AsyncClient, semaphore: asyncio.Semaphore, product: Product
) -> Image.Image | None:
    """Return the decoded photo, or None (logged) so one bad product never aborts the build."""
    async with semaphore:
        try:
            content = await _fetch_bytes(client, product.image_url)
        except httpx.HTTPError as exc:
            logger.warning("Product %d: download failed: %s", product.product_id, exc)
            return None
    try:
        image = Image.open(BytesIO(content))
        # Decode now so corrupt files fail here, not in the middle of an encode batch.
        image.load()
    except OSError as exc:
        logger.warning("Product %d: not a readable image: %s", product.product_id, exc)
        return None
    return image


async def index_products(
    products: Sequence[Product],
    encoder: Encoder,
    store: ProductVectorStore,
    batch_size: int,
) -> BuildSummary:
    summary = BuildSummary()
    semaphore = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(
        timeout=DOWNLOAD_TIMEOUT, headers=headers, follow_redirects=True
    ) as client:
        with tqdm(total=len(products), unit="img", desc="Indexing") as progress:
            for batch in batched(products, batch_size, strict=False):
                images = await asyncio.gather(
                    *(_download_image(client, semaphore, product) for product in batch)
                )
                downloaded = [
                    (product, image)
                    for product, image in zip(batch, images, strict=True)
                    if image is not None
                ]
                if downloaded:
                    ok_products = [product for product, _ in downloaded]
                    vectors = encoder.encode_images([image for _, image in downloaded])
                    store.upsert(ok_products, vectors)
                summary.indexed += len(downloaded)
                summary.failed += len(batch) - len(downloaded)
                progress.update(len(batch))
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    # Libraries (httpx, huggingface_hub) log every HTTP request at INFO; keep them at
    # WARNING so this script's progress and summary stay readable.
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger.setLevel(logging.INFO)
    args = parse_args(argv)
    settings = get_settings()

    catalog = ProductCatalog.from_parquet(settings.products_path)
    products = list(catalog)
    if args.limit is not None:
        products = select_demo_subset(products, args.limit)

    encoder = ClipEncoder(settings.clip_model_name)
    store = ProductVectorStore.from_settings(settings)
    if args.recreate and store.exists():
        logger.info("Dropping existing collection '%s'", settings.qdrant_collection)
        store.delete_collection()
    store.ensure_collection(encoder.dimension)

    already_indexed = store.indexed_ids()
    pending = [p for p in products if p.product_id not in already_indexed]
    skipped = len(products) - len(pending)

    summary = asyncio.run(index_products(pending, encoder, store, args.batch_size))
    logger.info(
        "Done: indexed=%d failed=%d skipped=%d (already indexed); collection holds %d vectors",
        summary.indexed,
        summary.failed,
        skipped,
        store.count(),
    )


if __name__ == "__main__":
    main()
