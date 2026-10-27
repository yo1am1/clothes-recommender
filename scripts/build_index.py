"""Embed the prepared product photos with CLIP and upsert them into the vector store.

Usage: uv run python -m scripts.build_index [--limit N] [--batch-size B] [--recreate]

Re-running refreshes catalog metadata and skips existing image vectors unless --recreate.
"""

import argparse
import logging
from collections import defaultdict
from collections.abc import Sequence
from itertools import batched, zip_longest
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from clothes_recommender.catalog import ProductCatalog
from clothes_recommender.config import get_settings
from clothes_recommender.domain import Encoder, Product
from clothes_recommender.embeddings import ClipEncoder
from clothes_recommender.vector_store import ProductVectorStore

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--limit",
        type=int,
        help="Index only N products, spread evenly across sections (quick demo build).",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Images per encode batch.")
    parser.add_argument(
        "--recreate", action="store_true", help="Drop the collection and index from scratch."
    )
    return parser.parse_args(argv)


def select_demo_subset(products: Sequence[Product], limit: int) -> list[Product]:
    """Pick products round-robin across sections so even a tiny index demos every section."""
    by_section: dict[str, list[Product]] = defaultdict(list)
    for product in sorted(products, key=lambda p: p.product_id):
        by_section[product.section].append(product)
    groups = [by_section[section] for section in sorted(by_section)]
    interleaved = [p for round_ in zip_longest(*groups) for p in round_ if p is not None]
    return interleaved[:limit]


def load_photo(path: Path) -> Image.Image | None:
    """Return the decoded photo, or None (logged) so one bad file never aborts the build."""
    try:
        image = Image.open(path)
        image.load()
    except (OSError, Image.DecompressionBombError) as error:
        logger.warning("Skipping %s: %s", path, error)
        return None
    return image


def index_products(
    products: Sequence[Product],
    encoder: Encoder,
    store: ProductVectorStore,
    images_dir: Path,
    batch_size: int,
) -> int:
    """Embed and upsert products in batches; return how many were indexed."""
    indexed = 0
    with tqdm(total=len(products), unit="img", desc="Indexing") as progress:
        for batch in batched(products, batch_size, strict=False):
            loaded = [
                (product, photo)
                for product in batch
                if (photo := load_photo(images_dir / f"{product.product_id}.jpg")) is not None
            ]
            if loaded:
                vectors = encoder.encode_images([photo for _, photo in loaded])
                store.upsert([product for product, _ in loaded], vectors)
            indexed += len(loaded)
            progress.update(len(batch))
    return indexed


def main(argv: Sequence[str] | None = None) -> None:
    # Libraries (huggingface_hub, httpx) log every HTTP request at INFO; keep them at
    # WARNING so this script's progress and summary stay readable.
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger.setLevel(logging.INFO)
    args = parse_args(argv)
    settings = get_settings()

    products = list(ProductCatalog.from_parquet(settings.products_path))

    encoder = ClipEncoder(settings.clip_model_name)
    store = ProductVectorStore.from_settings(settings)
    if args.recreate and store.exists():
        logger.info("Dropping existing collection '%s'", settings.qdrant_collection)
        store.delete_collection()
    store.ensure_collection(encoder.dimension)

    already_indexed = store.indexed_ids()
    store.refresh_payloads([p for p in products if p.product_id in already_indexed])
    if args.limit is not None:
        products = select_demo_subset(products, args.limit)
    pending = [p for p in products if p.product_id not in already_indexed]
    indexed = index_products(pending, encoder, store, settings.images_dir, args.batch_size)
    logger.info(
        "Done: indexed=%d failed=%d skipped=%d (already indexed); collection holds %d vectors",
        indexed,
        len(pending) - indexed,
        len(products) - len(pending),
        store.count(),
    )


if __name__ == "__main__":
    main()
