"""Fetch exact H&M article photos from a public, revision-pinned image mirror.

Usage: uv run python -m scripts.fetch_photos --data-dir data [--workers 4] [--limit 1]
Re-running skips fully decoded JPEGs. No archive, credentials, or substitutes are used.
Photos larger than prepare_data.PHOTO_MAX_SIZE are saved downsized, like extracted ones.

Origin: H&M Group / Kaggle Personalized Fashion Recommendations (2022).
Mirror: einrafh/hnm-fashion-recommendations-data on Hugging Face. Its README maps
zero-padded article IDs to data/raw/images/{first_three_digits}/{article_id}.jpg.
The mirror specifies non-commercial and research use under the original terms;
photos are local data, not redistributed source assets. See the saved source report.
"""

import argparse
import hashlib
import json
import logging
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from numbers import Integral
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx
import pandas as pd
from PIL import Image

from scripts.prepare_data import PHOTO_MAX_SIZE

SOURCE_REPO = "https://huggingface.co/datasets/einrafh/hnm-fashion-recommendations-data"
SOURCE_REVISION = "acde210b0090be673abdba5f989c48925c856f78"
SOURCE = {
    "origin": "H&M Group / Kaggle Personalized Fashion Recommendations (2022)",
    "origin_url": "https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations",
    "mirror": SOURCE_REPO,
    "revision": SOURCE_REVISION,
    "source_card": f"{SOURCE_REPO}/blob/{SOURCE_REVISION}/README.md",
    "license_notice": "Non-commercial and research use; subject to original distribution terms.",
    "mapping": "article_id zero-padded to 10 digits; images/{first 3 digits}/{article_id}.jpg",
}
MAX_PHOTO_BYTES = 10 * 1024 * 1024
logger = logging.getLogger(__name__)


def photo_url(product_id: int) -> str:
    if not isinstance(product_id, Integral) or not 0 < product_id < 10**10:
        raise ValueError(f"Invalid H&M article ID: {product_id!r}")
    padded = f"{product_id:010d}"
    return f"{SOURCE_REPO}/resolve/{SOURCE_REVISION}/data/raw/images/{padded[:3]}/{padded}.jpg"


def validate_jpeg(source: Path | BytesIO) -> None:
    with Image.open(source) as image:
        if image.format != "JPEG":
            raise ValueError("Expected a JPEG photo")
        image.load()


def downsized(content: bytes) -> bytes:
    """Shrink photos larger than PHOTO_MAX_SIZE, matching prepare_data; keep smaller ones."""
    with Image.open(BytesIO(content)) as image:
        if image.width <= PHOTO_MAX_SIZE[0] and image.height <= PHOTO_MAX_SIZE[1]:
            return content
        photo = image.convert("RGB")
    photo.thumbnail(PHOTO_MAX_SIZE)
    output = BytesIO()
    photo.save(output, format="JPEG", quality=90)
    return output.getvalue()


def atomic_write(path: Path, content: bytes) -> None:
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as raw:
            temporary = Path(raw.name)
            raw.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_photo(client: httpx.Client, product_id: int, images_dir: Path) -> dict:
    url = photo_url(product_id)
    path = images_dir / f"{product_id}.jpg"
    try:
        validate_jpeg(path)
        return {"product_id": product_id, "status": "skipped"}
    except (OSError, ValueError, Image.DecompressionBombError):
        pass

    for attempt in range(3):
        try:
            content = bytearray()
            with client.stream("GET", url) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    if len(content) > MAX_PHOTO_BYTES:
                        raise ValueError("Photo exceeds the 10 MiB download limit")
            validate_jpeg(BytesIO(content))
            # The report describes the mirror's file; the saved copy may be downsized.
            atomic_write(path, downsized(bytes(content)))
            return {
                "product_id": product_id,
                "status": "downloaded",
                "url": url,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        except (httpx.HTTPError, OSError, ValueError, Image.DecompressionBombError) as error:
            retryable = isinstance(error, httpx.TransportError) or (
                isinstance(error, httpx.HTTPStatusError)
                and (error.response.status_code == 429 or error.response.status_code >= 500)
            )
            if retryable and attempt < 2:
                time.sleep(0.5 * 2**attempt)
                continue
            return {"product_id": product_id, "status": "failed", "url": url, "error": str(error)}
    raise AssertionError("Unreachable retry state")


def fetch_photos(data_dir: Path, workers: int = 4, limit: int | None = None) -> dict:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    ids = pd.read_parquet(data_dir / "products.parquet", columns=["product_id"])["product_id"]
    for product_id in ids:
        photo_url(product_id)
    product_ids = sorted(set(int(pid) for pid in ids))
    if limit is not None:
        product_ids = product_ids[:limit]
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with (
        httpx.Client(
            timeout=httpx.Timeout(30, connect=10),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=workers, max_keepalive_connections=workers),
        ) as client,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        pending = [pool.submit(download_photo, client, pid, images_dir) for pid in product_ids]
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            if result["status"] == "failed":
                logger.warning("Photo %s failed: %s", result["product_id"], result["error"])
            if len(results) % 100 == 0 or len(results) == len(product_ids):
                logger.info("Checked %d/%d photos", len(results), len(product_ids))
    results.sort(key=lambda row: row["product_id"])
    report = {
        "source": SOURCE,
        "total": len(results),
        "downloaded": sum(row["status"] == "downloaded" for row in results),
        "skipped": sum(row["status"] == "skipped" for row in results),
        "bytes_downloaded": sum(row.get("bytes", 0) for row in results),
        "failed_ids": [row["product_id"] for row in results if row["status"] == "failed"],
        "photos": results,
    }
    atomic_write(data_dir / "photo_downloads.json", json.dumps(report, indent=2).encode())
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    logger.setLevel(logging.INFO)
    report = fetch_photos(args.data_dir, args.workers, args.limit)
    print(json.dumps({key: value for key, value in report.items() if key != "photos"}, indent=2))
    if report["failed_ids"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
