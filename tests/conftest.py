"""Shared fixtures: a fake encoder, a tiny catalog and an in-memory vector store."""

import hashlib
from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd
import pytest
from PIL.Image import Image
from qdrant_client import QdrantClient

from clothes_recommender.catalog import ProductCatalog
from clothes_recommender.domain import Product
from clothes_recommender.vector_store import ProductVectorStore


class FakeEncoder:
    """Deterministic `Encoder`: equal inputs map to the same random unit vector.

    Texts and images live in unrelated regions of the space, so a search only finds a
    product when queried with exactly the text or image its vector was built from.
    """

    dimension = 8

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        return self._unit_vectors([text.encode() for text in texts])

    def encode_images(self, images: Sequence[Image]) -> np.ndarray:
        return self._unit_vectors([image.tobytes() for image in images])

    def _unit_vectors(self, payloads: list[bytes]) -> np.ndarray:
        vectors = np.empty((len(payloads), self.dimension), dtype=np.float32)
        for row, payload in enumerate(payloads):
            seed = int.from_bytes(hashlib.sha256(payload).digest()[:8])
            vector = np.random.default_rng(seed).standard_normal(self.dimension)
            vectors[row] = vector / np.linalg.norm(vector)
        return vectors


def _product(
    product_id: int,
    name: str,
    category: str,
    brand: str,
    color: str,
    price: float,
    rating: float,
    review_count: int,
) -> Product:
    return Product(
        product_id=product_id,
        name=name,
        category=category,
        brand=brand,
        color=color,
        price=price,
        rating=rating,
        review_count=review_count,
        image_url=f"https://example.com/images/{product_id}.jpg",
        product_url=f"https://example.com/products/{product_id}",
    )


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def sample_products() -> list[Product]:
    """Six products over four categories; brand/color are lowercase like the real data."""
    return [
        _product(1, "Levi's Slim Fit Jeans", "jeans", "levi's", "blue", 40.0, 4.4, 1200),
        _product(2, "Relaxed Fit Jeans", "jeans", "ben", "black", 25.0, 3.6, 80),
        _product(3, "Floral Summer Dress", "western wear", "zara", "red", 35.0, 4.1, 300),
        _product(4, "Cotton Polo Shirt", "t-shirts & polos", "levi's", "white", 18.0, 4.7, 950),
        _product(5, "Running Sneakers", "shoes", "nike", "black", 90.0, 4.5, 5000),
        _product(6, "Leather Ankle Boots", "shoes", "clarks", "brown", 120.0, 3.9, 40),
    ]


@pytest.fixture
def sample_catalog(sample_products: list[Product]) -> ProductCatalog:
    return ProductCatalog(pd.DataFrame([product.model_dump() for product in sample_products]))


@pytest.fixture
def vector_store(
    fake_encoder: FakeEncoder, sample_products: list[Product]
) -> Iterator[ProductVectorStore]:
    """In-memory store holding every sample product, embedded from its name as text."""
    client = QdrantClient(":memory:")
    store = ProductVectorStore(client, "test-products")
    store.ensure_collection(fake_encoder.dimension)
    store.upsert(sample_products, fake_encoder.encode_texts([p.name for p in sample_products]))
    yield store
    client.close()
