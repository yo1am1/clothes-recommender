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


PRODUCT_GROUPS = {
    "Trousers": "Garment Lower body",
    "Dress": "Garment Full body",
    "Polo shirt": "Garment Upper body",
    "Sneakers": "Shoes",
    "Boots": "Shoes",
}


def _product(
    product_id: int, name: str, product_type: str, section: str, colour: str, price: float
) -> Product:
    return Product(
        product_id=product_id,
        name=name,
        product_type=product_type,
        product_group=PRODUCT_GROUPS[product_type],
        section=section,
        colour=colour,
        description=f"{name} in {colour.lower()}.",
        price=price,
        shopify_category=f"Apparel & Accessories > {PRODUCT_GROUPS[product_type]}",
        shopify_attributes={"color": colour},
    )


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def sample_products() -> list[Product]:
    """Six H&M-style products over five product types and three sections."""
    return [
        _product(1, "Slim Jeans", "Trousers", "Menswear", "Blue", 29.99),
        _product(2, "Wide Leg Trousers", "Trousers", "Ladieswear", "Black", 24.99),
        _product(3, "Floral Summer Dress", "Dress", "Ladieswear", "Red", 34.99),
        _product(4, "Cotton Polo Shirt", "Polo shirt", "Menswear", "White", 17.99),
        _product(5, "Running Sneakers", "Sneakers", "Sport", "Black", 89.99),
        _product(6, "Leather Ankle Boots", "Boots", "Ladieswear", "Brown", 119.99),
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
