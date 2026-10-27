"""Index build/resume checks with tiny photos, a fake encoder and in-memory Qdrant."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from qdrant_client import QdrantClient

from clothes_recommender.catalog import ProductCatalog
from clothes_recommender.domain import ProductFilter
from clothes_recommender.vector_store import ProductVectorStore
from scripts import build_index


@pytest.fixture
def index_setup(tmp_path, monkeypatch, sample_products, fake_encoder):
    products = [
        sample_products[0].model_copy(update={"section": "Ladieswear", "price": 10.0}),
        sample_products[1].model_copy(update={"section": "Menswear", "price": 40.0}),
    ]
    for product in products:
        Image.new("RGB", (4, 6), "red").save(tmp_path / f"{product.product_id}.jpg")
    store = ProductVectorStore(QdrantClient(":memory:"), "test-build")
    settings = SimpleNamespace(
        products_path=tmp_path / "products.parquet",
        images_dir=tmp_path,
        clip_model_name="fake",
        qdrant_collection="test-build",
    )
    encoded_batches = []
    encode_images = fake_encoder.encode_images

    def record_images(images):
        encoded_batches.append(len(images))
        return encode_images(images)

    monkeypatch.setattr(fake_encoder, "encode_images", record_images)
    monkeypatch.setattr(build_index, "get_settings", lambda: settings)
    monkeypatch.setattr(build_index, "ClipEncoder", lambda _: fake_encoder)
    monkeypatch.setattr(ProductVectorStore, "from_settings", lambda _: store)
    monkeypatch.setattr(
        ProductCatalog,
        "from_parquet",
        lambda _: ProductCatalog(pd.DataFrame([product.model_dump() for product in products])),
    )
    yield products, store, encoded_batches
    store.close()


def test_resume_refreshes_section_and_price_without_encoding_existing_images(index_setup, tmp_path):
    products, store, encoded_batches = index_setup
    build_index.main([])
    before = store.get_vectors([1, 2])
    query = before[1]
    assert [
        hit.product_id
        for hit in store.search(
            query, product_filter=ProductFilter(sections=("Ladieswear",), max_price=20.0)
        )
    ] == [1]

    build_index.main([])
    products[0] = products[0].model_copy(update={"section": "Menswear", "price": 80.0})
    (tmp_path / "1.jpg").unlink()  # Existing vectors need no photo to refresh their metadata.
    build_index.main([])

    assert encoded_batches == [2]
    assert store.count() == 2
    assert store.search(query, product_filter=ProductFilter(sections=("Ladieswear",))) == []
    assert store.search(query, product_filter=ProductFilter(max_price=20.0)) == []
    assert [
        hit.product_id
        for hit in store.search(
            query,
            product_filter=ProductFilter(sections=("Menswear",), min_price=80.0, max_price=80.0),
        )
    ] == [1]
    for pid, vector in before.items():
        np.testing.assert_array_equal(store.get_vectors([pid])[pid], vector)


def test_limit_refreshes_existing_products_outside_the_new_subset(index_setup):
    products, store, encoded_batches = index_setup
    build_index.main(["--limit", "1"])
    before = store.get_vectors([1])[1]
    products[0] = products[0].model_copy(update={"section": "Sport", "price": 80.0})
    products[1] = products[1].model_copy(update={"section": "Ladieswear", "price": 30.0})

    build_index.main(["--limit", "1"])

    assert encoded_batches == [1, 1]
    assert store.indexed_ids() == {1, 2}
    assert store.search(before, product_filter=ProductFilter(max_price=20.0)) == []
    assert [
        hit.product_id
        for hit in store.search(
            before, product_filter=ProductFilter(sections=("Sport",), min_price=80.0)
        )
    ] == [1]
    np.testing.assert_array_equal(store.get_vectors([1])[1], before)


def test_recreate_reencodes_existing_images_and_removes_old_points(index_setup):
    products, store, encoded_batches = index_setup
    build_index.main([])
    store.upsert(
        [products[0].model_copy(update={"product_id": 99})],
        np.asarray([store.get_vectors([1])[1]]),
    )

    build_index.main(["--recreate", "--batch-size", "1"])

    assert encoded_batches == [2, 1, 1]
    assert store.indexed_ids() == {1, 2}
