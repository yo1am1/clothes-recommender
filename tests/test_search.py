import pytest
from PIL import Image

from clothes_recommender.domain import Encoder, Product, ProductFilter
from clothes_recommender.search import ProductSearch
from clothes_recommender.vector_store import ProductVectorStore


@pytest.fixture
def search(fake_encoder: Encoder, vector_store: ProductVectorStore) -> ProductSearch:
    return ProductSearch(fake_encoder, vector_store)


def test_by_text_returns_best_match_first(search: ProductSearch):
    hits = search.by_text("Cotton Polo Shirt", top_k=2)

    assert len(hits) == 2
    assert hits[0].product_id == 4


def test_by_text_applies_filter(search: ProductSearch):
    hits = search.by_text(
        "Cotton Polo Shirt", top_k=10, product_filter=ProductFilter(categories=("shoes",))
    )

    assert {hit.product_id for hit in hits} == {5, 6}


def test_by_image_finds_product_indexed_from_same_photo(
    search: ProductSearch,
    fake_encoder: Encoder,
    vector_store: ProductVectorStore,
    sample_products: list[Product],
):
    photo = Image.new("RGB", (4, 4), color=(200, 30, 30))
    boots = sample_products[5]
    vector_store.upsert([boots], fake_encoder.encode_images([photo]))

    hits = search.by_image(photo, top_k=1)

    assert hits[0].product_id == boots.product_id
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
