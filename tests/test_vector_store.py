import numpy as np
import pytest
from qdrant_client import QdrantClient

from clothes_recommender.domain import Encoder, Product, ProductFilter
from clothes_recommender.vector_store import IndexNotBuiltError, ProductVectorStore


def _matching_ids(
    store: ProductVectorStore, encoder: Encoder, product_filter: ProductFilter
) -> set[int]:
    query = encoder.encode_texts(["any query"])[0]
    hits = store.search(query, top_k=100, product_filter=product_filter)
    return {hit.product_id for hit in hits}


def test_reads_on_missing_collection_raise_actionable_error(fake_encoder: Encoder):
    store = ProductVectorStore(QdrantClient(":memory:"), "missing")
    query = fake_encoder.encode_texts(["jeans"])[0]

    assert not store.exists()
    with pytest.raises(IndexNotBuiltError, match=r"scripts\.build_index"):
        store.search(query)
    with pytest.raises(IndexNotBuiltError):
        store.count()


def test_ensure_collection_is_idempotent(vector_store: ProductVectorStore, fake_encoder: Encoder):
    vector_store.ensure_collection(fake_encoder.dimension)

    assert vector_store.count() == 6


def test_ensure_collection_rejects_dimension_change(
    vector_store: ProductVectorStore, fake_encoder: Encoder
):
    with pytest.raises(ValueError, match="--recreate"):
        vector_store.ensure_collection(fake_encoder.dimension + 1)


def test_upsert_rejects_mismatched_vectors(
    vector_store: ProductVectorStore, fake_encoder: Encoder, sample_products: list[Product]
):
    with pytest.raises(ValueError, match="one vector per product"):
        vector_store.upsert(sample_products, fake_encoder.encode_texts(["only one"]))


def test_search_ranks_exact_match_first(vector_store: ProductVectorStore, fake_encoder: Encoder):
    query = fake_encoder.encode_texts(["Floral Summer Dress"])[0]

    hits = vector_store.search(query, top_k=3)

    assert len(hits) == 3
    assert hits[0].product_id == 3
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize(
    ("product_filter", "expected_ids"),
    [
        (ProductFilter(), {1, 2, 3, 4, 5, 6}),
        (ProductFilter(product_types=("Sneakers", "Trousers")), {1, 2, 5}),
        (ProductFilter(product_groups=("Shoes",)), {5, 6}),
        (ProductFilter(sections=("  menswear ",)), {1, 4}),
        (ProductFilter(colours=("BLACK",)), {2, 5}),
        (ProductFilter(min_price=30.0, max_price=100.0), {3, 5}),
        (ProductFilter(max_price=20.0), {4}),
        (ProductFilter(exclude_ids=(1, 2, 3)), {4, 5, 6}),
        (ProductFilter(product_types=("Boots", "Sneakers"), colours=("black",)), {5}),
        (ProductFilter(sections=("Ladieswear",), exclude_ids=(2,)), {3, 6}),
    ],
)
def test_filters_combine_and_across_fields_or_within(
    vector_store: ProductVectorStore,
    fake_encoder: Encoder,
    product_filter: ProductFilter,
    expected_ids: set[int],
):
    assert _matching_ids(vector_store, fake_encoder, product_filter) == expected_ids


def test_get_vectors_omits_unknown_ids(vector_store: ProductVectorStore, fake_encoder: Encoder):
    vectors = vector_store.get_vectors([5, 999])

    assert set(vectors) == {5}
    expected = fake_encoder.encode_texts(["Running Sneakers"])[0]
    np.testing.assert_allclose(vectors[5], expected, atol=1e-6)


def test_indexed_ids_lists_every_product(vector_store: ProductVectorStore):
    assert vector_store.indexed_ids() == {1, 2, 3, 4, 5, 6}


def test_delete_collection(vector_store: ProductVectorStore):
    vector_store.delete_collection()

    assert not vector_store.exists()
