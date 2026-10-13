"""Semantic product search by text query or example photo."""

from PIL.Image import Image

from clothes_recommender.domain import Encoder, ProductFilter, ScoredProduct
from clothes_recommender.vector_store import ProductVectorStore


class ProductSearch:
    def __init__(self, encoder: Encoder, store: ProductVectorStore) -> None:
        self._encoder = encoder
        self._store = store

    def by_text(
        self, query: str, top_k: int = 10, product_filter: ProductFilter | None = None
    ) -> list[ScoredProduct]:
        vector = self._encoder.encode_texts([query])[0]
        return self._store.search(vector, top_k, product_filter)

    def by_image(
        self, image: Image, top_k: int = 10, product_filter: ProductFilter | None = None
    ) -> list[ScoredProduct]:
        vector = self._encoder.encode_images([image])[0]
        return self._store.search(vector, top_k, product_filter)
