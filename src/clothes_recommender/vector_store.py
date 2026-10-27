"""Qdrant-backed store of product image embeddings with metadata filtering."""

import warnings
from collections.abc import Sequence
from itertools import batched

import numpy as np
from qdrant_client import QdrantClient, models

from clothes_recommender.config import Settings
from clothes_recommender.domain import Product, ProductFilter, ScoredProduct

_KEYWORD_FIELDS = ("product_type", "product_group", "section", "colour")
_FLOAT_FIELDS = ("price",)


class IndexNotBuiltError(RuntimeError):
    def __init__(self, collection: str) -> None:
        super().__init__(
            f"Vector collection '{collection}' does not exist. "
            "Build it with `uv run python -m scripts.build_index` "
            "(add `--limit 500` for a quick demo build)."
        )


class ProductVectorStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    @classmethod
    def from_settings(cls, settings: Settings) -> "ProductVectorStore":
        """Connect to a Qdrant server when a URL is configured, else use the embedded store."""
        if settings.qdrant_url:
            api_key = (
                settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
            )
            client = QdrantClient(url=settings.qdrant_url, api_key=api_key)
        else:
            client = QdrantClient(path=str(settings.qdrant_path))
        return cls(client, settings.qdrant_collection)

    def exists(self) -> bool:
        return self._client.collection_exists(self._collection)

    def count(self) -> int:
        self._require_collection()
        return self._client.count(self._collection, exact=True).count

    def ensure_collection(self, dimension: int) -> None:
        """Create the collection and its payload indexes unless it already exists."""
        if self.exists():
            existing_dimension = self._client.get_collection(
                self._collection
            ).config.params.vectors.size
            if existing_dimension != dimension:
                raise ValueError(
                    f"Collection '{self._collection}' stores {existing_dimension}-d vectors "
                    f"but the encoder produces {dimension}-d vectors. Rebuild it with `--recreate`."
                )
            return
        self._client.create_collection(
            self._collection,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )
        with warnings.catch_warnings():
            # Embedded Qdrant ignores payload indexes (and warns); they only speed up a server.
            warnings.filterwarnings("ignore", message="Payload indexes have no effect")
            for field in _KEYWORD_FIELDS:
                self._client.create_payload_index(
                    self._collection, field, models.PayloadSchemaType.KEYWORD
                )
            for field in _FLOAT_FIELDS:
                self._client.create_payload_index(
                    self._collection, field, models.PayloadSchemaType.FLOAT
                )

    def close(self) -> None:
        # Embedded Qdrant holds a file lock until closed.
        self._client.close()

    def delete_collection(self) -> None:
        self._client.delete_collection(self._collection)

    def upsert(
        self, products: Sequence[Product], vectors: np.ndarray, batch_size: int = 256
    ) -> None:
        if vectors.ndim != 2 or len(products) != len(vectors):
            raise ValueError(
                f"Expected one vector per product: got {len(products)} products and "
                f"vectors of shape {vectors.shape}."
            )
        points = [
            models.PointStruct(
                id=product.product_id, vector=vector.tolist(), payload=_payload(product)
            )
            for product, vector in zip(products, vectors, strict=True)
        ]
        for batch in batched(points, batch_size, strict=False):
            self._client.upsert(self._collection, points=list(batch))

    def refresh_payloads(self, products: Sequence[Product], batch_size: int = 256) -> None:
        """Update filtering metadata without changing existing image vectors."""
        self._require_collection()
        for batch in batched(products, batch_size, strict=False):
            self._client.batch_update_points(
                self._collection,
                update_operations=[
                    models.SetPayloadOperation(
                        set_payload=models.SetPayload(
                            payload=_payload(product), points=[product.product_id]
                        )
                    )
                    for product in batch
                ],
                wait=True,
            )

    def search(
        self, vector: np.ndarray, top_k: int = 10, product_filter: ProductFilter | None = None
    ) -> list[ScoredProduct]:
        self._require_collection()
        response = self._client.query_points(
            self._collection,
            query=vector.tolist(),
            query_filter=_to_qdrant_filter(product_filter),
            limit=top_k,
            with_payload=False,
        )
        return [ScoredProduct(int(point.id), point.score) for point in response.points]

    def get_vectors(self, product_ids: Sequence[int]) -> dict[int, np.ndarray]:
        """Stored vectors by product id; ids that are not indexed are omitted."""
        self._require_collection()
        records = self._client.retrieve(
            self._collection, ids=list(product_ids), with_payload=False, with_vectors=True
        )
        return {int(record.id): np.asarray(record.vector, dtype=np.float32) for record in records}

    def indexed_ids(self) -> set[int]:
        """Ids of every indexed product, so an index build can resume where it stopped."""
        self._require_collection()
        ids: set[int] = set()
        offset = None
        while True:
            records, offset = self._client.scroll(
                self._collection, limit=10_000, offset=offset, with_payload=False
            )
            ids.update(int(record.id) for record in records)
            if offset is None:
                return ids

    def _require_collection(self) -> None:
        if not self.exists():
            raise IndexNotBuiltError(self._collection)


def _normalise(value: str) -> str:
    """Keyword payloads are lowercase, so filters must be too to match."""
    return value.strip().lower()


def _payload(product: Product) -> dict[str, str | float]:
    return {
        "product_type": _normalise(product.product_type),
        "product_group": _normalise(product.product_group),
        "section": _normalise(product.section),
        "colour": _normalise(product.colour),
        "price": product.price,
    }


def _to_qdrant_filter(product_filter: ProductFilter | None) -> models.Filter | None:
    if product_filter is None or product_filter.is_empty():
        return None

    must: list[models.Condition] = []
    keyword_constraints = {
        "product_type": product_filter.product_types,
        "product_group": product_filter.product_groups,
        "section": product_filter.sections,
        "colour": product_filter.colours,
    }
    for field, values in keyword_constraints.items():
        if values:
            match = models.MatchAny(any=[_normalise(value) for value in values])
            must.append(models.FieldCondition(key=field, match=match))
    if product_filter.min_price is not None or product_filter.max_price is not None:
        price_range = models.Range(gte=product_filter.min_price, lte=product_filter.max_price)
        must.append(models.FieldCondition(key="price", range=price_range))

    must_not: list[models.Condition] = []
    if product_filter.exclude_ids:
        must_not.append(models.HasIdCondition(has_id=list(product_filter.exclude_ids)))

    return models.Filter(must=must or None, must_not=must_not or None)
