"""Production service construction with fake models and an in-memory vector store."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from clothes_recommender import services as service_module
from clothes_recommender.api import create_app
from clothes_recommender.catalog import CustomerData
from clothes_recommender.vector_store import ProductVectorStore


@pytest.fixture
def construction(monkeypatch, tmp_path, sample_catalog):
    purchases = pd.DataFrame(
        [(1, 1, date(2020, 9, 1))],
        columns=["user_id", "product_id", "purchase_date"],
    )
    customers = CustomerData(pd.DataFrame({"user_id": [1], "age": [30]}), purchases)
    settings = SimpleNamespace(
        products_path=tmp_path / "products.parquet",
        users_path=tmp_path / "users.parquet",
        purchases_path=tmp_path / "purchases.parquet",
        shopper_purchases_path=tmp_path / "shopper_purchases.parquet",
        co_purchases_path=tmp_path / "co_purchases.parquet",
        ranker_path=tmp_path / "ranker.json",
        clip_model_name="fake",
        llm_configured=False,
        images_dir=tmp_path / "images",
        query_photos_dir=tmp_path / "query_photos",
        evaluation_path=tmp_path / "evaluation.json",
    )
    encoder = Mock(dimension=2)
    encoder.encode_texts.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
    client = QdrantClient(":memory:")
    store = ProductVectorStore(client, "test-services")
    close = Mock(wraps=store.close)
    monkeypatch.setattr(store, "close", close)
    monkeypatch.setattr(service_module.ProductCatalog, "from_parquet", lambda _: sample_catalog)
    monkeypatch.setattr(service_module.CustomerData, "from_parquet", lambda *_: customers)
    monkeypatch.setattr(service_module.ProductVectorStore, "from_settings", lambda _: store)
    monkeypatch.setattr(service_module, "ClipEncoder", Mock(return_value=encoder))
    monkeypatch.setattr(
        service_module,
        "load_shopper_purchases",
        Mock(return_value=purchases.rename(columns={"user_id": "shopper_id"})),
    )
    yield SimpleNamespace(settings=settings, encoder=encoder, store=store, close=close)
    client.close()


def test_existing_collection_must_match_loaded_encoder(construction):
    construction.store.ensure_collection(3)

    with pytest.raises(ValueError, match=r"3-d vectors.*2-d vectors.*--recreate"):
        service_module.build_services(construction.settings)

    construction.encoder.load.assert_called_once_with()
    construction.close.assert_called_once_with()


def test_compatible_collection_remains_open_and_searchable(construction, sample_products):
    construction.store.ensure_collection(2)
    construction.store.upsert([sample_products[0]], np.array([[1.0, 0.0]], dtype=np.float32))

    services = service_module.build_services(construction.settings)

    assert services.store is construction.store
    assert [hit.product_id for hit in services.search.by_text("jeans")] == [1]
    assert services.store.count() == 1
    construction.close.assert_not_called()


def test_missing_collection_is_not_created_and_search_stays_unavailable(construction, monkeypatch):
    ensure = Mock(wraps=construction.store.ensure_collection)
    monkeypatch.setattr(construction.store, "ensure_collection", ensure)

    services = service_module.build_services(construction.settings)

    assert not services.store.exists()
    ensure.assert_not_called()
    with TestClient(create_app(services)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["indexed_products"] == 0
        response = client.post("/search/text", json={"query": "jeans"})
        assert response.status_code == 503
        assert "scripts.build_index" in response.json()["detail"]
    assert not services.store.exists()
    construction.close.assert_not_called()


@pytest.mark.parametrize("stage", ["encoder_constructor", "encoder_load", "shoppers", "graph"])
def test_initialization_failure_closes_owned_store(construction, monkeypatch, stage):
    error = RuntimeError("synthetic initialization failure")
    if stage == "encoder_constructor":
        service_module.ClipEncoder.side_effect = error
    elif stage == "encoder_load":
        construction.encoder.load.side_effect = error
    elif stage == "shoppers":
        service_module.load_shopper_purchases.side_effect = error
    else:
        monkeypatch.setattr(service_module, "GraphRecommender", Mock(side_effect=error))

    with pytest.raises(RuntimeError, match="synthetic initialization failure") as raised:
        service_module.build_services(construction.settings)

    assert raised.value is error
    construction.close.assert_called_once_with()


@pytest.mark.parametrize("version,width", [(1, 10), (2, 15)])
def test_service_loads_versioned_ranker(construction, version, width):
    from clothes_recommender.recommenders.learned import RankerModel

    model = RankerModel(
        feature_version=version,
        weights=[0.0] * width,
        trained_through=date(2020, 8, 31),
        training_window_ends=[date(2020, 8, 17), date(2020, 8, 24), date(2020, 8, 31)],
        training_rows=10,
        training_positives=1,
    )
    construction.settings.ranker_path.write_text(model.model_dump_json())
    services = service_module.build_services(construction.settings)
    assert services.learned is not None
    assert services.learned.recommend(1)
