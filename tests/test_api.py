import dataclasses
import io
from collections.abc import Iterator
from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from clothes_recommender.api import create_app
from clothes_recommender.catalog import CustomerData
from clothes_recommender.chat.assistant import AssistantReply
from clothes_recommender.chat.intent import SearchIntent
from clothes_recommender.recommenders.collaborative import CollaborativeRecommender
from clothes_recommender.recommenders.content_based import ContentBasedRecommender
from clothes_recommender.recommenders.popularity import PopularityRecommender
from clothes_recommender.search import ProductSearch
from clothes_recommender.services import Services


class EchoAssistant:
    def reply(self, message, history) -> AssistantReply:
        return AssistantReply(
            text=f"echo: {message}", products=[], intent=SearchIntent(query=message)
        )


class BrokenAssistant:
    def reply(self, message, history) -> AssistantReply:
        raise ConnectionError("gateway down")


@pytest.fixture
def services(sample_catalog, vector_store, fake_encoder) -> Services:
    users = pd.DataFrame(
        [
            {
                "user_id": uid,
                "name": f"User {uid}",
                "email": f"u{uid}@example.com",
                "city": "Utrecht",
                "country": "Netherlands",
                "gender": "F",
            }
            for uid in (1, 2)
        ]
    )
    purchases = pd.DataFrame(
        [
            {
                "purchase_id": "a",
                "user_id": 1,
                "product_id": 1,
                "purchase_date": date(2025, 1, 2),
                "quantity": 1,
            },
            {
                "purchase_id": "b",
                "user_id": 2,
                "product_id": 1,
                "purchase_date": date(2025, 1, 3),
                "quantity": 1,
            },
            {
                "purchase_id": "c",
                "user_id": 2,
                "product_id": 3,
                "purchase_date": date(2025, 1, 4),
                "quantity": 2,
            },
        ]
    )
    customers = CustomerData(users, purchases)
    popularity = PopularityRecommender(sample_catalog)
    return Services(
        catalog=sample_catalog,
        customers=customers,
        store=vector_store,
        search=ProductSearch(fake_encoder, vector_store),
        content_based=ContentBasedRecommender(customers, sample_catalog, vector_store, popularity),
        collaborative=CollaborativeRecommender(customers, popularity),
        assistant=None,
    )


@pytest.fixture
def client(services) -> Iterator[TestClient]:
    with TestClient(create_app(services)) as client:
        yield client


def test_health_reports_index_size_and_chat_status(client) -> None:
    assert client.get("/health").json() == {
        "status": "ok",
        "indexed_products": 6,
        "chat_enabled": False,
    }


def test_get_product_and_missing_product(client) -> None:
    assert client.get("/products/3").json()["name"] == "Floral Summer Dress"
    assert client.get("/products/999").status_code == 404


def test_text_search_returns_scored_products(client) -> None:
    hits = client.post("/search/text", json={"query": "Floral Summer Dress", "top_k": 2}).json()
    assert len(hits) == 2
    assert hits[0]["product"]["product_id"] == 3


def test_image_search_accepts_images_and_rejects_other_files(client) -> None:
    png = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(png, format="PNG")
    response = client.post("/search/image", files={"file": ("red.png", png.getvalue())})
    assert response.status_code == 200
    assert len(response.json()) > 0

    response = client.post("/search/image", files={"file": ("notes.txt", b"not an image")})
    assert response.status_code == 400

    truncated = png.getvalue()[:-20]
    response = client.post("/search/image", files={"file": ("cut.png", truncated)})
    assert response.status_code == 400


def test_user_profile_lists_purchases_with_products(client) -> None:
    assert client.get("/users").json() == [1, 2]
    profile = client.get("/users/2").json()
    assert [p["product"]["product_id"] for p in profile["purchases"]] == [3, 1]
    assert client.get("/users/404").status_code == 404


@pytest.mark.parametrize("strategy", ["content", "collaborative"])
def test_recommendations_exclude_purchases(client, strategy) -> None:
    response = client.get("/users/1/recommendations", params={"strategy": strategy, "top_k": 3})
    assert response.status_code == 200
    ids = [hit["product"]["product_id"] for hit in response.json()]
    assert ids and 1 not in ids


def test_collaborative_recommends_what_similar_users_bought(client) -> None:
    assert client.get("/users/1/similar").json()[0]["user_id"] == 2
    hits = client.get("/users/1/recommendations", params={"strategy": "collaborative"}).json()
    assert hits[0]["product"]["product_id"] == 3


def test_chat_is_503_without_llm(client) -> None:
    assert client.post("/chat", json={"message": "hi"}).status_code == 503


def test_chat_returns_assistant_reply(services) -> None:
    with TestClient(create_app(dataclasses.replace(services, assistant=EchoAssistant()))) as client:
        response = client.post("/chat", json={"message": "hi", "history": []})
        assert response.json()["text"] == "echo: hi"


def test_chat_maps_provider_failure_to_502(services) -> None:
    with TestClient(
        create_app(dataclasses.replace(services, assistant=BrokenAssistant()))
    ) as client:
        response = client.post("/chat", json={"message": "hi"})
        assert response.status_code == 502
        assert "gateway down" not in response.json()["detail"]
