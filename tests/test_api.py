import dataclasses
import io
import json
from collections.abc import Iterator
from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from clothes_recommender.api import create_app
from clothes_recommender.catalog import CustomerData, ProductCatalog
from clothes_recommender.chat.assistant import AssistantReply
from clothes_recommender.chat.intent import SearchIntent
from clothes_recommender.recommenders.collaborative import CollaborativeRecommender
from clothes_recommender.recommenders.content_based import ContentBasedRecommender
from clothes_recommender.recommenders.graph import GraphRecommender
from clothes_recommender.recommenders.popularity import PopularityRecommender
from clothes_recommender.recommenders.repeat import RepeatPurchaseRecommender
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
def services(sample_catalog, vector_store, fake_encoder, tmp_path) -> Services:
    users = pd.DataFrame({"user_id": [1, 2], "age": [34, 27]})
    purchases = pd.DataFrame(
        [(1, 1, date(2020, 9, 2)), (2, 1, date(2020, 9, 3)), (2, 3, date(2020, 9, 4))],
        columns=["user_id", "product_id", "purchase_date"],
    )
    Image.new("RGB", (4, 6), "red").save(tmp_path / "3.jpg")
    customers = CustomerData(users, purchases)
    popularity = PopularityRecommender(customers.purchases)
    content_based = ContentBasedRecommender(customers, sample_catalog, vector_store, popularity)
    # The sampled customers double as the whole crowd in this tiny fixture.
    crowd = purchases.rename(columns={"user_id": "shopper_id"})
    collaborative = CollaborativeRecommender(customers, sample_catalog, crowd, popularity)
    return Services(
        catalog=sample_catalog,
        customers=customers,
        store=vector_store,
        search=ProductSearch(fake_encoder, vector_store),
        content_based=content_based,
        collaborative=collaborative,
        graph=GraphRecommender(
            customers,
            sample_catalog,
            popularity,
            pd.DataFrame([(1, 3, 2, 1.5)], columns=["product_id", "other_id", "count", "lift"]),
        ),
        repeat=RepeatPurchaseRecommender(customers, popularity),
        images_dir=tmp_path,
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


def test_product_image_is_served_from_images_dir(client) -> None:
    response = client.get("/products/3/image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert client.get("/products/1/image").status_code == 404


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


def test_demo_examples_are_real_and_leave_all_users_available(client) -> None:
    users_before = client.get("/users").json()
    response = client.get("/demo/customers")
    assert response.status_code == 200
    examples = response.json()
    ids = [example["user_id"] for example in examples]
    # The fixture's customers are too small to qualify; tests/test_demo.py covers selection.
    assert len(ids) <= 10
    assert len(ids) == len(set(ids))
    assert set(ids) <= set(users_before)
    for example in examples:
        assert example["label"] and example["description"]
        profile = client.get(f"/users/{example['user_id']}").json()
        assert example["focus_product_id"] in {
            purchase["product"]["product_id"] for purchase in profile["purchases"]
        }
    assert client.get("/users").json() == users_before
    assert client.get("/demo/customers").json() == examples


@pytest.mark.parametrize("strategy", ["content", "collaborative", "graph"])
def test_recommendations_exclude_purchases(client, strategy) -> None:
    response = client.get("/users/1/recommendations", params={"strategy": strategy, "top_k": 3})
    assert response.status_code == 200
    ids = [hit["product"]["product_id"] for hit in response.json()]
    assert ids and 1 not in ids


def test_collaborative_recommends_what_shoppers_bought_together_with_reasons(client) -> None:
    hits = client.get("/users/1/recommendations", params={"strategy": "collaborative"}).json()
    # Shopper 2 bought product 1, like user 1, and also product 3.
    assert hits[0]["product"]["product_id"] == 3
    assert hits[0]["reasons"] == ["1 shopper who bought Slim Jeans (blue) also bought this."]
    assert client.get("/users/1/similar").status_code == 404  # replaced by per-hit reasons


def test_graph_recommendations_include_evidence_and_reject_unknown_users(client) -> None:
    hits = client.get("/users/1/recommendations", params={"strategy": "graph"}).json()
    linked = next(hit for hit in hits if hit["product"]["product_id"] == 3)
    assert linked["reasons"]
    assert client.get("/users/404/recommendations", params={"strategy": "graph"}).status_code == 404


def test_graph_neighborhood_has_valid_edges_and_bounded_requests(client) -> None:
    response = client.get("/products/1/graph", params={"top_k": 2})
    assert response.status_code == 200
    graph = response.json()
    ids = {node["id"] for node in graph["nodes"]}
    assert "product:1" in ids
    assert graph["edges"]
    assert all(edge["source"] in ids and edge["target"] in ids for edge in graph["edges"])
    assert client.get("/products/404/graph").status_code == 404
    assert client.get("/products/1/graph", params={"top_k": 51}).status_code == 422


@pytest.mark.parametrize("kind", ["complement", "alternative"])
def test_related_products_validate_products_and_relation(client, kind) -> None:
    response = client.get("/products/1/related", params={"kind": kind})
    assert response.status_code == 200
    assert all(hit["product"]["product_id"] != 1 for hit in response.json())
    assert client.get("/products/404/related", params={"kind": kind}).status_code == 404
    assert client.get("/products/1/related", params={"kind": "anything"}).status_code == 422


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


def test_text_search_can_be_limited_to_a_section(client) -> None:
    assert client.get("/sections").json() == ["Ladieswear", "Menswear", "Sport"]
    hits = client.post(
        "/search/text", json={"query": "Floral Summer Dress", "section": "Menswear"}
    ).json()
    assert hits and {hit["product"]["section"] for hit in hits} == {"Menswear"}


def test_similar_products_exclude_the_product_itself(services, sample_products) -> None:
    unindexed = sample_products[0].model_copy(update={"product_id": 7})
    catalog = ProductCatalog(pd.DataFrame([p.model_dump() for p in [*sample_products, unindexed]]))
    with TestClient(create_app(dataclasses.replace(services, catalog=catalog))) as client:
        hits = client.get("/products/3/similar", params={"top_k": 3}).json()
        assert len(hits) == 3
        assert 3 not in {hit["product"]["product_id"] for hit in hits}
        assert client.get("/products/404/similar").status_code == 404
        response = client.get("/products/7/similar")
        assert response.status_code == 404
        assert "image index" in response.json()["detail"]


def test_query_photos_are_listed_and_served(services, tmp_path) -> None:
    photos = tmp_path / "query_photos"
    photos.mkdir()
    Image.new("RGB", (4, 6), "blue").save(photos / "900000001.jpg")
    labels = [
        {"photo_id": 900000001, "name": "Denim jacket", "product_type": "Jacket", "colour": "Blue"}
    ]
    (photos / "query_photos.json").write_text(json.dumps(labels))
    with TestClient(create_app(dataclasses.replace(services, query_photos_dir=photos))) as client:
        assert client.get("/search/examples").json() == labels
        image = client.get("/search/examples/900000001/image")
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/jpeg"
        assert client.get("/search/examples/1/image").status_code == 404


def test_query_photos_are_empty_until_fetched(client) -> None:
    assert client.get("/search/examples").json() == []
    assert client.get("/search/examples/1/image").status_code == 404


def test_evaluation_is_served_when_present(services, tmp_path) -> None:
    path = tmp_path / "evaluation.json"
    with TestClient(create_app(dataclasses.replace(services, evaluation_path=path))) as client:
        missing = client.get("/evaluation")
        assert missing.status_code == 404
        assert "scripts.evaluate" in missing.json()["detail"]
        path.write_text(json.dumps({"top_k": 12, "results": {}}))
        assert client.get("/evaluation").json() == {"top_k": 12, "results": {}}


def test_repeat_strategy_includes_purchase_and_grounded_reason(client):
    response = client.get("/users/1/recommendations", params={"strategy": "repeat", "top_k": 3})
    assert response.status_code == 200
    assert response.json()[0]["product"]["product_id"] == 1
    assert response.json()[0]["reasons"] == ["Buy again: last purchased on 2020-09-02."]
    assert (
        client.get("/users/404/recommendations", params={"strategy": "repeat"}).status_code == 404
    )
