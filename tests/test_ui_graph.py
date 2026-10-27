"""Offline graph explorer checks using real Streamlit controls and mocked HTTP responses."""

import io
import json
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from clothes_recommender import ui

PRODUCT = {
    "product_id": 1,
    "name": 'Coat "blue" <winter>',
    "price": 30.0,
    "product_type": "Coat",
    "colour": "Blue",
}
RECOMMENDED = {**PRODUCT, "product_id": 2, "name": "Wool coat", "price": 45.0}
GRAPH = {
    "nodes": [
        {"id": "product:1", "label": PRODUCT["name"], "kind": "product", "product_id": 1},
        {"id": "product:2", "label": "Wool coat", "kind": "product", "product_id": 2},
        {
            "id": "category:coat",
            "label": "Outerwear > Coats",
            "kind": "taxonomy",
            "product_id": None,
        },
    ],
    "edges": [
        {
            "source": "product:1",
            "target": "category:coat",
            "relation": "in_category",
            "weight": 1.0,
        },
        {"source": "product:1", "target": "product:2", "relation": "alternative", "weight": 0.5},
    ],
}


def run_ui():
    import streamlit as st

    from clothes_recommender import ui

    # Component harness; native page routing has its own regression below.
    tabs = st.tabs(
        [
            "Recommendations",
            "Knowledge graph",
            "Text search",
            "Image search",
            "Assistant",
            "Evaluation",
        ]
    )
    for tab, render in zip(
        tabs,
        [
            ui.recommendations_tab,
            ui.knowledge_graph_tab,
            ui.text_search_tab,
            ui.image_search_tab,
            ui.chat_tab,
            ui.evaluation_tab,
        ],
        strict=True,
    ):
        with tab:
            render()


def test_search_tabs_wait_for_a_query_and_missing_photos_have_a_caption(mocked_api):
    calls, _ = mocked_api
    app = AppTest.from_function(run_ui).run()
    assert not app.exception
    assert app.text_input(key="text_query").value == ""
    assert app.get("file_uploader")
    # Nothing is searched until the shopper types, picks or uploads something.
    assert not any(path in ("/search/text", "/search/image") for path, _ in calls)
    assert any(caption.value == "No photo available." for caption in app.caption)
    # A missing evaluation file is an expected state, not an error.
    assert any(info.value == "No evaluation results yet." for info in app.info)


@pytest.fixture
def request_bodies() -> dict[str, list[bytes]]:
    """Request bodies by path, recorded by mocked_api."""
    return {}


@pytest.fixture
def mocked_api(monkeypatch, request_bodies):
    calls = []
    overrides = {}

    def request(request):
        path = request.url.path
        calls.append((path, dict(request.url.params)))
        request_bodies.setdefault(path, []).append(request.content)
        if path in overrides:
            return httpx.Response(overrides[path][0], **overrides[path][1])
        if path.endswith("/image"):
            return httpx.Response(404, json={"detail": "No image"})
        if path == "/health":
            payload = {"status": "ok", "indexed_products": 2, "chat_enabled": False}
        elif path in ("/products/1", "/products/2"):
            payload = PRODUCT if path.endswith("/1") else RECOMMENDED
        elif path.endswith("/similar"):
            payload = [{"product": RECOMMENDED, "score": 0.8, "reasons": []}]
        elif path == "/demo/customers":
            payload = [
                {
                    "user_id": 7,
                    "label": "Coat connections",
                    "description": "Explore shared coat evidence.",
                    "focus_product_id": 1,
                },
                {
                    "user_id": 8,
                    "label": "Another purchase history",
                    "description": "Explore a different purchase.",
                    "focus_product_id": 3,
                },
            ]
        elif path == "/users":
            payload = [7, 8, 9]
        elif path in ("/users/7", "/users/8", "/users/9"):
            payload = {
                "user": {"user_id": int(path.rsplit("/", 1)[1]), "age": None},
                "purchases": [{"product": PRODUCT, "purchase_date": "2020-01-01"}],
            }
        elif path.endswith("/recommendations") or path.endswith("/related"):
            payload = [
                {"product": RECOMMENDED, "score": 0.8, "reasons": ["Shares a purchase connection."]}
            ]
        elif path.endswith("/graph"):
            payload = GRAPH
        elif path == "/sections":
            payload = ["Ladieswear", "Menswear"]
        elif path == "/search/examples":
            payload = []
        elif path == "/evaluation":
            return httpx.Response(404, json={"detail": "No evaluation results yet."})
        else:
            raise AssertionError(f"Unexpected API call: {request.url}")
        return httpx.Response(200, json=payload)

    client = httpx.Client(base_url="http://offline.test", transport=httpx.MockTransport(request))
    monkeypatch.setattr(ui, "api_client", lambda: client)
    ui.product_photo.clear()
    ui.query_photo.clear()
    yield calls, overrides
    ui.product_photo.clear()
    ui.query_photo.clear()
    client.close()


def test_graph_explorer_focus_strategy_reasons_and_related_controls(mocked_api):
    calls, overrides = mocked_api
    overrides["/users/8"] = (
        200,
        {
            "json": {
                "user": {"user_id": 8, "age": None},
                "purchases": [
                    {"product": {**PRODUCT, "product_id": 3, "name": "Different purchase"}}
                ],
            }
        },
    )
    app = AppTest.from_function(run_ui).run()
    assert not app.exception
    assert "Knowledge graph" in [tab.label for tab in app.tabs]
    focus = next(widget for widget in app.selectbox if widget.label == "Focus product")
    assert focus.value == 1
    assert "Purchased" in focus.options[0]
    assert "Recommended" in focus.options[1]
    relationships = next(frame.value for frame in app.dataframe if "Relationship" in frame.value)
    assert set(relationships["Relationship"]) == {"in_category", "alternative"}
    assert any(title.value == "Product connections" for title in app.title)
    relationship_details = next(
        expander for expander in app.expander if expander.label == "Relationships"
    )
    assert not relationship_details.proto.expanded
    assert any(text.value == "Shares a purchase connection." for text in app.text)
    assert any(caption.value == "No photo available." for caption in app.caption)
    assert ("/products/1/graph", {"top_k": "12"}) in calls

    strategy = next(widget for widget in app.radio if widget.label == "Strategy")
    strategy.set_value("graph").run()
    assert not app.exception
    assert ("/users/7/recommendations", {"strategy": "graph"}) in calls

    focus = next(widget for widget in app.selectbox if widget.label == "Focus product")
    focus.set_value(2).run()
    assert not app.exception
    assert ("/products/2/graph", {"top_k": "12"}) in calls
    related = next(widget for widget in app.radio if widget.label == "Relationship")
    related.set_value("alternative").run()
    assert not app.exception
    assert ("/products/2/related", {"kind": "alternative"}) in calls
    assert any("higher approximate price" in caption.value for caption in app.caption)

    customer = next(widget for widget in app.selectbox if widget.label == "Customer")
    customer.set_value(8).run()
    assert not app.exception
    assert ("/users/8/recommendations", {"strategy": "graph"}) in calls
    focus = next(widget for widget in app.selectbox if widget.label == "Focus product")
    assert focus.value == 3
    assert focus.options == ["Purchased · Different purchase (#3)", "Recommended · Wool coat (#2)"]
    assert ("/products/3/graph", {"top_k": "12"}) in calls


@pytest.mark.parametrize(
    ("endpoint", "payload", "message"),
    [
        ("/demo/customers", [], "No example customers"),
        ("/products/1/graph", {"nodes": [], "edges": []}, "No relationships"),
        ("/products/1/related", [], "No co-purchased products"),
    ],
)
def test_graph_empty_states(mocked_api, endpoint, payload, message):
    _, overrides = mocked_api
    overrides[endpoint] = (200, {"json": payload})
    app = AppTest.from_function(run_ui).run()
    assert not app.exception
    assert any(message in info.value for info in app.info)


def test_graph_endpoint_error_is_visible_without_crashing(mocked_api):
    _, overrides = mocked_api
    overrides["/products/1/graph"] = (503, {"json": {"detail": "Graph is not available yet."}})
    app = AppTest.from_function(run_ui).run()
    assert not app.exception
    assert any("Graph is not available yet." in error.value for error in app.error)


def test_missing_photos_tolerate_network_errors(monkeypatch):
    def unavailable(request):
        raise httpx.ConnectError("offline", request=request)

    client = httpx.Client(
        base_url="http://offline.test", transport=httpx.MockTransport(unavailable)
    )
    monkeypatch.setattr(ui, "api_client", lambda: client)
    ui.product_photo.clear()
    assert ui.product_photo(123456) is None
    ui.product_photo.clear()
    client.close()


def test_graph_spec_keeps_unusual_labels_as_data_and_has_legend_tooltips():
    label = '"</script><img src=x onerror=alert(1)>\\\n雪 [x.y]'
    graph = {
        "nodes": [
            {**node, "label": label if node["product_id"] == 1 else node["label"]}
            for node in GRAPH["nodes"]
        ],
        "edges": GRAPH["edges"],
    }
    spec, table = ui.graph_view(graph, 1)
    assert json.loads(json.dumps(spec)) == spec
    points = spec["layer"][1]["data"]["values"]
    assert next(point for point in points if point["id"] == "product:1")["Label"] == label
    assert table[0]["Source"] == label
    node_layer = spec["layer"][1]
    assert node_layer["encoding"]["color"]["legend"]["title"] == "Node type"
    assert {item["field"] for item in node_layer["encoding"]["tooltip"]} >= {"Label", "Kind"}
    assert node_layer["params"][0]["select"]["fields"] == ["id"]
    assert spec["layer"][0]["mark"]["color"] == "#64748B"


def test_graph_view_is_bounded_deterministic_and_excludes_dangling_edges():
    nodes = [GRAPH["nodes"][0]] + [
        {"id": f"p:{i}", "label": f"Product {i}", "kind": "product", "product_id": i}
        for i in range(2, 30)
    ]
    edges = [
        {"source": "product:1", "target": node["id"], "relation": "co_purchased", "weight": i}
        for i, node in enumerate(nodes[1:])
    ]
    edges += [{"source": "missing", "target": "product:1", "relation": "invalid", "weight": 100}]
    graph = {"nodes": nodes, "edges": edges}
    spec, table = ui.graph_view(graph, 1)
    assert len(spec["layer"][1]["data"]["values"]) == 12
    assert len(table) <= 24
    assert spec == ui.graph_view({"nodes": nodes[::-1], "edges": edges[::-1]}, 1)[0]
    assert all(row["Relationship"] != "invalid" for row in table)
    assert {point["id"] for point in spec["layer"][1]["data"]["values"]} == {
        "product:1",
        *[f"p:{i}" for i in range(19, 30)],
    }


def test_empty_graph_view_has_no_rows():
    spec, table = ui.graph_view({"nodes": [], "edges": []}, 1)
    assert table == []
    assert spec["layer"][1]["data"]["values"] == []


def test_customer_without_purchases_can_focus_a_graph_recommendation(mocked_api):
    _, overrides = mocked_api
    overrides["/users/7"] = (200, {"json": {"user": {"user_id": 7, "age": None}, "purchases": []}})
    app = AppTest.from_function(run_ui).run()
    assert not app.exception
    focus = next(widget for widget in app.selectbox if widget.label == "Focus product")
    assert focus.value == 2
    assert focus.options == ["Recommended · Wool coat (#2)"]


def test_customer_without_purchases_or_recommendations_has_no_focus_control(mocked_api):
    calls, overrides = mocked_api
    overrides["/users/7"] = (200, {"json": {"user": {"user_id": 7, "age": None}, "purchases": []}})
    overrides["/users/7/recommendations"] = (200, {"json": []})
    app = AppTest.from_function(run_ui).run()
    assert not app.exception
    assert not any(widget.label == "Focus product" for widget in app.selectbox)
    assert any("No purchases or graph recommendations" in info.value for info in app.info)
    assert not any(path.endswith("/graph") for path, _ in calls)


def test_selecting_product_node_filters_table_and_fetches_its_related_hits(mocked_api, monkeypatch):
    calls, _ = mocked_api
    monkeypatch.setattr(
        ui.st,
        "vega_lite_chart",
        lambda *args, **kwargs: {"selection": {"node": [{"id": "product:2"}]}},
    )
    app = AppTest.from_function(run_ui).run()
    assert not app.exception
    relationships = next(frame.value for frame in app.dataframe if "Relationship" in frame.value)
    assert len(relationships) == 1
    assert relationships.iloc[0]["Target"] == "Wool coat"
    assert ("/products/2/related", {"kind": "complement"}) in calls


@pytest.mark.parametrize(
    ("theme", "label", "edge", "product_color"),
    [("dark", "#FAFAFA", "#9CA3AF", "#55B7CA"), ("light", "#31333F", "#64748B", "#287D8E")],
)
def test_graph_contrast_uses_viewer_theme(theme, label, edge, product_color):
    spec, _ = ui.graph_view(GRAPH, 1, theme)
    text = spec["layer"][2]["mark"]
    assert text["color"] == label
    assert text["fontSize"] >= 12
    assert spec["config"]["legend"]["labelColor"] == label
    assert spec["layer"][0]["mark"]["color"] == edge
    scale = spec["layer"][1]["encoding"]["color"]["scale"]
    assert dict(zip(scale["domain"], scale["range"], strict=True))["Product"] == product_color


def test_graph_tab_passes_actual_viewer_theme(mocked_api, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        type(ui.st.context), "theme", property(lambda _: SimpleNamespace(type="dark"))
    )
    rendered = []
    original_view = ui.graph_view

    def view(graph, product_id, theme_type):
        rendered.append(theme_type)
        return original_view(graph, product_id, theme_type)

    monkeypatch.setattr(ui, "graph_view", view)
    app = AppTest.from_function(run_ui).run()
    assert not app.exception
    assert rendered == ["dark"]


def test_alternative_empty_state_explains_required_category_evidence(mocked_api):
    _, overrides = mocked_api
    overrides["/products/1/related"] = (200, {"json": []})
    app = AppTest.from_function(run_ui).run()
    next(widget for widget in app.radio if widget.label == "Relationship").set_value(
        "alternative"
    ).run()
    assert not app.exception
    assert any("matching specific Shopify category" in info.value for info in app.info)
    assert any("product type and section" in caption.value for caption in app.caption)
    assert any("Unclassified or broad categories are excluded" in info.value for info in app.info)


def test_curated_defaults_and_all_customer_access_are_separate_per_tab(mocked_api):
    calls, _ = mocked_api
    app = AppTest.from_function(run_ui).run()
    assert not app.exception
    for label in ("User", "Customer"):
        selector = next(widget for widget in app.selectbox if widget.label == label)
        assert len(selector.options) == 2
        assert "Coat connections" in selector.options[0]
    # The measured strong repeat baseline is the default.
    assert next(widget for widget in app.radio if widget.label == "Strategy").value == "repeat"
    assert any(caption.value == "Explore shared coat evidence." for caption in app.caption)
    assert not any(path == "/users" for path, _ in calls)
    assert not any(path == "/users/8" for path, _ in calls)

    app.toggle(key="graph_user_browse_all").set_value(True).run()
    assert not app.exception
    assert ("/users", {}) in calls
    customer = next(widget for widget in app.selectbox if widget.label == "Customer")
    assert customer.options == ["7", "8", "9"]
    customer.set_value(9).run()
    assert not app.exception
    assert ("/users/9", {}) in calls
    assert next(widget for widget in app.selectbox if widget.label == "User").options == [
        "Coat connections (#7)",
        "Another purchase history (#8)",
    ]
    app.toggle(key="graph_user_browse_all").set_value(False).run()
    assert not app.exception
    assert next(widget for widget in app.selectbox if widget.label == "Customer").value == 7


def test_curated_example_focus_overrides_first_purchase(mocked_api):
    calls, overrides = mocked_api
    overrides["/demo/customers"] = (
        200,
        {
            "json": [
                {
                    "user_id": 7,
                    "label": "Specific evidence",
                    "description": "Starts at the informative product.",
                    "focus_product_id": 2,
                }
            ]
        },
    )
    overrides["/users/7"] = (
        200,
        {
            "json": {
                "user": {"user_id": 7, "age": None},
                "purchases": [{"product": PRODUCT}, {"product": RECOMMENDED}],
            }
        },
    )
    app = AppTest.from_function(run_ui).run()
    assert not app.exception
    assert next(widget for widget in app.selectbox if widget.label == "Focus product").value == 2
    assert ("/products/2/graph", {"top_k": "12"}) in calls


def test_curated_options_are_bounded_to_ten(mocked_api):
    _, overrides = mocked_api
    overrides["/demo/customers"] = (
        200,
        {
            "json": [
                {
                    "user_id": uid,
                    "label": f"Example {uid}",
                    "description": "Catalog evidence.",
                    "focus_product_id": 1,
                }
                for uid in range(7, 19)
            ]
        },
    )
    app = AppTest.from_function(run_ui).run()
    assert not app.exception
    for label in ("User", "Customer"):
        assert len(next(widget for widget in app.selectbox if widget.label == label).options) == 10


@pytest.mark.parametrize("key", ["graph_user", "recommendation_user"])
def test_all_customer_access_is_direct_and_can_be_repeated(mocked_api, key):
    calls, _ = mocked_api
    app = AppTest.from_function(run_ui).run()
    for _ in range(2):
        calls.clear()
        app.toggle(key=f"{key}_browse_all").set_value(True).run()
        assert not app.exception
        assert ("/users", {}) in calls
        assert not any(button.label == "Continue" for button in app.button)
        app.toggle(key=f"{key}_browse_all").set_value(False).run()
        assert not app.exception
        assert not app.toggle(key=f"{key}_browse_all").value


def run_product_cards():
    from clothes_recommender import ui

    products = [
        {
            "product_id": pid,
            "name": f"+ Plain product {pid}",
            "price": 20.0,
            "product_type": "Coat",
            "colour": "Black",
        }
        for pid in range(1, 8)
    ]
    evidence = [
        "Bought together with + Jane Swimsuit (product 823518001) in 2 baskets (lift 146).",
        "Shares age-group=Adults with purchased Top (product 123).",
        "Shares target-gender=Female with purchased Top (product 123).",
    ]
    ui.show_products(products, [146.5] * 7, [evidence] * 7)


def test_cards_have_fresh_three_column_rows_plain_names_and_collapsed_evidence(monkeypatch):
    monkeypatch.setattr(ui, "product_photo", lambda _: None)
    app = AppTest.from_function(run_product_cards).run()
    assert not app.exception
    assert len(app.get("column")) == 9
    assert sum(text.value.startswith("+ Plain product") for text in app.text) == 7
    assert not any("Plain product" in markdown.value for markdown in app.markdown)
    assert len(app.expander) == 7
    assert all(not expander.proto.expanded for expander in app.expander)
    assert all(expander.label == "Recommendation details" for expander in app.expander)
    assert (
        sum(
            caption.value == "Bought together with + Jane Swimsuit in 2 baskets."
            for caption in app.caption
        )
        == 7
    )
    for expander in app.expander:
        assert any("Ranking score:" in caption.value for caption in expander.caption)
        assert any("lift 146" in text.value for text in expander.text)


def test_profile_history_is_limited_and_does_not_fetch_purchase_photos(mocked_api):
    calls, overrides = mocked_api
    overrides["/users/7"] = (
        200,
        {
            "json": {
                "user": {"user_id": 7, "age": None},
                "purchases": [{"product": PRODUCT, "purchase_date": "2020-01-01"}] * 20,
            }
        },
    )
    app = AppTest.from_function(run_ui).run()
    assert not app.exception
    history = next(frame.value for frame in app.dataframe if "Product" in frame.value)
    assert len(history) == 10
    assert not any(path == "/products/1/image" for path, _ in calls)


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        (
            [
                "Shares target-gender=Female with purchased X.",
                "Shares age-group=Adults with purchased X.",
            ],
            "Connected through shared catalog attributes.",
        ),
        (
            ["Shares Shopify category Apparel > Shirts with purchased X."],
            "Shares a product category with a previous purchase.",
        ),
        (
            ["Popularity fallback from recent purchases; no graph evidence from history."],
            "Popular in recent purchases; no personal graph connections were found.",
        ),
        (
            [
                "3 shoppers who bought Strap top also bought this.",
                "1 shopper who bought Socks also bought this.",
            ],
            "3 shoppers who bought Strap top also bought this.",
        ),
        (
            ["Popular in the last week; no shopper bought it with this customer's items."],
            "Popular in the last week; no shopper bought it with this customer's items.",
        ),
        ([], None),
    ],
)
def test_primary_reason_does_not_repeat_technical_attributes(evidence, expected):
    assert ui.recommendation_reason(evidence) == expected


HIT = {"product": RECOMMENDED, "score": 0.31, "reasons": []}


def _jpeg(colour="navy") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (6, 9), colour).save(output, format="JPEG")
    return output.getvalue()


JPEG = _jpeg()


def test_text_search_sends_query_and_section(mocked_api, request_bodies):
    _, overrides = mocked_api
    overrides["/search/text"] = (200, {"json": [HIT]})
    app = AppTest.from_function(run_ui).run()
    app.selectbox[-1].select("Menswear")  # the search tab's section filter
    app.text_input(key="text_query").input("wool coat")
    app.button(key="description_search_submit").click().run()
    assert not app.exception
    assert any(text.value == "Wool coat" for text in app.text)
    assert json.loads(request_bodies["/search/text"][-1]) == {
        "query": "wool coat",
        "top_k": 9,
        "section": "Menswear",
    }


def test_text_example_searches_immediately_and_empty_submission_does_not_search(
    mocked_api, request_bodies
):
    calls, overrides = mocked_api
    overrides["/search/text"] = (200, {"json": [HIT]})
    app = AppTest.from_function(run_ui).run()
    app.button(key="description_search_submit").click().run()
    assert not any(path == "/search/text" for path, _ in calls)
    app.selectbox[-1].select("Menswear")
    app.button(key="text_example_2").click().run()
    assert not app.exception
    assert app.text_input(key="text_query").value == "chunky knit cardigan"
    assert json.loads(request_bodies["/search/text"][-1]) == {
        "query": "chunky knit cardigan",
        "top_k": 9,
        "section": "Menswear",
    }
    assert any(heading.value == "Matching pieces" for heading in app.subheader)


@pytest.mark.parametrize("has_upload", [False, True])
def test_example_photo_runs_an_image_search(mocked_api, request_bodies, monkeypatch, has_upload):
    calls, overrides = mocked_api
    examples = [
        {"photo_id": 900, "name": "Denim jacket", "product_type": "Jacket", "colour": "Blue"},
        {"photo_id": 901, "name": "White sneakers", "product_type": "Sneakers", "colour": "White"},
    ]
    photos = {900: JPEG, 901: _jpeg("white")}
    overrides["/search/examples"] = (200, {"json": examples})
    overrides["/search/image"] = (200, {"json": [HIT]})
    for photo_id, content in photos.items():
        overrides[f"/search/examples/{photo_id}/image"] = (200, {"content": content})
    uploaded = _jpeg("red")
    if has_upload:
        original_keys = []

        def file_uploader(*args, key=None, **kwargs):
            # A new widget identity clears the previous upload in Streamlit.
            original_keys.append(key)
            if key == original_keys[0]:
                return SimpleNamespace(name="uploaded.jpg", getvalue=lambda: uploaded)
            return None

        monkeypatch.setattr(ui.st, "file_uploader", file_uploader)
    app = AppTest.from_function(run_ui).run()
    assert not app.exception, app.exception
    if has_upload:
        assert uploaded in request_bodies["/search/image"][-1]
    else:
        assert not any(path == "/search/image" for path, _ in calls)
    for photo_id, content in photos.items():
        app.button(key=f"query_photo_{photo_id}").click().run()
        assert not app.exception
        assert ("/search/image", {"top_k": "9"}) in calls
        body = request_bodies["/search/image"][-1]
        assert f'filename="{photo_id}.jpg"'.encode() in body
        uploaded_content = body.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 2)[0]
        assert uploaded_content == content
        assert any(text.value == "Wool coat" for text in app.text)
    assert not any(path == "/search/text" for path, _ in calls)


def test_discover_switches_between_description_and_photo_controls(mocked_api):
    calls, _ = mocked_api
    app = AppTest.from_string("from clothes_recommender import ui; ui.discovery_page()").run()
    assert not app.exception
    assert app.text_input(key="text_query")
    assert not app.get("file_uploader")
    assert not any(path == "/search/examples" for path, _ in calls)
    app.button_group(key="discovery_mode").set_value("Photo").run()
    assert not app.exception
    assert not app.text_input
    assert app.get("file_uploader")
    assert any(path == "/search/examples" for path, _ in calls)


def test_graph_tab_can_show_visually_similar_products(mocked_api):
    calls, overrides = mocked_api
    overrides["/products/1/similar"] = (200, {"json": [HIT]})
    app = AppTest.from_function(run_ui).run()
    app.radio(key="graph_related_kind").set_value("similar").run()
    assert not app.exception
    assert any(path == "/products/1/similar" for path, _ in calls)
    assert any("CLIP image index" in caption.value for caption in app.caption)


def test_evaluation_tab_compares_strategies_with_popularity(mocked_api):
    _, overrides = mocked_api
    metrics = {
        "popularity": (0.002, 0.004),
        "graph": (0.006, 0.008),
    }
    overrides["/evaluation"] = (
        200,
        {
            "json": {
                "cutoff": "2020-09-16",
                "holdout_end": "2020-09-22",
                "holdout_days": 7,
                "top_k": 12,
                "users_evaluated": 321,
                "content_index": None,
                "sample_scope": {"prepared_users": 5000, "catalog_products": 14300},
                "repeat_purchase_limitation": {
                    "full_holdout_unique_user_items": 900,
                    "novel_holdout_unique_user_items": 800,
                },
                "results": {
                    name: {
                        "novel_item_holdout": {"map_at_12": novel, "users_evaluated": 300},
                        "full_holdout": {"map_at_12": full, "users_evaluated": 321},
                        "catalog_coverage": 0.1,
                        "recommendation_fill": 1.0,
                    }
                    for name, (novel, full) in metrics.items()
                },
            }
        },
    )
    app = AppTest.from_function(run_ui).run()
    assert not app.exception
    table = next(frame.value for frame in app.dataframe if "Strategy" in frame.value)
    # Best first.
    assert table["vs baseline"].round(2).tolist() == [2.0, 1.0]
    assert any("321" in markdown.value for markdown in app.markdown)
    assert any("without an image index" in caption.value for caption in app.caption)


def run_main():
    from clothes_recommender import ui

    ui.main()


def test_main_renders_only_selected_native_page(mocked_api):
    calls, _ = mocked_api
    app = AppTest.from_function(run_main).run()
    assert not app.exception
    assert app.title[0].value == "For you"
    assert any(path == "/health" for path, _ in calls)
    assert not any(
        path in ("/evaluation", "/search/examples") or path.endswith("/graph") for path, _ in calls
    )
    assert not app.get("file_uploader")


def test_product_detail_is_reachable_from_catalog_and_keeps_names_plain(mocked_api):
    calls, _ = mocked_api
    app = AppTest.from_function(run_main).run()
    app.button(key="detail_recommendations_2_0").click().run()
    assert not app.exception
    assert ("/products/2", {}) in calls
    assert any(header.value == "Wool coat" for header in app.subheader)
    assert any("current stock is unknown" in caption.value for caption in app.caption)


def test_disabled_assistant_has_a_recovery_path(mocked_api):
    app = AppTest.from_string("from clothes_recommender import ui; ui.chat_tab()").run()
    assert not app.exception
    assert any("discover pieces by description or photo" in info.value for info in app.info)
    assert not app.chat_input


def test_validated_ranker_is_default_when_api_reports_it_ready(mocked_api):
    calls, overrides = mocked_api
    overrides["/health"] = (
        200,
        {
            "json": {
                "status": "ok",
                "indexed_products": 2,
                "catalog_products": 6,
                "chat_enabled": False,
                "ranking_enabled": True,
            }
        },
    )
    app = AppTest.from_function(run_main).run()
    assert not app.exception
    assert next(widget for widget in app.radio if widget.label == "Strategy").value == "learned"
    assert ("/users/7/recommendations", {"strategy": "learned", "top_k": "12"}) in calls


@pytest.mark.parametrize(
    ("status", "detail"),
    [(400, "Upload a valid PNG or JPEG image."), (413, "Image resolution is too large.")],
)
def test_rejected_photo_is_not_decoded_for_preview(mocked_api, monkeypatch, status, detail):
    _, overrides = mocked_api
    overrides["/search/image"] = (status, {"json": {"detail": detail}})
    monkeypatch.setattr(
        ui.st,
        "file_uploader",
        lambda *args, **kwargs: SimpleNamespace(
            name="corrupt.jpg", getvalue=lambda: b"not an image"
        ),
    )
    app = AppTest.from_string("from clothes_recommender import ui; ui.image_search_tab()").run()
    assert not app.exception
    assert app.error[0].value == detail
    assert not app.get("imgs")


def test_chat_keeps_long_sessions_within_request_limits(mocked_api, request_bodies):
    from clothes_recommender.api import ChatRequest

    _, overrides = mocked_api
    overrides["/health"] = (
        200,
        {"json": {"status": "ok", "indexed_products": 2, "chat_enabled": True}},
    )
    overrides["/chat"] = (200, {"json": {"text": "x" * 4001, "products": []}})
    app = AppTest.from_string("from clothes_recommender import ui; ui.chat_tab()").run()
    assert app.chat_input[0].proto.max_chars == 2000
    for number in range(28):
        app.chat_input[0].set_value(f"Show me coat {number}").run()
        assert not app.exception
        payload = ChatRequest.model_validate_json(request_bodies["/chat"][-1])
        assert len(payload.history) == min(number * 2, 10)
        assert all(len(turn.content) <= 4000 for turn in payload.history)
    assert len(app.session_state["chat_history"]) == 56
    assert app.session_state["chat_history"][-1]["content"] == "x" * 4001
    assert payload.history[-2].content == "Show me coat 26"


def test_popup_is_lazy_and_disabled_chat_has_recovery(mocked_api):
    calls, _ = mocked_api
    app = AppTest.from_string("from clothes_recommender import ui; ui.assistant_popup()").run()
    assert not app.exception
    assert calls == []
    assert not app.chat_input
    app.session_state["assistant_popup"] = True
    app.run()
    assert not app.exception
    assert any("discover pieces by description or photo" in info.value for info in app.info)
    assert not app.chat_input


def test_popup_keeps_conversation_on_close_and_can_reset(mocked_api, request_bodies):
    _, overrides = mocked_api
    overrides["/health"] = (
        200,
        {"json": {"status": "ok", "indexed_products": 2, "chat_enabled": True}},
    )
    overrides["/chat"] = (200, {"json": {"text": "Try this coat.", "products": [PRODUCT]}})
    app = AppTest.from_string("from clothes_recommender import ui; ui.assistant_popup()")
    app.session_state["assistant_popup"] = True
    app.run()
    app.chat_input(key="chat_input_popup").set_value("A blue coat").run()
    assert not app.exception
    assert json.loads(request_bodies["/chat"][-1]) == {"message": "A blue coat", "history": []}
    assert len(app.session_state["chat_history"]) == 2
    assert any(text.value == PRODUCT["name"] for text in app.text)
    app.button(key="chat_close").click().run()
    assert not app.chat_input
    assert len(app.session_state["chat_history"]) == 2
    app.session_state["assistant_popup"] = True
    app.run()
    assert any(text.value == PRODUCT["name"] for text in app.text)
    app.button(key="chat_view_popup_1_1").click().run()
    assert not app.exception
    assert any(header.value == PRODUCT["name"] for header in app.subheader)
    # Close the product dialog before clearing the conversation.
    app.run()
    app.button(key="chat_reset_popup").click().run()
    assert not app.exception
    assert app.session_state["chat_history"] == []
    assert not app.text


def test_popup_provider_error_does_not_poison_history(mocked_api, request_bodies):
    _, overrides = mocked_api
    overrides["/health"] = (
        200,
        {"json": {"status": "ok", "indexed_products": 2, "chat_enabled": True}},
    )
    overrides["/chat"] = (502, {"json": {"detail": "Assistant unavailable. Please try again."}})
    app = AppTest.from_string("from clothes_recommender import ui; ui.assistant_popup()")
    app.session_state["assistant_popup"] = True
    app.run()
    app.chat_input[0].set_value("A blue coat").run()
    assert not app.exception
    assert app.error[0].value == "Assistant unavailable. Please try again."
    assert app.session_state["chat_history"] == []
    overrides["/chat"] = (200, {"json": {"text": "Try a coat.", "products": []}})
    app.chat_input[0].set_value("A blue coat").run()
    assert not app.exception
    assert json.loads(request_bodies["/chat"][-1])["history"] == []
    assert len(app.session_state["chat_history"]) == 2
