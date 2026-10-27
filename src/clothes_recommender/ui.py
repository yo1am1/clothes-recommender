"""Clothy fashion-discovery MVP: a thin Streamlit client over the HTTP API.

Run with: uv run streamlit run src/clothes_recommender/ui.py
"""

import math
import re
from pathlib import Path

import httpx
import streamlit as st

from clothes_recommender.config import get_settings

GRID_COLUMNS = 3
QUERY_PHOTO_COLUMNS = 6
EXAMPLE_QUERIES = [
    "black leather biker jacket",
    "striped summer dress",
    "kids t-shirt with a dinosaur print",
    "white sneakers",
    "floral bikini",
    "chunky knit cardigan",
]
STRATEGY_NAMES = {
    "learned": "Personalized ranking",
    "repeat": "Buy again + best-sellers",
    "popularity": "Recent best-sellers",
    "popularity_novel": "Best-sellers, new items only",
    "content": "Visual style (CLIP)",
    "collaborative": "Collaborative",
    "graph": "Purchase connections",
}


@st.cache_resource
def api_client() -> httpx.Client:
    # CLIP inference on CPU can take a few seconds on a cold start.
    return httpx.Client(base_url=get_settings().api_url, timeout=120.0)


def call_api(method: str, path: str, missing_ok: bool = False, **kwargs) -> dict | list | None:
    """Call the API; on failure show the server's explanation instead of a stack trace.

    With missing_ok, a 404 is an expected empty state and its explanation is shown as info.
    """
    try:
        response = api_client().request(method, path, **kwargs)
    except httpx.HTTPError:
        st.error("The catalog service is unavailable. Please try again.")
        return None
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:  # e.g. a plain-text 500 from a crashed worker
            detail = response.text
        (st.info if missing_ok and response.status_code == 404 else st.error)(detail)
        return None
    return response.json()


@st.cache_data(ttl=30, max_entries=2000)
def product_photo(product_id: int) -> bytes | None:
    # Fetched server-side rather than by URL: in Docker the browser can't resolve the API host.
    try:
        response = api_client().get(f"/products/{product_id}/image")
    except httpx.HTTPError:
        return None
    return response.content if response.is_success else None


@st.cache_data(max_entries=100)
def query_photo(photo_id: int) -> bytes | None:
    try:
        response = api_client().get(f"/search/examples/{photo_id}/image")
    except httpx.HTTPError:
        return None
    return response.content if response.is_success else None


def recommendation_reason(reasons: list[str]) -> str | None:
    """Summarize the strongest supplied evidence; keep raw evidence in the expander."""
    if reasons and reasons[0].startswith("Buy again:"):
        return reasons[0]
    if reasons and (" who bought " in reasons[0] or reasons[0].startswith("Popular in")):
        return reasons[0]  # collaborative: the purchase that points here most strongly
    for reason in reasons:
        if reason.startswith("Bought together with"):
            return re.sub(r"\s*\(product \d+\)|\s*\(lift [^)]+\)", "", reason)
    if any(reason.startswith("Bought on the same day") for reason in reasons):
        return "Bought together by the same shoppers."
    if any(reason.startswith("Same specific Shopify category") for reason in reasons):
        return "Same category, type and section, at a higher approximate price."
    if any(reason.startswith("Shares Shopify category") for reason in reasons):
        return "Shares a product category with a previous purchase."
    if any(reason.startswith("Shares ") for reason in reasons):
        return "Connected through shared catalog attributes."
    if any(reason.startswith("Popularity fallback") for reason in reasons):
        return "Popular in recent purchases; no personal graph connections were found."
    return None


def show_products(
    products: list[dict],
    scores: list[float] | None = None,
    reasons: list[list[str]] | None = None,
    *,
    context: str = "products",
) -> None:
    for start in range(0, len(products), GRID_COLUMNS):
        columns = st.columns(GRID_COLUMNS)
        for offset, product in enumerate(products[start : start + GRID_COLUMNS]):
            i = start + offset
            with (
                columns[offset],
                st.container(key=f"product_card_{context}_{product['product_id']}_{i}"),
            ):
                if (photo := product_photo(product["product_id"])) is not None:
                    st.image(photo, width="stretch")
                else:
                    st.caption("No photo available.")
                st.text(product["name"])
                st.markdown(f"**€{product['price']:.2f}**")
                st.caption(f"{product['product_type']} · {product['colour']}")
                evidence = reasons[i] if reasons is not None else []
                if reason := recommendation_reason(evidence):
                    st.caption(reason)
                if scores is not None or evidence:
                    with st.expander("Recommendation details"):
                        if scores is not None:
                            st.caption(f"Ranking score: {scores[i]:.3f}")
                        for detail in dict.fromkeys(evidence):
                            st.text(detail)
                if st.button(
                    f"View {product['name']}",
                    key=f"detail_{context}_{product['product_id']}_{i}",
                    width="stretch",
                ):
                    product_details(product["product_id"])


def show_hits(hits: list[dict] | None, *, context: str = "results") -> None:
    if hits:
        show_products(
            [h["product"] for h in hits],
            [h["score"] for h in hits],
            [h.get("reasons", []) for h in hits],
            context=context,
        )
    elif hits is not None:
        st.info("No matching products.")


def customer_selector(label: str, key: str) -> dict | None:
    """Start with a few explained examples; fetch the full ID list only on request."""
    examples = call_api("GET", "/demo/customers") or []
    examples_by_id = {example["user_id"]: example for example in examples[:10]}
    saved_mode = f"saved_{key}_browse_all"
    selected = st.session_state.get("selected_customer")
    default_all = st.session_state.get(saved_mode, False) or (
        selected is not None and selected not in examples_by_id
    )
    browse_all = st.toggle("Browse all customers", value=default_all, key=f"{key}_browse_all")
    st.session_state[saved_mode] = browse_all
    user_ids = call_api("GET", "/users") if browse_all else list(examples_by_id)
    if not user_ids:
        st.info(
            "No customers are available to explore."
            if browse_all
            else "No example customers are available. Browse all customers to explore."
        )
        return None
    user_id = st.selectbox(
        label,
        user_ids,
        index=user_ids.index(st.session_state.get("selected_customer"))
        if st.session_state.get("selected_customer") in user_ids
        else 0,
        key=f"{key}_{'all' if browse_all else 'examples'}",
        format_func=lambda uid: (
            str(uid) if browse_all else f"{examples_by_id[uid]['label']} (#{uid})"
        ),
    )
    st.session_state["selected_customer"] = user_id
    example = examples_by_id.get(user_id, {"user_id": user_id})
    if example.get("description"):
        st.caption(example["description"])
    return example


def recommendations_tab() -> None:
    left, right = st.columns([0.7, 0.3], gap="large")
    with left:
        st.title("For you")
        st.caption("Your recent favorites, and new pieces to explore.")
    with right, st.popover("Shopping history", width="stretch"):
        example = customer_selector("User", "recommendation_user")
        if example is None:
            return
        user_id = example["user_id"]
        profile = call_api("GET", f"/users/{user_id}")
        if profile:
            purchases = profile["purchases"]
            st.caption(f"Shopping history #{user_id}")
            if purchases:
                with st.expander("Recent purchases (up to 10)"):
                    st.dataframe(
                        [
                            {
                                "Product": item["product"]["name"],
                                "Type": item["product"]["product_type"],
                                "Price (€)": item["product"]["price"],
                                "Date": item.get("purchase_date", ""),
                            }
                            for item in purchases[:10]
                        ],
                        hide_index=True,
                        width="stretch",
                    )
    strategies = ["repeat", "collaborative", "content", "graph"]
    if st.session_state.get("service_health", {}).get("ranking_enabled"):
        strategies.insert(0, "learned")
    with st.expander("How picks are chosen"):
        strategy = st.radio(
            "Strategy",
            strategies,
            index=strategies.index(st.session_state.get("saved_strategy", strategies[0]))
            if st.session_state.get("saved_strategy", strategies[0]) in strategies
            else 0,
            format_func={
                "learned": "Personalized ranking",
                "repeat": "Buy again",
                "collaborative": "Similar shoppers",
                "content": "Visual style",
                "graph": "Purchase connections",
            }.get,
            horizontal=True,
            key="recommendation_strategy",
        )
        st.session_state["saved_strategy"] = strategy
        st.caption(
            "Buy again is a strong simple baseline. Personalized ranking learns from "
            "earlier purchases; discovery approaches deliberately show only new items."
        )
    descriptions = {
        "learned": "Favorites and new finds, ranked by your purchase patterns and recent demand.",
        "repeat": "Your most recent purchases first, followed by recent best-sellers.",
        "collaborative": "New pieces bought by shoppers with items like yours.",
        "content": "New pieces with photos close to the styles in your history.",
        "graph": "New pieces connected through purchases and catalog attributes.",
    }
    st.subheader("Worth another look" if strategy == "repeat" else "Selected for your history")
    st.caption(descriptions[strategy])
    with st.spinner("Finding your pieces…"):
        show_hits(
            call_api(
                "GET",
                f"/users/{user_id}/recommendations",
                params={"strategy": strategy, "top_k": 12},
            ),
            context="recommendations",
        )


@st.dialog("Product details", width="large")
def product_details(product_id: int) -> None:
    product = call_api("GET", f"/products/{product_id}")
    if product is None:
        return
    left, right = st.columns([0.45, 0.55], gap="large")
    with left:
        if (photo := product_photo(product_id)) is not None:
            st.image(photo, width="stretch")
        else:
            st.info("A photo is not available for this item.")
    with right:
        st.subheader(product["name"])
        st.markdown(f"### €{product['price']:.2f}")
        st.caption(
            f"{product['product_type']} · {product['colour']} · {product.get('section', '')}"
        )
        st.text(product.get("description") or "No further description is available.")
        st.caption("H&M research catalog, 2020. Prices are approximate; current stock is unknown.")
    related = st.radio(
        "Explore this piece",
        ["similar", "complement"],
        horizontal=True,
        format_func={"similar": "Similar styles", "complement": "Bought together"}.get,
        key=f"detail_relation_{product_id}",
    )
    path = (
        f"/products/{product_id}/similar"
        if related == "similar"
        else f"/products/{product_id}/related"
    )
    params = {"top_k": 3} if related == "similar" else {"top_k": 3, "kind": "complement"}
    hits = call_api("GET", path, params=params, missing_ok=True)
    if hits:
        # Related-item buttons cannot open a second native dialog inside this one.
        for hit in hits:
            item = hit["product"]
            st.text(f"{item['name']} · {item['colour']} · €{item['price']:.2f}")
    elif hits is not None:
        st.info("No related pieces are available yet.")


def graph_view(
    graph: dict, product_id: int, theme_type: str | None = None
) -> tuple[dict, list[dict]]:
    """A bounded one-hop view, with stable positions and the same edges as a text table."""
    nodes = {node["id"]: node for node in graph["nodes"]}
    focus = next((node for node in nodes.values() if node["product_id"] == product_id), None)
    edges = sorted(
        (edge for edge in graph["edges"] if edge["source"] in nodes and edge["target"] in nodes),
        key=lambda edge: (-edge["weight"], edge["source"], edge["target"], edge["relation"]),
    )
    visible = {}
    if focus is not None:
        visible[focus["id"]] = focus
        for edge in edges:
            if focus["id"] in (edge["source"], edge["target"]):
                neighbor = edge["target"] if edge["source"] == focus["id"] else edge["source"]
                if len(visible) < 12:
                    visible[neighbor] = nodes[neighbor]
    ordered = sorted(visible.values(), key=lambda node: (node["kind"], node["id"]))
    neighbors = [node for node in ordered if node is not focus]
    positions = {focus["id"]: (0.0, 0.0)} if focus else {}
    for i, node in enumerate(neighbors):
        angle = 2 * math.pi * i / len(neighbors)
        positions[node["id"]] = (math.cos(angle), math.sin(angle))
    points = [
        {
            "id": node["id"],
            "Label": node["label"],
            "Kind": node["kind"].replace("_", " ").title(),
            "Product ID": node["product_id"],
            "x": positions[node["id"]][0],
            "y": positions[node["id"]][1],
            "short_label": node["label"][:22] + ("…" if len(node["label"]) > 22 else ""),
            "focused": node is focus,
        }
        for node in ordered
    ]
    links = []
    table = []
    for edge in edges:
        if edge["source"] not in visible or edge["target"] not in visible:
            continue
        if len(links) == 24:
            break
        source, target = visible[edge["source"]], visible[edge["target"]]
        links.append(
            {
                **edge,
                "x": positions[source["id"]][0],
                "y": positions[source["id"]][1],
                "x2": positions[target["id"]][0],
                "y2": positions[target["id"]][1],
                "Source": source["label"],
                "Target": target["label"],
            }
        )
        table.append(
            {
                "Source": source["label"],
                "Source type": source["kind"],
                "Relationship": edge["relation"],
                "Target": target["label"],
                "Target type": target["kind"],
                "Weight": edge["weight"],
            }
        )
    dark = theme_type == "dark"
    label_color = "#FAFAFA" if dark else "#31333F"
    colors = (
        {"product": "#55B7CA", "attribute": "#E0AC58", "taxonomy": "#B699E7"}
        if dark
        else {"product": "#287D8E", "attribute": "#A66A16", "taxonomy": "#7254A3"}
    )
    kinds = sorted({node["kind"] for node in ordered})
    xy = {
        axis: {
            "field": axis,
            "type": "quantitative",
            "axis": None,
            "scale": {"domain": [-1.45, 1.45]},
        }
        for axis in ("x", "y")
    }
    spec = {
        "description": (
            "Product neighborhood. Node types appear in the legend; "
            "full relationships are listed in the table below."
        ),
        "height": 360,
        "layer": [
            {
                "data": {"values": links},
                "mark": {
                    "type": "rule",
                    "color": "#9CA3AF" if dark else "#64748B",
                    "opacity": 0.75,
                },
                "encoding": {
                    **xy,
                    "x2": {"field": "x2"},
                    "y2": {"field": "y2"},
                    "tooltip": [
                        {"field": field, "type": "nominal"}
                        for field in ("Source", "relation", "Target", "weight")
                    ],
                },
            },
            {
                "data": {"values": points},
                "mark": {"type": "circle", "opacity": 1},
                "params": [
                    {"name": "node", "select": {"type": "point", "fields": ["id"], "toggle": False}}
                ],
                "encoding": {
                    **xy,
                    "size": {"condition": {"test": "datum.focused", "value": 350}, "value": 180},
                    "color": {
                        "field": "Kind",
                        "type": "nominal",
                        "scale": {
                            "domain": [kind.replace("_", " ").title() for kind in kinds],
                            "range": [colors.get(kind, "#64748B") for kind in kinds],
                        },
                        "legend": {"title": "Node type", "orient": "bottom", "columns": 3},
                    },
                    "tooltip": [
                        {"field": field, "type": "nominal"}
                        for field in ("Label", "Kind", "Product ID", "id")
                    ],
                },
            },
            {
                "data": {"values": points},
                "mark": {"type": "text", "dy": -19, "fontSize": 12, "color": label_color},
                "encoding": {**xy, "text": {"field": "short_label", "type": "nominal"}},
            },
        ],
        "config": {
            "view": {"stroke": None},
            "legend": {
                "labelColor": label_color,
                "titleColor": label_color,
                "labelFontSize": 12,
                "titleFontSize": 12,
            },
        },
    }
    return spec, table


def knowledge_graph_tab() -> None:
    st.subheader("Explore product connections")
    st.caption(
        "Start with a purchase or graph recommendation. Select a node to inspect its connections."
    )
    with st.spinner("Loading customer examples…"):
        example = customer_selector("Customer", "graph_user")
    if example is None:
        return
    user_id = example["user_id"]
    with st.spinner("Loading purchases and graph recommendations…"):
        profile = call_api("GET", f"/users/{user_id}")
        hits = call_api("GET", f"/users/{user_id}/recommendations", params={"strategy": "graph"})
    purchases = profile["purchases"] if profile else []
    products = {purchase["product"]["product_id"]: purchase["product"] for purchase in purchases}
    purchased_ids = set(products)
    recommended = {hit["product"]["product_id"]: hit for hit in hits or []}
    products.update(
        {pid: hit["product"] for pid, hit in recommended.items() if pid not in products}
    )
    if not products:
        st.info("No purchases or graph recommendations are available for this customer.")
        return
    product_ids = list(products)
    focus_id = example.get("focus_product_id")
    product_id = st.selectbox(
        "Focus product",
        product_ids,
        index=product_ids.index(focus_id) if focus_id in products else 0,
        key=f"graph_focus_{user_id}_{focus_id}",
        format_func=lambda pid: (
            f"{'Purchased' if pid in purchased_ids else 'Recommended'} · "
            f"{products[pid]['name']} (#{pid})"
        ),
    )
    product = products[product_id]
    st.text(f"{product['name']} · €{product['price']:.2f} · {product['product_type']}")
    for reason in recommended.get(product_id, {}).get("reasons", []):
        st.caption(reason)
    with st.spinner("Loading product neighborhood…"):
        graph = call_api("GET", f"/products/{product_id}/graph", params={"top_k": 12})
    selected_id = None
    selected_product_id = product_id
    selected_label = product["name"]
    if graph is not None:
        spec, table = graph_view(graph, product_id, st.context.theme.type)
        if not table:
            st.info("No relationships are available for this product.")
        else:
            st.caption(
                "Up to 12 nodes and 24 relationships. "
                "Positions are for readability; distance has no meaning."
            )
            event = st.vega_lite_chart(
                spec,
                width="stretch",
                key=f"graph_chart_{user_id}_{product_id}",
                on_select="rerun",
                selection_mode="node",
            )
            selection = event.get("selection", {}).get("node", [])
            if selection:
                selected_id = selection[0].get("id")
            selected = next((node for node in graph["nodes"] if node["id"] == selected_id), None)
            if selected:
                st.write(f"Selected node: {selected['label']} ({selected['kind']})")
                if selected["product_id"] is not None:
                    selected_product_id = selected["product_id"]
                    selected_label = selected["label"]
                edges = spec["layer"][0]["data"]["values"]
                table = [
                    row
                    for row, edge in zip(table, edges, strict=True)
                    if selected_id in (edge["source"], edge["target"])
                ]
            st.subheader("Relationships")
            st.dataframe(table, hide_index=True, width="stretch")
    st.subheader(f"Related products · {selected_label}")
    kind = st.radio(
        "Relationship",
        ["complement", "alternative", "similar"],
        horizontal=True,
        key="graph_related_kind",
        format_func={
            "complement": "Bought together",
            "alternative": "Higher-price alternatives",
            "similar": "Looks similar",
        }.get,
    )
    if kind == "complement":
        st.caption("Co-purchased products from different product groups.")
    elif kind == "similar":
        st.caption(
            "Nearest product photos in the CLIP image index. No purchase data is used, "
            "so this works for products nobody has bought together yet."
        )
        with st.spinner("Finding look-alikes…"):
            show_hits(
                call_api("GET", f"/products/{selected_product_id}/similar"), context="graph_similar"
            )
        return
    else:
        st.caption(
            "Same specific Shopify category, product type and section, "
            "at a higher approximate price. "
            "Unclassified or broad categories are excluded. "
            "Price does not indicate quality or proven upsell value."
        )
    with st.spinner("Loading related products…"):
        related = call_api("GET", f"/products/{selected_product_id}/related", params={"kind": kind})
    if related == []:
        st.info(
            "No higher-price alternatives with a matching specific Shopify category, product type "
            "and section. Unclassified or broad categories are excluded."
            if kind == "alternative"
            else "No co-purchased products from different product groups."
        )
    else:
        show_hits(related, context="graph_related")


def use_example_query() -> None:
    if example := st.session_state.get("text_example"):
        st.session_state["text_query"] = example


def text_search_tab() -> None:
    st.caption(
        "Search product photos in everyday language. Describe a color, shape, material or occasion."
    )
    st.pills("Try an example", EXAMPLE_QUERIES, key="text_example", on_change=use_example_query)
    left, right = st.columns([0.7, 0.3])
    with left:
        query = st.text_input(
            "Describe what you are looking for", key="text_query", placeholder=EXAMPLE_QUERIES[0]
        )
    with right:
        sections = call_api("GET", "/sections") or []
        section = st.selectbox("Section", [None, *sections], format_func=lambda s: s or "All")
    if query.strip():
        with st.spinner("Searching photos…"):
            show_hits(
                call_api(
                    "POST", "/search/text", json={"query": query, "top_k": 9, "section": section}
                )
            )


def choose_query_photo(photo_id: int) -> None:
    st.session_state["query_photo_id"] = photo_id


def image_search_tab() -> None:
    st.caption("Find similar pieces from a garment photo. Upload a PNG or JPEG, or try an example.")
    upload = st.file_uploader(
        "Upload a photo of a garment", type=["png", "jpg", "jpeg"], max_upload_size=10
    )
    examples = call_api("GET", "/search/examples") or []
    if examples:
        chosen = st.session_state.get("query_photo_id") is not None
        with st.expander("…or pick an example photo", expanded=upload is None and not chosen):
            st.caption(
                "These are H&M garments that are not in this catalog, so every result is a "
                "different product that looks alike."
            )
            for start in range(0, len(examples), QUERY_PHOTO_COLUMNS):
                columns = st.columns(QUERY_PHOTO_COLUMNS)
                for column, example in zip(
                    columns, examples[start : start + QUERY_PHOTO_COLUMNS], strict=False
                ):
                    with column:
                        if (photo := query_photo(example["photo_id"])) is not None:
                            st.image(photo, width="stretch")
                        st.button(
                            example["product_type"],
                            key=f"query_photo_{example['photo_id']}",
                            help=f"{example['name']} · {example['colour']}",
                            on_click=choose_query_photo,
                            args=(example["photo_id"],),
                            width="stretch",
                        )

    if upload is not None:
        name, content = upload.name, upload.getvalue()
    elif (photo_id := st.session_state.get("query_photo_id")) is not None:
        name, content = f"{photo_id}.jpg", query_photo(photo_id)
    else:
        return
    if content is None:
        st.error("The example photo is no longer available.")
        return
    left, right = st.columns([0.2, 0.8])
    with left:
        st.image(content, caption="Query photo", width="stretch")
    with right, st.spinner("Searching photos…"):
        show_hits(
            call_api("POST", "/search/image", params={"top_k": 9}, files={"file": (name, content)})
        )


def evaluation_chart_spec(rows: list[dict], baseline: float | None) -> dict:
    """Full-holdout MAP bars; the rule marks the strong repeat baseline when available."""
    values = [
        {"Strategy": row["Strategy"], "MAP@12": row["MAP@12"], "Label": f"{row['MAP@12']:.4f}"}
        for row in rows
        if row["MAP@12"] is not None
    ]
    y = {
        "field": "Strategy",
        "type": "nominal",
        "sort": "-x",
        "title": None,
        "axis": {"labelLimit": 220, "ticks": False, "domain": False},
    }
    x = {
        "field": "MAP@12",
        "type": "quantitative",
        "title": "MAP@12 · all held-out purchases",
        "scale": {"zero": True},
        "axis": {"format": ".3f", "tickCount": 5},
    }
    layers = [
        {
            "mark": {"type": "bar", "color": "#9c4935", "cornerRadiusEnd": 2},
            "encoding": {
                "y": y,
                "x": x,
                "tooltip": [{"field": "Strategy"}, {"field": "MAP@12", "format": ".4f"}],
            },
        },
        {
            "mark": {"type": "text", "align": "left", "dx": 6},
            "encoding": {"y": y, "x": x, "text": {"field": "Label"}},
        },
    ]
    if baseline is not None:
        layers.append(
            {
                "data": {"values": [{"baseline": baseline}]},
                "mark": {"type": "rule", "strokeDash": [4, 4], "opacity": 0.7},
                "encoding": {"x": {"field": "baseline", "type": "quantitative"}},
            }
        )
    return {
        "description": "Full-holdout MAP@12 with the baseline as a dashed rule.",
        "data": {"values": values},
        "height": 44 * len(values),
        "layer": layers,
        "config": {"view": {"stroke": None}},
    }


def evaluation_tab() -> None:
    st.title("Model quality")
    st.caption("Measured against what people bought next.")
    evaluation = call_api("GET", "/evaluation", missing_ok=True)
    if evaluation is None:
        return
    scope = evaluation["sample_scope"]
    if evaluation.get("evaluation_role") == "validation":
        st.caption("Earlier validation week; not the final holdout.")
    st.markdown(
        f"**{evaluation['users_evaluated']:,} customers** · "
        f"{evaluation['cutoff']} to {evaluation['holdout_end']} · "
        f"{evaluation['top_k']} recommendations each"
    )
    st.caption(
        "Prepared catalog and customer selection may favor every strategy; "
        "these are sample-level results, not population or leaderboard estimates."
    )
    results = evaluation["results"]
    if not results:
        st.info("No evaluated strategies are available yet.")
        return
    winner = max(results, key=lambda name: results[name]["full_holdout"]["map_at_12"] or 0)
    best = results[winner]["full_holdout"]
    st.subheader(STRATEGY_NAMES.get(winner, winner))
    st.caption("Highest full-holdout MAP in this measured sample.")
    columns = st.columns(3)
    columns[0].metric(
        "Customers reached",
        f"{best['hit_rate']:.1%}" if best.get("hit_rate") is not None else "Not recorded",
    )
    columns[1].metric(
        "Purchased items recovered",
        f"{best['recall_at_12']:.1%}" if best.get("recall_at_12") is not None else "Not recorded",
    )
    columns[2].metric("Ranking quality · MAP@12", f"{best['map_at_12']:.4f}")
    if gain := evaluation.get("learned_vs_repeat"):
        statement = (
            "The interval includes zero; the single-week advantage remains uncertain."
            if (gain["lower_95"] <= 0 <= gain["upper_95"])
            else "Interval covers selected-customer uncertainty only."
        )
        st.caption(
            f"Observed MAP gain vs buy-again: {gain['mean_delta']:+.4f}. "
            f"95% paired-customer interval: {gain['lower_95']:+.4f} to {gain['upper_95']:+.4f}. "
            + statement
        )
    baseline_name = "repeat" if "repeat" in results else "popularity"
    baseline = results.get(baseline_name, {}).get("full_holdout", {}).get("map_at_12")
    rows = [
        {
            "Strategy": STRATEGY_NAMES.get(name, name),
            "MAP@12": values["full_holdout"]["map_at_12"],
            "Hit rate": values["full_holdout"].get("hit_rate"),
            "Recall@12": values["full_holdout"].get("recall_at_12"),
            "vs baseline": values["full_holdout"]["map_at_12"] / baseline if baseline else None,
            "Catalog coverage": values["catalog_coverage"],
            "List fill": values["recommendation_fill"],
        }
        for name, values in results.items()
    ]
    rows.sort(key=lambda row: -(row["MAP@12"] or 0))
    st.vega_lite_chart(evaluation_chart_spec(rows, baseline), width="stretch")
    st.caption(
        f"Dashed rule: {STRATEGY_NAMES.get(baseline_name, baseline_name)}. "
        "Repeat purchases are included for every strategy."
    )
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_config={
            "MAP@12": st.column_config.NumberColumn(format="%.4f"),
            "Hit rate": st.column_config.NumberColumn(format="percent"),
            "Recall@12": st.column_config.NumberColumn(format="percent"),
            "vs baseline": st.column_config.NumberColumn(format="%.2fx"),
            "Catalog coverage": st.column_config.NumberColumn(format="percent"),
            "List fill": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.caption(
        "Hit rate: customers with at least one matching recommendation. "
        "Recall: average share of each customer's purchased items recovered. "
        "MAP@12 measures ranking quality, not a probability of buying."
    )
    index = evaluation.get("content_index")
    if index is None:
        st.caption("Content-based results are missing: the evaluation ran without an image index.")
    elif index["indexed_products"] < scope["catalog_products"]:
        st.caption(
            f"Visual-style results use {index['indexed_products']:,} of "
            f"{scope['catalog_products']:,} product photos. This partial index limits coverage."
        )
    with st.expander("Data, protocol and limits"):
        st.write(
            f"The {scope['prepared_users']:,}-customer sample contains "
            f"{scope['catalog_products']:,} products. Every purchase-derived signal is trained "
            "strictly before the holdout. Cohort and catalog selection may use the full window; "
            "results describe this sample, not the population or a competition leaderboard."
        )
        st.write(
            "Novel-item metrics use the original ranked lists and are diagnostic. "
            "A repeat strategy and a discovery strategy solve different tasks."
        )
        st.json(evaluation.get("repeat_purchase_limitation", {}))


def discovery_page() -> None:
    st.title("Discover")
    st.caption("Describe a piece you have in mind, or start with a photo.")
    mode = st.radio("Search with", ["Description", "Photo"], horizontal=True, key="discovery_mode")
    if mode == "Description":
        text_search_tab()
    else:
        image_search_tab()


def chat_tab() -> None:
    st.title("Shopping assistant")
    health = call_api("GET", "/health")
    if health is None:
        return
    if not health.get("chat_enabled"):
        st.info(
            "The assistant is currently unavailable. "
            "You can still discover pieces by description or photo."
        )
        return
    history: list[dict] = st.session_state.setdefault("chat_history", [])
    for index, turn in enumerate(history):
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("products"):
                show_products(turn["products"], context=f"chat_{index}")

    message = st.chat_input("Ask Clothy, e.g. 'black wide-leg trousers for women under €40'")
    if not message:
        return
    with st.chat_message("user"):
        st.markdown(message)
    with st.chat_message("assistant"), st.spinner("Thinking..."):
        past_turns = [{"role": t["role"], "content": t["content"]} for t in history]
        reply = call_api("POST", "/chat", json={"message": message, "history": past_turns})
        if reply is None:
            return
        st.markdown(reply["text"])
        show_products(reply["products"], context="chat_reply")
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply["text"], "products": reply["products"]})


def main() -> None:
    st.set_page_config(page_title="Clothy · Fashion discovery", page_icon="C", layout="wide")
    st.html("<style>" + Path(__file__).with_name("ui.css").read_text() + "</style>")
    health = call_api("GET", "/health")
    st.html('<div class="clothy-brand">Clothy<span>FASHION DISCOVERY</span></div>')
    if health is None:
        st.info("Clothy cannot connect right now. Please try again shortly.")
        if st.button("Retry connection"):
            st.rerun()
        return
    indexed = f"{health['indexed_products']:,}"
    total = f" of {health['catalog_products']:,}" if health.get("catalog_products") else ""
    st.caption(f"H&M 2020 · Photo search: {indexed}{total} pieces")
    st.session_state["service_health"] = health
    page = st.navigation(
        [
            st.Page(recommendations_tab, title="For you", default=True),
            st.Page(discovery_page, title="Discover", url_path="discover"),
            st.Page(knowledge_graph_tab, title="Product connections", url_path="connections"),
            st.Page(evaluation_tab, title="Model quality", url_path="quality"),
            st.Page(chat_tab, title="Assistant", url_path="assistant"),
        ],
        position="top",
    )
    page.run()


if __name__ == "__main__":
    main()
