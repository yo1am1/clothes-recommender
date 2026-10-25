"""Streamlit demo UI: a thin client over the HTTP API.

Run with: uv run streamlit run src/clothes_recommender/ui.py
"""

import math
import re

import httpx
import streamlit as st

from clothes_recommender.config import get_settings

GRID_COLUMNS = 3


@st.cache_resource
def api_client() -> httpx.Client:
    # CLIP inference on CPU can take a few seconds on a cold start.
    return httpx.Client(base_url=get_settings().api_url, timeout=120.0)


def call_api(method: str, path: str, **kwargs) -> dict | list | None:
    """Call the API; on failure show the server's explanation instead of a stack trace."""
    try:
        response = api_client().request(method, path, **kwargs)
    except httpx.HTTPError as error:
        st.error(f"API unreachable at {get_settings().api_url}: {error}")
        return None
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:  # e.g. a plain-text 500 from a crashed worker
            detail = response.text
        st.error(detail)
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


def recommendation_reason(reasons: list[str]) -> str | None:
    """Summarize the strongest supplied evidence; keep raw evidence in the expander."""
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
) -> None:
    for start in range(0, len(products), GRID_COLUMNS):
        columns = st.columns(GRID_COLUMNS)
        for offset, product in enumerate(products[start : start + GRID_COLUMNS]):
            i = start + offset
            with columns[offset]:
                st.caption("Product images coming soon.")
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


def show_hits(hits: list[dict] | None) -> None:
    if hits:
        show_products(
            [h["product"] for h in hits],
            [h["score"] for h in hits],
            [h.get("reasons", []) for h in hits],
        )
    elif hits is not None:
        st.info("No matching products.")


def request_customer_browse(key: str) -> None:
    """A toggle requests confirmation; only Continue activates full-list browsing."""
    if st.session_state[f"{key}_browse_all"]:
        st.session_state["customer_browse_pending"] = key
        st.session_state[f"{key}_browse_all"] = False
    else:
        st.session_state[f"{key}_all_confirmed"] = False


def enable_customer_browse(key: str) -> None:
    st.session_state[f"{key}_all_confirmed"] = True


@st.dialog("Browse all customers?", on_dismiss="rerun")
def confirm_customer_browse(key: str) -> None:
    st.write(
        "The full dataset is not curated. Some customer histories and recommendation "
        "results may be sparse. You can return to the curated examples at any time."
    )
    if st.button("Continue", type="primary", on_click=enable_customer_browse, args=(key,)):
        st.rerun()
    if st.button("Keep curated examples"):
        st.rerun()


def customer_selector(label: str, key: str) -> dict | None:
    """Start with a few explained examples; fetch the full ID list only on request."""
    examples = call_api("GET", "/demo/customers") or []
    examples_by_id = {example["user_id"]: example for example in examples[:10]}
    browse_all = st.session_state.get(f"{key}_all_confirmed", False)
    st.session_state[f"{key}_browse_all"] = browse_all
    st.toggle(
        "Browse all customers",
        key=f"{key}_browse_all",
        on_change=request_customer_browse,
        args=(key,),
    )
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
        key=f"{key}_{'all' if browse_all else 'examples'}",
        format_func=lambda uid: (
            str(uid) if browse_all else f"{examples_by_id[uid]['label']} (#{uid})"
        ),
    )
    example = examples_by_id.get(user_id, {"user_id": user_id})
    if example.get("description"):
        st.caption(example["description"])
    return example


def recommendations_tab() -> None:
    left, right = st.columns([0.55, 0.45])
    with left:
        example = customer_selector("User", "recommendation_user")
        if example is None:
            return
        user_id = example["user_id"]
        strategy = st.radio(
            "Strategy",
            ["graph", "content", "collaborative"],
            format_func={
                "content": "Content-based (CLIP taste vector)",
                "collaborative": "Collaborative (similar shoppers)",
                "graph": "Knowledge graph (purchase connections)",
            }.get,
        )
    profile = call_api("GET", f"/users/{user_id}")
    with right:
        if profile:
            user = profile["user"]
            purchases = profile["purchases"]
            distinct = len({purchase["product"]["product_id"] for purchase in purchases})
            st.text(f"Customer #{user['user_id']}")
            st.caption(f"{len(purchases)} purchase records · {distinct} distinct products")
            if purchases:
                with st.expander("Recent purchases (up to 10)"):
                    st.dataframe(
                        [
                            {
                                "Product": purchase["product"]["name"],
                                "Type": purchase["product"]["product_type"],
                                "Price (€)": purchase["product"]["price"],
                                "Date": purchase.get("purchase_date", ""),
                            }
                            for purchase in purchases[:10]
                        ],
                        hide_index=True,
                        width="stretch",
                    )

    if strategy == "collaborative":
        similar = call_api("GET", f"/users/{user_id}/similar")
        if similar:
            st.markdown(
                "**Most similar shoppers:** "
                + ", ".join(f"#{s['user_id']} ({s['similarity']:.2f})" for s in similar)
            )
    st.subheader("Recommended for this user")
    with st.spinner("Finding recommendations…"):
        show_hits(
            call_api("GET", f"/users/{user_id}/recommendations", params={"strategy": strategy})
        )


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
        ["complement", "alternative"],
        horizontal=True,
        key="graph_related_kind",
        format_func={
            "complement": "Bought together",
            "alternative": "Higher-price alternatives",
        }.get,
    )
    if kind == "complement":
        st.caption("Co-purchased products from different product groups.")
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
        show_hits(related)


def text_search_tab() -> None:
    st.info("Text → image search is coming soon.")


def image_search_tab() -> None:
    st.info("Image search is coming soon.")


def chat_tab() -> None:
    history: list[dict] = st.session_state.setdefault("chat_history", [])
    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("products"):
                show_products(turn["products"])

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
        show_products(reply["products"])
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply["text"], "products": reply["products"]})


def main() -> None:
    st.set_page_config(page_title="Clothy · Fashion Recommender", page_icon="👗", layout="wide")
    st.title("👗 Clothy: multimodal fashion recommender")
    tabs = st.tabs(
        [
            "Recommendations",
            "Knowledge graph",
            "Text → image search",
            "Image search",
            "Shopping assistant",
        ]
    )
    for tab, render in zip(
        tabs,
        [recommendations_tab, knowledge_graph_tab, text_search_tab, image_search_tab, chat_tab],
        strict=True,
    ):
        with tab:
            render()
    # Pop before opening: Esc/X reruns without reopening or enabling full-list mode.
    if pending := st.session_state.pop("customer_browse_pending", None):
        confirm_customer_browse(pending)


if __name__ == "__main__":
    main()
