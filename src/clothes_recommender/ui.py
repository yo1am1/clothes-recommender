"""Streamlit demo UI: a thin client over the HTTP API.

Run with: uv run streamlit run src/clothes_recommender/ui.py
"""

import httpx
import streamlit as st

from clothes_recommender.config import get_settings

GRID_COLUMNS = 5
# Catalog photos are mostly portrait; a fixed width keeps cards compact so captions stay visible.
PHOTO_WIDTH = 160


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


@st.cache_data(max_entries=2000)
def product_photo(product_id: int) -> bytes | None:
    # Fetched server-side rather than by URL: in Docker the browser can't resolve the API host.
    response = api_client().get(f"/products/{product_id}/image")
    return response.content if response.is_success else None


def show_products(products: list[dict], scores: list[float] | None = None) -> None:
    columns = st.columns(GRID_COLUMNS)
    for i, product in enumerate(products):
        with columns[i % GRID_COLUMNS]:
            if photo := product_photo(product["product_id"]):
                st.image(photo, width=PHOTO_WIDTH)
            st.markdown(product["name"])
            caption = f"€{product['price']:.2f} · {product['product_type']} · {product['colour']}"
            if scores is not None:
                caption += f" · score {scores[i]:.3f}"
            st.caption(caption)


def show_hits(hits: list[dict] | None) -> None:
    if hits:
        show_products([h["product"] for h in hits], [h["score"] for h in hits])
    elif hits is not None:
        st.info("No matching products.")


def recommendations_tab() -> None:
    user_ids = call_api("GET", "/users") or []
    if not user_ids:
        return
    left, right = st.columns([0.3, 0.7])
    with left:
        user_id = st.selectbox("User", user_ids)
        strategy = st.radio(
            "Strategy",
            ["content", "collaborative"],
            format_func={
                "content": "Content-based (CLIP taste vector)",
                "collaborative": "Collaborative (similar shoppers)",
            }.get,
        )
    profile = call_api("GET", f"/users/{user_id}")
    with right:
        if profile:
            user = profile["user"]
            age = user["age"] if user["age"] is not None else "unknown"
            st.markdown(f"**Customer #{user['user_id']}** · age {age}")
            with st.expander(f"Purchase history ({len(profile['purchases'])} items)"):
                show_products([p["product"] for p in profile["purchases"]])

    if strategy == "collaborative":
        similar = call_api("GET", f"/users/{user_id}/similar")
        if similar:
            st.markdown(
                "**Most similar shoppers:** "
                + ", ".join(f"#{s['user_id']} ({s['similarity']:.2f})" for s in similar)
            )
    st.subheader("Recommended for this user")
    show_hits(call_api("GET", f"/users/{user_id}/recommendations", params={"strategy": strategy}))


def text_search_tab() -> None:
    query = st.text_input(
        "Describe what you are looking for", placeholder="red floral summer dress"
    )
    if query.strip():
        show_hits(call_api("POST", "/search/text", json={"query": query}))


def image_search_tab() -> None:
    upload = st.file_uploader("Upload a photo of a garment", type=["png", "jpg", "jpeg"])
    if upload:
        st.image(upload, width=PHOTO_WIDTH)
        show_hits(
            call_api("POST", "/search/image", files={"file": (upload.name, upload.getvalue())})
        )


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
    tabs = st.tabs(["Recommendations", "Text → image search", "Image search", "Shopping assistant"])
    for tab, render in zip(
        tabs, [recommendations_tab, text_search_tab, image_search_tab, chat_tab], strict=True
    ):
        with tab:
            render()


if __name__ == "__main__":
    main()
