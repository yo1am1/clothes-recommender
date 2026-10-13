"""Recommender System
This script implements a Streamlit application for a clothes recommender system.
It allows users to search for products using text or images, get personalized recommendations,
and interact with a chatbot for queries related to clothing items.
"""

from datetime import date, datetime
import json
import os
import pathlib
from langchain_core.structured_query import StructuredQuery
import pandas as pd
from PIL import Image
import pinecone
from streamlit.runtime.uploaded_file_manager import UploadedFile
import torch
import streamlit as st
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from transformers.models.clip import CLIPProcessor, CLIPModel


from utils.embeddings.azure_embeddings import __get_openai_model
from utils.prompts import init_message_prompt, eval_prompt
from utils.pinecone_worker import (
    get_image_by_id_from_s3,
    recommend_for_user,
    image_search,
    text_search,
)
from utils.collaborative_recommendations import (
    get_collaborative_recommendations,
    get_similar_users,
)

# Set up Torch classes path
if torch.classes.__file__ is not None:
    torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
else:
    torch.classes.__path__ = []

# Define file paths and data directories
DATA_DIR: pathlib.Path = pathlib.Path(__file__).parent / "data"
USER_PROFILES_PATH: pathlib.Path = DATA_DIR / "foo_customers.csv"
PURCHASE_HISTORY_PATH: pathlib.Path = DATA_DIR / "foo_purchase_history.csv"
PREPROCESSED_CSV: pathlib.Path = (
    DATA_DIR / "new_dataset" / "foo-amazon-products-merged-updated.csv"
)

S3_BUCKET = "dev-ai-sandbox"
S3_IMAGES_PREFIX = "recommendation-engine-data-demo"


# Define metadata field information for query construction
metadata_field_info: list[AttributeInfo] = [
    AttributeInfo(
        name="categfilter_dictry",
        description="Product category (e.g., jeans, shoes, t-shirts & polos, western wear, lingerie & nightwear).",
        type="string",
    ),
    AttributeInfo(
        name="ratings",
        description="Average product rating on Amazon (e.g., 4.5, 3.8, 4.0).",
        type="float",
    ),
    AttributeInfo(
        name="price",
        description="Product price (e.g., 999.0, 1499.34, 2999.5).",
        type="float",
    ),
    AttributeInfo(
        name="brand",
        description="Product brand (e.g., nike, adidas, puma).",
        type="string",
    ),
    AttributeInfo(
        name="color",
        description="Product color (e.g., red, blue, green).",
        type="string",
    ),
]


# ============================================================================
# CACHING FUNCTIONS - Critical for performance optimization
# ============================================================================


@st.cache_resource(show_spinner="Loading CLIP models...", ttl=60 * 5)
def load_clip_models():
    """Load CLIP model and processor once and cache them."""
    _model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
    _processor = CLIPProcessor.from_pretrained(
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", use_fast=True
    )
    return _model, _processor


@st.cache_resource(show_spinner="Connecting to Pinecone...")
def get_pinecone_client():
    """Initialize Pinecone client once and cache it."""
    return pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


@st.cache_resource(show_spinner="Initializing AI model...")
def get_chat_model():
    """Get OpenAI chat model with caching."""
    return __get_openai_model()


@st.cache_data(show_spinner="Loading user data...", ttl=60 * 5)
def load_user_profiles() -> pd.DataFrame:
    """Load user profiles from CSV."""
    if not USER_PROFILES_PATH.exists():
        st.warning(f"User profiles file not found: {USER_PROFILES_PATH}")
        return pd.DataFrame()
    # Reduced from 500 to 200 for better performance
    return pd.read_csv(USER_PROFILES_PATH, nrows=200)


@st.cache_data(show_spinner="Loading purchase history...", ttl=60 * 5)
def load_purchase_history() -> pd.DataFrame:
    """Load purchase history from CSV."""
    if not PURCHASE_HISTORY_PATH.exists():
        st.warning(f"Purchase history file not found: {PURCHASE_HISTORY_PATH}")
        return pd.DataFrame()
    # Reduced from 5000 to 2000 for better performance
    return pd.read_csv(PURCHASE_HISTORY_PATH, nrows=2000)


def __session_init() -> None:
    """Initialize session state for conversation and photo flag."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "ai", "content": "Hi! I am Clothy. How can I help you?"}
        ]
    if "requires_photos" not in st.session_state:
        st.session_state["requires_photos"] = False


def __get_response(chain, user_input, context_str, history_str) -> str:
    """Invoke the model chain to get a response."""
    return chain.invoke(
        {
            "query": user_input,
            "context": context_str,
            "history": history_str,
        }
    )


def __add_human_message() -> None:
    """Add a human (user) message to the conversation."""
    st.session_state.messages.append(
        {"role": "user", "content": st.session_state.chat_input}
    )
    st.session_state["requires_photos"] = False


def __add_ai_message(ai_text, images=None) -> None:
    """Add an AI message (optionally with images) to the conversation."""
    if images:
        st.session_state.messages.append(
            {"role": "ai", "content": ai_text, "images": images}
        )
    else:
        st.session_state.messages.append({"role": "ai", "content": ai_text})
    st.session_state["requires_photos"] = False


def __validate_response(chain, ai_text, user_input):
    """Validate the AI response using an evaluation chain."""
    return chain.invoke({"ai_text": ai_text, "user_input": user_input})


def __translate_metadata_filter(filter_obj) -> dict:
    """Recursively build a metadata filter dict from a filter object."""
    if filter_obj is None:
        return {}
    if hasattr(filter_obj, "operator"):
        op_name = filter_obj.operator.name.lower()  # "and" or "or"
        sub_filters = [__translate_metadata_filter(arg) for arg in filter_obj.arguments]
        sub_filters = [f for f in sub_filters if f]
        if not sub_filters:
            return {}
        if op_name == "and":
            return sub_filters[0] if len(sub_filters) == 1 else {"$and": sub_filters}
        elif op_name == "or":
            return sub_filters[0] if len(sub_filters) == 1 else {"$or": sub_filters}
        else:
            return {}
    else:
        comparator = filter_obj.comparator
        comp_name = (
            comparator.name.lower()
            if hasattr(comparator, "name")
            else str(comparator).lower()
        )
        attribute = filter_obj.attribute
        value = filter_obj.value
        if comp_name == "eq":
            return {attribute: {"$eq": value}}
        elif comp_name == "gt":
            return {attribute: {"$gt": value}}
        elif comp_name == "lt":
            return {attribute: {"$lt": value}}
        elif comp_name == "gte":
            return {attribute: {"$gte": value}}
        elif comp_name == "lte":
            return {attribute: {"$lte": value}}
        elif comp_name == "contain":
            return {attribute: {"$in": [value]}}
        else:
            return {}


def __split_images_by_gender(images, captions):
    """
    Split images and captions into two groups based on whether the caption contains "men", "man", or "polo" (for men's)
    or "women" (for women's).

    Parameters:
      images: List of PIL.Image objects.
      captions: List of caption strings.

    Returns:
      Tuple: ((men_images, men_captions), (women_images, women_captions))
    """
    men_images, men_captions = [], []
    women_images, women_captions = [], []
    for img, cap in zip(images, captions):
        cap_lower = cap.lower().strip()
        # Check if any of the men's keywords are in the caption
        if any(keyword in cap_lower for keyword in ["men", "man", "polo"]) and not any(
            keyword in cap_lower for keyword in ["woman", "women"]
        ):
            men_images.append(img)
            men_captions.append(cap)
        elif any(keyword in cap_lower for keyword in ["woman", "women"]):
            women_images.append(img)
            women_captions.append(cap)
    return (men_images, men_captions), (women_images, women_captions)


##############################
# Main App Layout
##############################


def init_page() -> None:
    """Initialize the Streamlit app with multiple tabs."""
    st.set_page_config(
        page_title="[DEMO] Recommender System", page_icon="👗", layout="wide"
    )

    st.title("[DEMO] Recommender System")

    # Load cached resources
    model, processor = load_clip_models()
    pc = get_pinecone_client()
    user_data: pd.DataFrame = load_user_profiles()
    purchase_data: pd.DataFrame = load_purchase_history()

    __session_init()

    # Reordered tabs and corresponding content blocks
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "User Recommendations",
            "Text to Image Search",
            "Search by Image",
            "Chatbot and Self-Query Retrieval",
        ]
    )

    with tab1:
        col1, col2 = st.columns([0.7, 0.3], border=True, vertical_alignment="center")
        with col1:
            st.subheader(
                "Get personalized recommendations based on your purchase history."
            )
            st.badge(
                "Recommendations are based on the full purchase history of the selected user.",
                icon="🔍",
            )

        with col2:
            rec_type = st.radio(
                "Please, chose the recommendation type:",
                options=["Item-based", "Collaborative-based"],
                horizontal=True,
                index=0,
                key="rec_type",
            )

        if rec_type == "Item-based":
            col1, col2 = st.columns([0.3, 0.7])
            with col1:
                # Input for filtering purchase history by date range
                date_from: date = st.date_input(
                    "From Date",
                    value=pd.to_datetime(datetime.now()).date()
                    - pd.DateOffset(days=365),
                    min_value=pd.to_datetime(datetime.now()).date()
                    - pd.DateOffset(days=365),
                    max_value=pd.to_datetime(datetime.now()).date(),
                )
                date_to: date = st.date_input(
                    "To Date",
                    value=pd.to_datetime(datetime.now()).date(),
                )

                # Select a user ID from the customer data
                user_ids: list[str] = (
                    user_data["user_id"].drop_duplicates().astype(str).tolist()
                    if not user_data.empty
                    else []
                )
                selected_user_id = (
                    st.selectbox("Select a User ID", options=user_ids)
                    if user_ids
                    else "None"
                )
                rec_button = st.button(
                    "Get Recommendations", type="primary", use_container_width=True
                )
            with col2:
                # Display last purchases for the selected user within the chosen date range
                if selected_user_id != "None":
                    # Ensure the 'purchase_date' column is in datetime format
                    if "purchase_date" in purchase_data.columns:
                        purchase_data["purchase_date"] = pd.to_datetime(
                            purchase_data["purchase_date"]
                        )
                    else:
                        st.error(
                            "'purchase_date' column not found in purchase history data."
                        )
                        st.dataframe(purchase_data)
                        return
                    # Filter for the selected user and date range
                    filtered_purchases = purchase_data[
                        (purchase_data["user_id"] == int(selected_user_id))
                        & (purchase_data["purchase_date"] >= pd.to_datetime(date_from))
                        & (purchase_data["purchase_date"] <= pd.to_datetime(date_to))
                    ]
                    # Sort by purchase date descending (most recent first)
                    filtered_purchases: pd.DataFrame = filtered_purchases.sort_values(
                        by="purchase_date", ascending=False
                    )

                    user = user_data[user_data["user_id"] == int(selected_user_id)]
                    st.data_editor(
                        user,
                        use_container_width=True,
                        hide_index=True,
                        key="user_id",
                    )

                    st.data_editor(
                        data=filtered_purchases,
                        use_container_width=False,
                        hide_index=True,
                        key="filtered_purchase_history",
                        column_config={
                            "user_id": "User ID",
                            "product_id": "Product ID",
                            "purchase_id": "Purchase ID",
                            "purchase_date": "Purchase Date",
                            "product_category": "Product Category",
                            "brand": "Brand",
                            "color": "Color",
                            "price": "Price",
                            "ratings": "Ratings",
                            "product_name": "Product Name",
                            "product_url": "Product URL",
                        },
                        disabled=True,
                    )

            # When "Get Recommendations" is clicked
            st.divider()
            if rec_button:
                if selected_user_id == "None":
                    st.warning("No user IDs found in the customers dataset!")
                else:
                    with st.spinner("Generating recommendations..."):
                        col1, col2 = st.columns([0.3, 0.7])
                        with col1:
                            rec_id_cat, rec_metas, rec_filter = recommend_for_user(
                                user_id=selected_user_id,
                                user_data=user_data,
                                purchase_data=purchase_data,
                                top_n=6,
                                processor=processor,
                                model=model,
                                pc=pc,
                            )
                            st.write("Recommended product IDs:")
                            st.json(rec_id_cat, expanded=False)
                            st.write("Recommended metadata filter:")
                            st.json(rec_filter, expanded=3)
                            st.write("Recommended metadata:")
                            st.json(rec_metas, expanded=3)

                        with col2:
                            # Load images for the entire recommendation list
                            images = [
                                get_image_by_id_from_s3(
                                    image_id=rec[0],
                                    category=rec[1],
                                )
                                for rec in rec_id_cat
                            ]
                            captions = [
                                rec.get("product_name")
                                for rec in rec_metas
                                if rec.get("product_name") is not None
                            ]
                            st.write("Recommended images:")

                            if images:
                                (men_images, men_caps), (women_images, women_caps) = (
                                    __split_images_by_gender(images, captions)
                                )
                                if men_images:
                                    st.write("Men:")
                                    st.image(men_images, caption=men_caps, width=200)
                                else:
                                    st.write("No Men recommendations found.")
                                if women_images:
                                    st.write("Women:")
                                    st.image(
                                        women_images, caption=women_caps, width=200
                                    )
                                else:
                                    st.write("No Women recommendations found.")
                            else:
                                st.warning(
                                    "No images found for the recommended IDs (or images failed to load)."
                                )
        else:
            col1, col2 = st.columns([0.3, 0.7])
            with col1:
                date_from: date = st.date_input(
                    "From Date",
                    value=datetime.now().date() - pd.DateOffset(months=5),
                    min_value=datetime.now().date() - pd.DateOffset(days=365),
                    max_value=datetime.now().date(),
                )
                date_to: date = st.date_input("To Date", value=datetime.now().date())

                user_ids: list[str] = user_data["user_id"].astype(str).unique().tolist()
                selected_user_id = st.selectbox("Select a User ID", user_ids)
                rec_button = st.button(
                    "Get Recommendations", type="primary", use_container_width=True
                )

            with col2:
                if selected_user_id:
                    purchase_data["purchase_date"] = pd.to_datetime(
                        purchase_data["purchase_date"]
                    )
                    filtered = purchase_data[
                        (purchase_data.user_id == int(selected_user_id))
                        & (
                            purchase_data.purchase_date.between(
                                pd.to_datetime(date_from), pd.to_datetime(date_to)
                            )
                        )
                    ].sort_values("purchase_date", ascending=False)

                    st.data_editor(
                        user_data[user_data.user_id == int(selected_user_id)],
                        hide_index=True,
                        use_container_width=True,
                    )
                    st.data_editor(
                        filtered,
                        hide_index=True,
                        key="purchase_hist",
                        use_container_width=False,
                    )

            # ----------- Perform collaborative recommendations --------------- #
            st.divider()
            if rec_button and selected_user_id:
                with st.spinner("Finding similar users & products…"):
                    #  1) Similar users
                    similars = get_similar_users(int(selected_user_id), 3)
                    st.subheader("Most similar users")
                    if similars:
                        sim_table = pd.DataFrame(similars)[
                            ["user_id", "similarity", "metadata"]
                        ]
                        st.table(sim_table.style.format({"similarity": "{:.3f}"}))
                    else:
                        st.write("No similar users found.")

                    #  2) Product recommendations
                    rec_ids = get_collaborative_recommendations(
                        int(selected_user_id),
                        purchase_data,
                        10,
                    )
                    st.subheader("Recommended product IDs")
                    st.json(rec_ids, expanded=False)

                    #  3) Retrieve metadata + images
                    if rec_ids:
                        meta_rows = purchase_data[
                            purchase_data.product_id.isin(rec_ids)
                        ]

                        seen_ids = set()
                        images = []
                        for _, row in meta_rows.iterrows():
                            if row.product_id not in seen_ids:
                                seen_ids.add(row.product_id)
                                images.append(
                                    get_image_by_id_from_s3(
                                        image_id=row.product_id,
                                        category=row.product_category,
                                    )
                                )
                        captions = meta_rows.product_name.fillna("").tolist()

                        (men_imgs, men_caps), (women_imgs, women_caps) = (
                            __split_images_by_gender(images, captions)
                        )

                        if men_imgs:
                            st.markdown("**Men**")
                            st.image(men_imgs, caption=men_caps, width=200)
                        if women_imgs:
                            st.markdown("**Women**")
                            st.image(women_imgs, caption=women_caps, width=200)
                        if not men_imgs and not women_imgs:
                            st.write("No images available for these products.")

    with tab2:
        st.markdown(
            "## Text to Image Search\nEnter a text-based query to find products."
        )
        query: str = st.text_input("Search query", placeholder="e.g., red dress")
        if query and query.strip() != "":
            with st.spinner("Searching..."):
                # Perform the search using the text query
                response = text_search(
                    query_text=query,
                    top_k=10,
                    retrieve_from_s3=True,
                    processor=processor,
                    model=model,
                    pc=pc,
                )

                st.write("Search Response:")
                st.json(response, expanded=False)
                st.image(
                    [val.get("image") for val in response],
                    caption=[rec["metadata"].get("product_name") for rec in response],
                    width=150,
                )

    with tab3:
        st.markdown(
            "## Search by Image\nUpload an image (PNG/JPG/JPEG) to find visually similar products."
        )
        col1, col2 = st.columns([0.7, 0.3], vertical_alignment="center")
        with col1:
            uploaded_image: UploadedFile | None = st.file_uploader(
                "Upload an image", type=["png", "jpg", "jpeg"]
            )
        if uploaded_image:
            with col2:
                st.image(uploaded_image, caption="Uploaded Image", width=150)
            if st.button("Search by Image", type="primary"):
                with st.spinner("Searching..."):
                    # Open and convert the uploaded image to a PIL image
                    image = Image.open(uploaded_image).convert("RGB")
                    response = image_search(
                        image,
                        processor,
                        model,
                        pc,
                        index_name="clip-test",
                        namespace="image-search",
                        top_k=10,
                        retrieve_from_s3=True,
                    )

                    st.write("Search Response:")
                    st.json(response, expanded=False)
                    st.image(
                        [val.get("image") for val in response],
                        caption=[
                            rec["metadata"].get("product_name") for rec in response
                        ],
                        width=150,
                    )

    with tab4:
        st.markdown(
            "## Chatbot and Self-Query Retrieval\nAsk questions or use metadata-based queries (brand, color, etc.)."
        )
        chat_model = get_chat_model()
        if chat_model is None:
            st.warning("Chatbot is not available due to missing configuration.")
        else:
            document_content_description = "Brief summary of the products on marketplace with all required information."
            prompt = get_query_constructor_prompt(
                document_content_description, metadata_field_info
            )
            output_parser: StructuredQueryOutputParser = (
                StructuredQueryOutputParser.from_components()
            )
            query_constructor = prompt | chat_model | output_parser
            msg_chain = (
                init_message_prompt.model_copy() | chat_model | StrOutputParser()
            )
            eval_chain = eval_prompt.model_copy() | chat_model | JsonOutputParser()

            # Display conversation history
            with st.container():
                for message in st.session_state["messages"]:
                    if message["role"] == "ai":
                        with st.chat_message(name="assistant"):
                            st.write(message["content"])
                            if "images" in message and message["images"]:
                                st.image(message["images"], width=150)
                    elif message["role"] == "user":
                        with st.chat_message(name="human"):
                            st.write(message["content"])

            # Get new user input
            user_input: str | None = st.chat_input(
                "Talk to Clothy...", key="chat_input", on_submit=__add_human_message
            )
            if user_input:
                structured_query: StructuredQuery = query_constructor.invoke(
                    {"query": user_input}
                )
                metadata_filter = (
                    __translate_metadata_filter(structured_query.filter) or None
                )

                filtered_context = text_search(
                    query_text=structured_query.query,
                    top_k=5,
                    retrieve_from_s3=False,
                    processor=processor,
                    model=model,
                    pc=pc,
                    filter_dict=metadata_filter,
                )

                # Remove debug prints for production
                # from icecream import ic
                # ic(filtered_context)
                # ic(structured_query.query)
                # ic(metadata_filter)

                with (
                    st.chat_message(name="assistant"),
                    st.spinner("Generating response..."),
                ):
                    col_text, col_images = st.columns([0.7, 0.3])
                    with col_text:
                        model_response: str = __get_response(
                            chain=msg_chain,
                            user_input=user_input,
                            context_str=(
                                filtered_context
                                if filtered_context is not None
                                else "No context found."
                            ),
                            history_str=st.session_state["messages"],
                        )
                        validate_resp = __validate_response(
                            chain=eval_chain,
                            ai_text=model_response,
                            user_input=user_input,
                        )
                        # ic(validate_resp)  # Debug only
                        # Attempt to parse validation response to set requires_photos flag
                        if isinstance(validate_resp, str):
                            try:
                                validate_resp = json.loads(validate_resp)
                            except json.JSONDecodeError:
                                st.error("Invalid JSON response from the model.")
                                return
                            st.session_state["requires_photos"] = (
                                False
                                if validate_resp.get("requires_photos") == "False"
                                else True
                            )
                        elif (
                            isinstance(validate_resp, dict)
                            and "requires_photos" in validate_resp
                        ):
                            st.session_state["requires_photos"] = (
                                False
                                if validate_resp.get("requires_photos") == "False"
                                else True
                            )
                        st.write(model_response)

                    # Ensure images is defined even if not retrieved.
                    images = []
                    if st.session_state.get("requires_photos", False):
                        with col_images:
                            matches = filtered_context.get("matches", [])
                            if matches:
                                images = [
                                    get_image_by_id_from_s3(
                                        image_id=match["id"],
                                        category=match["metadata"].get("category"),
                                    )
                                    for match in matches
                                ]
                                # ic(images)  # Debug only
                            else:
                                st.write("No images found in filtered context.")
                            if images:
                                st.image(images, width=150)

                        __add_ai_message(model_response, images=images)
                    else:
                        __add_ai_message(model_response)


if __name__ == "__main__":
    init_page()
