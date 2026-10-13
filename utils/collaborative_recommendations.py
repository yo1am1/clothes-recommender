import os
import json
import logging
import pathlib
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pinecone
from transformers.models.clip import CLIPModel, CLIPProcessor

from utils.pinecone_worker import get_text_embedding  # unchanged

# --------------------------------------------------------------------------- #
#  Setup
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format="%(message)s")

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
USER_PROFILES_PATH = DATA_DIR / "foo_customers.csv"
PURCHASE_HISTORY_PATH = DATA_DIR / "foo_purchase_history.csv"

INDEX_NAME = "clip-test"
NAMESPACE = "user-vectors"

#  (kept your original CLIP model – easy swap later if desired)
model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
processor = CLIPProcessor.from_pretrained(
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", use_fast=True
)

pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)


# --------------------------------------------------------------------------- #
#  Data helpers
# --------------------------------------------------------------------------- #
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(USER_PROFILES_PATH),
        pd.read_csv(PURCHASE_HISTORY_PATH),
    )


# ---------- document builder (also used for debug prints) ------------------ #
def build_user_vector_text(
    user_id: int,
    user_df: pd.DataFrame,
    purchase_df: pd.DataFrame,
) -> str:
    """Re-create the exact text string we embed for *user_id*."""
    u = user_df[user_df.user_id == user_id].iloc[0]

    city = str(u.get("city", "") or "")
    country = str(u.get("country", "") or "")
    gender = str(u.get("gender", "") or "")

    purchases = purchase_df[purchase_df.user_id == user_id]
    cats = [c for c, _ in Counter(purchases["product_category"].dropna()).most_common()]
    brands = [b for b, _ in Counter(purchases["brand"].dropna()).most_common(3)]

    cats_s = ",".join(c.replace(" ", "_") for c in cats)
    brands_s = ",".join(brands)

    return f"{city},{country},{gender};{cats_s};{brands_s}"


def pretty_user(user_id: int, user_df: pd.DataFrame) -> str:
    r = user_df[user_df.user_id == user_id].iloc[0]
    return f"{user_id:<5} | {r.gender or '?'} | {r.city},{r.country}"


# --------------------------------------------------------------------------- #
#  Vector-building + upsert
# --------------------------------------------------------------------------- #
def add_user_to_collection(
    user_id: int,
    document: str,
    user_df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    if metadata is None:
        metadata = {}

    row = user_df[user_df.user_id == user_id]
    if not row.empty:
        metadata.setdefault("gender", row.gender.iloc[0])

    vector = get_text_embedding(document, processor, model)
    index.upsert(
        [{"id": str(user_id), "values": vector, "metadata": metadata}],
        namespace=NAMESPACE,
    )


def build_user_vectors(user_df: pd.DataFrame, purchase_df: pd.DataFrame) -> None:
    for uid in user_df.user_id.unique():
        doc = build_user_vector_text(uid, user_df, purchase_df)
        meta = {
            "purchase_count": int(purchase_df[purchase_df.user_id == uid].shape[0]),
            "city": user_df[user_df.user_id == uid].city.iloc[0],
            "country": user_df[user_df.user_id == uid].country.iloc[0],
            "gender": user_df[user_df.user_id == uid].gender.iloc[0],
        }
        add_user_to_collection(uid, doc, user_df, meta)
    logging.info("Finished building user vectors.")


# --------------------------------------------------------------------------- #
#  Similarity + recommendations
# --------------------------------------------------------------------------- #
def get_similar_users(
    user_id: int,
    n_similar: int = 5,
    filter_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    fetch = index.fetch(ids=[str(user_id)], namespace=NAMESPACE)
    vec = fetch.vectors.get(str(user_id))
    if not vec:
        logging.warning("User %s not found in index.", user_id)
        return []

    query = index.query(
        vector=vec.values,
        top_k=n_similar + 1,
        include_metadata=True,
        namespace=NAMESPACE,
    )

    out = []
    for m in query.get("matches", []):
        if m["id"] == str(user_id):
            continue
        if filter_metadata:
            md = m.get("metadata", {})
            if not all(
                md.get(k) == v.get("$eq", v) for k, v in filter_metadata.items()
            ):
                continue
        out.append(
            {
                "user_id": int(m["id"]),
                "similarity": m["score"],
                "metadata": m.get("metadata", {}),
            }
        )
    return out[:n_similar]


def get_collaborative_recommendations(
    user_id: int,
    purchase_df: pd.DataFrame,
    n_recommendations: int = 5,
    filter_metadata: Optional[Dict[str, Any]] = None,
) -> List[int]:
    similars = get_similar_users(user_id, n_recommendations + 5, filter_metadata)
    if not similars:
        return []

    user_bought = set(
        purchase_df[purchase_df.user_id == user_id].product_id.astype(str)
    )
    sim_map = {u["user_id"]: u["similarity"] for u in similars}

    scores: Dict[str, float] = {}
    for _, row in purchase_df[purchase_df.user_id.isin(sim_map.keys())].iterrows():
        pid = str(row.product_id)
        if pid in user_bought:
            continue
        scores[pid] = scores.get(pid, 0.0) + sim_map[row.user_id]  # >>> changed

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [int(pid) for pid, _ in ranked[:n_recommendations]]


# --------------------------------------------------------------------------- #
#  Demo / debugging prints
# --------------------------------------------------------------------------- #
def print_similars(
    base_id: int,
    similars: List[Dict[str, Any]],
    user_df: pd.DataFrame,
    purchase_df: pd.DataFrame,
) -> None:
    logging.info(
        "\nTop-%d similar to %s:", len(similars), pretty_user(base_id, user_df)
    )
    for m in similars:
        logging.info(
            "  • %s  sim=%.3f", pretty_user(m["user_id"], user_df), m["similarity"]
        )
        logging.info(
            "    doc: %s", build_user_vector_text(m["user_id"], user_df, purchase_df)
        )


def example_usage() -> None:
    users, purchases = load_data()

    # build_user_vectors(users, purchases)  # run once, then comment out

    target = users.user_id.iloc[0]

    logging.info(
        "\n>>> Document for user %s:\n    %s\n",
        target,
        build_user_vector_text(target, users, purchases),
    )

    sims = get_similar_users(target, 3)
    print_similars(target, sims, users, purchases)

    recs = get_collaborative_recommendations(target, purchases, 10)
    logging.info("\nRecommended product IDs for user %s:\n  %s\n", target, recs)

    gender = users[users.user_id == target].gender.iloc[0]
    g_filter = {"gender": {"$eq": gender}}
    g_recs = get_collaborative_recommendations(target, purchases, 5, g_filter)
    logging.info("Gender-filtered recommendations (gender=%s):\n  %s", gender, g_recs)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    example_usage()
