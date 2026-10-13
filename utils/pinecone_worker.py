"""This module contains functions to interact with Pinecone for vector similarity search.
It provides functionality for image recommendations using a Pinecone index,
# including text search, image search, and user-based recommendations.
"""

import os
import boto3
import io
import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
from typing import Dict, Optional
from utils.config import Config
from collections import Counter


def get_image_by_id_from_s3(image_id, category):
    s3_client = boto3.client(
        "s3",
        region_name=Config.AWS_REGION,
        aws_access_key_id=Config.AWS_ACCESS_KEY,
        aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
    )
    bucket_name = Config.AWS_BUCKET_NAME
    s3_prefix = "recommendation-engine-data-demo"
    s3_key = f"{s3_prefix}/{category.replace(' ', '_')}_{image_id}.jpeg"

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        image_bytes = response["Body"].read()
        img = Image.open(io.BytesIO(image_bytes))
        print(f"Successfully retrieved image {s3_key} from S3")
    except Exception as e:
        print(f"Error retrieving image {s3_key} from S3: {e}")
        return None

    return img


def get_image_embedding(image, processor, model) -> np.ndarray:
    """
    Process an image and get its embedding vector using CLIP.

    Args:
        image: PIL.Image image object

    Returns:
        numpy.ndarray: Image embedding vector
    """
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_embeddings = model.get_image_features(**inputs)

    return image_embeddings.cpu().numpy()[0].tolist()


def get_text_embedding(text: str, processor, model) -> np.ndarray:
    """
    Process text and get its embedding vector using CLIP.

    Args:
        text: Input text to embed

    Returns:
        numpy.ndarray: Text embedding vector
    """
    inputs = processor(text=[text], return_tensors="pt")
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)

    return text_embedding.cpu().numpy()[0].tolist()


def list_indexes(pc):
    """List all available indexes in the Pinecone account."""
    return pc.list_indexes()


def has_index(index_name: str, pc) -> bool:
    """Check if an index exists."""
    try:
        indexes = list_indexes(pc)
        return index_name in [val.get("name", None) for val in indexes]
    except Exception as e:
        print(f"Error checking index: {e}")
        return False


def add_vectors_from_dataframe(
    df: pd.DataFrame,
    processor,
    model,
    pc,
    image_dir: str,
    index_name: str = "clip-test",
    namespace: str = "image-search",
    batch_size: int = 1000,
) -> int:
    """
    Process images from a DataFrame and add their vectors to Pinecone.

    Args:
        df: DataFrame with image information
        index_name: Name of the Pinecone index
        image_dir: Directory containing images
        namespace: Namespace in Pinecone
        batch_size: Number of vectors to upload at once

    Returns:
        int: Number of vectors successfully added
    """
    # Load model if not already loaded

    # Check and prepare DataFrame
    required_columns = ["image_path"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Filter dataframe for valid image paths
    df = df[(df["image_path"].notna()) & (df["image_downloaded"] == True)].copy()

    # Get index
    index = pc.Index(index_name)

    vectors = []
    success_count = 0

    for row in tqdm(df.itertuples(), total=len(df), desc="Adding images with metadata"):
        image_file = getattr(row, "image_path", None)
        if not image_file:
            continue

        full_image_path = (
            image_file
            if os.path.isabs(image_file)
            else os.path.join(image_dir, image_file)
        )
        if not os.path.exists(full_image_path):
            print(f"Image not found: {full_image_path}")
            continue

        try:
            embedding = get_image_embedding(full_image_path, processor, model)
        except Exception as e:
            print(f"Error computing embedding for {full_image_path}: {e}")
            continue

        img_id = str(getattr(row, "product_id", getattr(row, "Index", None)))

        # Prepare metadata from row attributes
        metadata = {}
        for field in [
            "category",
            "product_url",
            "ratings",
            "reviews",
            "price",
            "brand",
            "color",
        ]:
            if hasattr(row, field):
                metadata[field] = getattr(row, field)

        vector = {"id": img_id, "values": embedding.tolist(), "metadata": metadata}

        vectors.append(vector)
        success_count += 1

    # Upsert any remaining vectors
    if vectors:
        index.upsert(
            vectors=vectors,
            namespace=namespace,
            show_progress=True,
            batch_size=batch_size,
        )

    # Get stats after insertion
    stats = index.describe_index_stats()
    print(f"Index stats after insertion: {stats}")

    return success_count


def text_search(
    query_text: str,
    processor,
    model,
    pc,
    namespace: str = "image-search",
    index_name: str = "clip-test",
    top_k: int = 5,
    filter_dict: Optional[Dict] = None,
    retrieve_from_s3: bool = False,
):
    """
    Search for similar vectors using a text query.

    Args:
        query_text: Text to search for
        namespace: Namespace to search in
        top_k: Number of results to return
        filter_dict: Dictionary of metadata filters

    Returns:
        Dict: Search results from Pinecone
    """
    # Get text embedding
    text_embedding = get_text_embedding(query_text, processor, model)

    # Get index
    index = pc.Index(index_name)

    # Query the index
    response = index.query(
        vector=text_embedding,
        top_k=top_k,
        namespace=namespace,
        include_values=False,
        include_metadata=True,
        filter=filter_dict,
    )

    if not retrieve_from_s3:
        return response

    # If retrieve_from_s3 is True, process the response to get S3 URLs
    results = []
    for match in response["matches"]:
        image = get_image_by_id_from_s3(match["id"], match["metadata"]["category"])
        if image:
            print(f"Image {match['id']} retrieved successfully.")
            results.append(
                {
                    "id": match["id"],
                    "image": image,
                    "metadata": match["metadata"],
                    "score": match["score"],
                }
            )
    return results


def image_search(
    image,
    processor,
    model,
    pc,
    index_name: str = "clip-test",
    namespace: str = "image-search",
    top_k: int = 5,
    filter_dict: Optional[Dict] = None,
    retrieve_from_s3: bool = False,
):
    """
    Search for similar vectors using an image.

    Args:
        image: PIL.Image image object
        processor: The image processor
        model: The CLIP model for generating embeddings
        pc: The Pinecone client
        index_name: Name of the index to query
        namespace: Namespace to search in
        top_k: Number of results to return
        filter_dict: Dictionary of metadata filters
        retrieve_from_s3: Flag to determine if S3 retrieval is needed

    Returns:
        Dict: Search results from Pinecone, or modified S3 results if retrieve_from_s3 is True
    """
    # Get image embedding
    image_embedding = get_image_embedding(image, processor, model)

    # Get index
    index = pc.Index(index_name)

    # Query the index
    response = index.query(
        vector=image_embedding,
        top_k=top_k,
        namespace=namespace,
        include_values=True,
        include_metadata=True,
        filter=filter_dict,
    )

    if not retrieve_from_s3:
        return response

    # If retrieve_from_s3 is True, process the response to get S3 URLs
    results = []
    for match in response["matches"]:
        image_from_s3 = get_image_by_id_from_s3(
            match["id"], match["metadata"]["category"]
        )
        if image_from_s3:
            print(f"Image {match['id']} retrieved successfully.")
            results.append(
                {
                    "id": match["id"],
                    "image": image_from_s3,
                    "metadata": match["metadata"],
                    "score": match["score"],
                }
            )

    return results


def recommend_for_user(
    pc,
    user_id: str,
    user_data: pd.DataFrame,
    purchase_data: pd.DataFrame,
    processor,
    model,
    index_name: str = "clip-test",
    namespace: str = "image-search",
    top_n: int = 6,
) -> tuple[list, list, Optional[dict]]:
    """
    Optimized recommendations using purchase history with Pinecone.
    Builds a metadata filter from top brands and product categories,
    queries for each top category and merges the results.
    Ensures minimal memory usage by limiting intermediate list sizes.

    Parameters:
      pc: Pinecone client.
      user_id: User ID as string.
      user_data: DataFrame of user profiles.
      purchase_data: DataFrame of purchase history.
      processor: CLIP processor.
      model: CLIP model.
      index_name: Pinecone index name.
      namespace: Pinecone namespace.
      top_n: Number of recommendations to return.

    Returns:
      Tuple of (recommended_ids, recommended_metas, metadata_filter).
    """
    # Retrieve user profile; fallback if none.
    row = user_data[user_data["user_id"] == int(user_id)]
    if row.empty:
        print(f"No user profile found for user_id={user_id}. Using default query.")
        default_result = text_search(
            query_text="jeans",
            processor=processor,
            model=model,
            pc=pc,
            namespace=namespace,
            index_name=index_name,
            top_k=20,
        )
        rec_tuples = []
        for match in default_result["matches"]:
            candidate_id = match.get("id")
            metadata = match.get("metadata", {})
            candidate_category = metadata.get("category") or metadata.get(
                "product_category", "unknown"
            )
            rec_tuples.append((candidate_id, candidate_category))
        return rec_tuples[:top_n], [], None

    # Get user purchase history and build metadata filter.
    user_purchases = purchase_data[purchase_data["user_id"] == int(user_id)]
    if user_purchases.empty:
        metadata_filter = None
        top_categories = []
    else:
        brands = (
            user_purchases["brand"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .tolist()
        )
        categories = (
            user_purchases["product_category"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .tolist()
        )
        top_brands = [b for b, _ in Counter(brands).most_common(3)] if brands else []
        top_categories = (
            [c for c, _ in Counter(categories).most_common(3)] if categories else []
        )
        if top_brands and top_categories:
            metadata_filter = {
                "$or": [
                    {"brand": {"$in": top_brands}},
                    {"product_category": {"$in": top_categories}},
                ]
            }
        elif top_brands:
            metadata_filter = {"brand": {"$in": top_brands}}
        elif top_categories:
            metadata_filter = {"product_category": {"$in": top_categories}}
        else:
            metadata_filter = None

    candidate_dict = {}  # using dict to avoid duplicate IDs
    if metadata_filter and top_categories:
        # Query Pinecone for each top category. We use top_k=2 to limit memory footprint.
        for cat in top_categories:
            result = text_search(
                query_text=cat,
                processor=processor,
                model=model,
                pc=pc,
                namespace=namespace,
                index_name=index_name,
                top_k=2,
                filter_dict=metadata_filter,
            )
            for match in result.get("matches", []):
                candidate_id = match.get("id")
                metadata = match.get("metadata", {})
                candidate_category = metadata.get("category")

                candidate_key = (candidate_id, candidate_category)
                candidate_dict[candidate_key] = metadata
    else:
        # Fallback query.
        result = text_search(
            query_text="jeans",
            processor=processor,
            model=model,
            pc=pc,
            namespace=namespace,
            index_name=index_name,
            top_k=5,
        )
        for match in result.get("matches", []):
            candidate_id = match.get("id")
            metadata = match.get("metadata", {})
            candidate_category = metadata.get("category")
            candidate_key = (candidate_id, candidate_category)
            candidate_dict[candidate_key] = metadata

    # Remove items already purchased.
    purchased_ids = set(user_purchases["product_id"].astype(str).tolist())
    rec_tuples = []
    rec_metas = []
    for (cand_id, cand_category), meta in candidate_dict.items():
        if cand_id in purchased_ids:
            continue
        rec_tuples.append((cand_id, cand_category))
        rec_metas.append(meta)
        if len(rec_tuples) == top_n:
            break

    if not rec_tuples:
        print("No new items found; user may already own top selections.")

    return rec_tuples, rec_metas, metadata_filter


if __name__ == "__main__":
    raise NotImplementedError(
        "This module is not intended to be run as a standalone script. "
        "Please use the functions provided in your application."
    )
