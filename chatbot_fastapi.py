"""This is a FastAPI application for a clothes recommender system.
It provides endpoints for recommending clothes to customers based on their purchase history and preferences.
It also includes endpoints for uploading CSV and JSON data, and for searching for products by image.
"""

import argparse
import io
import logging
import pathlib
import time
import os
from botocore.exceptions import ClientError
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import ORJSONResponse
from PIL import Image

import chromadb
from chromadb.utils.embedding_functions.open_clip_embedding_function import (
    OpenCLIPEmbeddingFunction,
)
from chromadb.utils.data_loaders import ImageLoader
from utils.vectorstore_worker import (
    format_metadata,
    recommend_for_user,
    collaborative_recommend_for_user,
    remove_duplicate_ids_with_metadata,
    search_by_image,
)
from utils.s3_client import get_s3_client, list_buckets, create_bucket, upload_file

# ---------------------------
# DATA DIRECTORIES AND FILE PATHS
# ---------------------------
DATA_DIR = pathlib.Path(__file__).parent / "data"
CUSTOMER_PROFILES_PATH = DATA_DIR / "foo_customers.csv"
PURCHASE_HISTORY_PATH = DATA_DIR / "foo_purchase_history.csv"
PRODUCT_DATA_PATH = DATA_DIR / "new_dataset" / "foo-amazon-products-merged-updated.csv"

# ---------------------------
# INITIALIZE EMBEDDING FUNCTION AND CHROMADB
# ---------------------------
embedding_function = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()

chroma_client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma"))
multimodal_collection = chromadb.PersistentClient(
    path=str(DATA_DIR / "chroma")
).get_or_create_collection(
    name="multimodal_images",
    embedding_function=embedding_function,
    data_loader=data_loader,
)


# ---------------------------
# HELPER FUNCTIONS TO LOAD CSVs
# ---------------------------
def __load_customer_profiles() -> pd.DataFrame:
    if not CUSTOMER_PROFILES_PATH.exists():
        logging.warning(f"Customer profiles file not found: {CUSTOMER_PROFILES_PATH}")
        return pd.DataFrame()
    return pd.read_csv(CUSTOMER_PROFILES_PATH)


def __load_purchase_history() -> pd.DataFrame:
    if not PURCHASE_HISTORY_PATH.exists():
        logging.warning(f"Purchase history file not found: {PURCHASE_HISTORY_PATH}")
        return pd.DataFrame()
    return pd.read_csv(PURCHASE_HISTORY_PATH)


def __load_product_data() -> pd.DataFrame:
    if not PRODUCT_DATA_PATH.exists():
        logging.warning(f"Product data file not found: {PRODUCT_DATA_PATH}")
        return pd.DataFrame()
    return pd.read_csv(PRODUCT_DATA_PATH)


# ---------------------------
# New function: Generate customer profiles from purchase history.
# ---------------------------
def generate_user_profiles(purchase_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a profile for each customer by analyzing their purchase history.
    Since we do not have colors or sizes now, we simply compute the top categories.
    """
    profiles = []
    from collections import Counter

    for user_id, group in purchase_df.groupby("user_id"):
        categories = (
            group["category"]
            .dropna()
            .astype(str)
            .apply(lambda x: x.strip().lower())
            .tolist()
        )
        top_categories = [cat for cat, count in Counter(categories).most_common(3)]
        preferences = f"Categories: {', '.join(top_categories)}"
        profiles.append(
            {
                "user_id": user_id,
                "preferences": preferences,
                "favorite_categories": ", ".join(top_categories),
            }
        )
    return pd.DataFrame(profiles)


# ---------------------------
# Lifespan: Load Data at Startup and Store in app.state.
# ---------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Loading customer profiles...")
    app.state.user_data = __load_customer_profiles()

    logging.info("Loading purchase history...")
    app.state.purchase_data = __load_purchase_history()

    logging.info("Loading product data...")
    app.state.product_data = __load_product_data()

    logging.info("Startup tasks completed.")
    yield

    logging.info("Shutting down...")
    app.state.user_data = None
    app.state.purchase_data = None
    app.state.product_data = None

    logging.info("Shutdown complete.")


# ---------------------------
# FASTAPI APP INITIALIZATION
# ---------------------------
app = FastAPI(
    title="Clothes Recommender API",
    version="0.0.1",
    description="Basic API for recommending clothes to customers.",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
    openapi_tags=[
        {
            "name": "Recommendations",
            "description": "Operations related to recommendations",
        },
        {
            "name": "Integrations",
            "description": "Operations related to data integration",
        },
        {"name": "Users", "description": "Operations related to customers"},
        {"name": "Settings", "description": "Operations related to settings"},
    ],
)


# ---------------------------
# API ENDPOINTS
# ---------------------------
@app.get("/api/v1/")
def read_root():
    return {
        "message": "Welcome to the Clothes Recommender API! Use /api/v1/recs/user/{user_id} to get recommendations.",
        "endpoints": {
            "/api/v1/recs/user/{user_id}": "Get recommendations for a customer"
        },
    }


@app.get("/api/v1/recs/hot", tags=["Recommendations"])
def best_sellers_endpoint(top_n: int = 5):
    try:
        product_df = app.state.product_data
    except Exception as e:
        logging.error(f"Failed to load product data: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to load product data."
        ) from e

    if "reviews" not in product_df.columns:
        raise HTTPException(
            status_code=500,
            detail="Product data does not contain a 'reviews' column for determining best-sellers.",
        )

    if "product_id" not in product_df.columns:
        product_df["product_id"] = product_df.index.astype(str)

    sorted_df = product_df.sort_values("reviews", ascending=False)
    best_sellers = sorted_df.head(top_n)
    best_seller_ids = best_sellers["product_id"].tolist()
    image_paths = [str(DATA_DIR / "images" / f"{pid}.jpeg") for pid in best_seller_ids]

    return {"best_seller_ids": best_seller_ids, "image_paths": image_paths}


@app.post("/api/v1/recs/image-search", tags=["Recommendations"])
async def image_search(
    file: UploadFile,
    top_k: int = 5,
):
    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        logging.error(f"Error reading image file: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file") from e

    result_ids, result_meta = search_by_image(
        image_array=np.array(image), top_k=top_k, collection=multimodal_collection
    )
    unique_ids, unique_metas = remove_duplicate_ids_with_metadata(
        result_ids, result_meta
    )
    return {"result_ids": unique_ids, "metadata": unique_metas}


@app.get("/api/v1/recs/user/{user_id}", tags=["Recommendations"])
def recommend_endpoint(
    user_id: int,
    top_n: int = 5,
    rec_type: Literal["content-based", "collaborative"] = "content-based",
):
    user_data = app.state.user_data
    purchase_data = app.state.purchase_data
    if user_data.empty:
        raise HTTPException(status_code=404, detail="No customer data loaded.")
    if purchase_data.empty:
        raise HTTPException(status_code=404, detail="No purchase history loaded.")

    if rec_type == "content-based":
        rec_ids, _, rec_filter = recommend_for_user(
            collection=multimodal_collection,
            user_id=user_id,
            user_data=user_data,
            purchase_data=purchase_data,
            top_n=top_n,
        )
        image_paths = [str(DATA_DIR / "images" / f"{pid}.jpeg") for pid in rec_ids]
        return {
            "user_id": user_id,
            "recommended_ids": rec_ids,
            "metadata_filter": rec_filter,
            "image_paths": image_paths,
            "rec_type": rec_type,
        }
    elif rec_type == "collaborative":
        rec_ids = collaborative_recommend_for_user(user_id, purchase_data, top_n)
        image_paths = [str(DATA_DIR / "images" / f"{pid}.jpeg") for pid in rec_ids]
        return {
            "user_id": user_id,
            "recommended_ids": rec_ids,
            "image_paths": image_paths,
            "rec_type": rec_type,
        }
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid rec_type. Must be 'content-based' or 'collaborative'.",
        )


@app.get("/api/v1/user/{user_id}/orders", tags=["Users"])
def user_orders(user_id: int):
    orders_df = app.state.purchase_data
    user_orders_df = orders_df[orders_df.get("user_id") == user_id]
    if user_orders_df.empty:
        raise HTTPException(status_code=404, detail="No orders found for this customer")
    orders = user_orders_df.to_dict(orient="records")
    return {"user_id": user_id, "orders": orders}


@app.get("/api/v1/user/{user_id}/purchase-history", tags=["Users"])
def user_purchases(user_id: int, period: int = 30, n: int = 5):
    """Get n-most recent purchases for a user by time period"""
    purchase_df = app.state.purchase_data
    if purchase_df.empty:
        raise HTTPException(status_code=404, detail="No purchase history available.")
    try:
        purchase_df["purchase_date"] = pd.to_datetime(purchase_df["purchase_date"])
    except Exception as e:
        logging.error(f"Error converting purchase_date to datetime: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to convert purchase_date to datetime."
        ) from e
    cutoff_date = datetime.now() - timedelta(days=period)
    recent_purchases = purchase_df[
        (purchase_df["user_id"] == user_id)
        & (purchase_df["purchase_date"] >= cutoff_date)
    ]
    recent_purchases = recent_purchases.sort_values(
        "purchase_date", ascending=False
    ).head(n)
    if recent_purchases.empty:
        raise HTTPException(
            status_code=404, detail="No recent purchases found for this customer."
        )
    purchases = recent_purchases.to_dict(orient="records")
    return {
        "user_id": user_id,
        "recent_purchases": purchases,
    }


# ---------------------------
# Modified integration endpoints to use S3 for document loading without temp files
# ---------------------------
@app.post("/api/v1/integrations/from-csv", tags=["Integrations"])
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file directly to S3 without using temporary files.
    """
    bucket_name = "recommend-uploads"
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"{timestamp}_{file.filename}"

    try:
        contents = file.file.read()
        temp_file = io.BytesIO()
        temp_file.write(contents)

        temp_file.seek(0)  # Reset the file pointer to the beginning

        # Ensure the bucket exists
        if bucket_name not in list_buckets():
            create_bucket(bucket_name)

        s3_client = get_s3_client()
        s3_client.upload_fileobj(temp_file, bucket_name, file.filename)
    except ClientError as e:
        logging.error(f"Error uploading CSV file to S3: {e}")
        raise HTTPException(
            status_code=500, detail="Error uploading CSV file to S3"
        ) from e
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise HTTPException(status_code=404, detail="File not found") from e
    except IOError as e:
        logging.error(f"IO error: {e}")
        raise HTTPException(status_code=500, detail="Error processing file") from e
    except Exception as e:
        logging.error(f"Error uploading CSV file to S3: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error uploading CSV file to S3: {str(e)}"
        ) from e
    finally:
        temp_file.close()

    return {
        "message": "CSV file uploaded to S3 successfully",
        "bucket": bucket_name,
        "object_key": file.filename,
    }


@app.post("/api/v1/integrations/from-json", tags=["Integrations"])
async def input_json(data: dict):
    """
    Convert JSON input to CSV in-memory and upload it directly to S3.
    """
    bucket_name = os.environ.get("S3_BUCKET", "recommend-uploads")
    try:
        if isinstance(data, dict):
            # Check if dictionary values are scalar and handle accordingly
            if all(not isinstance(v, (list, dict)) for v in data.values()):
                # Create a single-row DataFrame with scalar values
                df = pd.DataFrame([data])
            else:
                df = pd.DataFrame(data)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError("Unsupported JSON format")
    except Exception as e:
        logging.error(f"Error converting JSON to DataFrame: {e}")
        raise HTTPException(
            status_code=400, detail="Invalid JSON format for CSV conversion"
        ) from e
    filename = f"input_{int(time.time())}.csv"
    try:
        # Convert DataFrame to CSV string and then to a BytesIO buffer
        csv_string = df.to_csv(index=False)
        csv_bytes = csv_string.encode("utf-8")
        csv_buffer = io.BytesIO(csv_bytes)
        # Ensure the bucket exists
        if bucket_name not in list_buckets():
            create_bucket(bucket_name)
        s3 = get_s3_client()
        # Upload the CSV file in-memory directly to S3
        s3.upload_fileobj(csv_buffer, bucket_name, filename)
    except Exception as e:
        logging.error(f"Error uploading CSV file from JSON to S3: {e}")
        raise HTTPException(
            status_code=500, detail="Error uploading CSV file from JSON to S3"
        ) from e
    return {
        "message": "CSV file created from JSON input and uploaded to S3 successfully",
        "bucket": bucket_name,
        "object_key": filename,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run FastAPI server for Clothes Recommender API"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8502, help="Port number")
    parser.add_argument(
        "--reload", action="store_true", default=False, help="Enable auto-reloading"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes"
    )
    args = parser.parse_args()

    uvicorn.run(
        "chatbot_fastapi:app",  # adjust module name if needed
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )
