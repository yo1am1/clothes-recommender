"""This script initializes a Pinecone index for storing image embeddings and metadata."""

import pathlib
import os
from PIL import Image
import pandas as pd
import pinecone
from tqdm import tqdm

# import argparse
import torch
import numpy as np

from transformers.models.clip import CLIPProcessor, CLIPModel

DATA_DIR = pathlib.Path(__file__).parent.parent.parent / "data"
INDEX_NAME = "clip-test"

model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
processor = CLIPProcessor.from_pretrained(
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", use_fast=True
)


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


def upsert_embeddings(
    index,
    vectors,
    namespace: str = "image-search",
    batch_size: int = 1000,
):
    """
    Upsert vectors to Pinecone index.

    Args:
        index: Pinecone index object
        vectors: List of vectors to upsert
        namespace: Namespace in Pinecone
        batch_size: Number of vectors to upload at once
    """
    index.upsert(
        vectors=vectors,
        namespace=namespace,
        show_progress=True,
        batch_size=batch_size,
    )


def add_images_with_metadata(csv_path, images_folder, namespace=""):
    """
    Reads a CSV file containing product information along with a valid image path,
    computes the CLIP embedding for each image, and upserts it to the Pinecone index along with its metadata.
    """
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path, low_memory=False)
    df = df[(df["image_path"].notna()) & (df["image_downloaded"] == True)].copy()

    vectors = []
    for row in tqdm(df.itertuples(), total=len(df), desc="Adding images with metadata"):
        image_file = getattr(row, "image_path", None)
        if not image_file:
            continue

        full_image_path = (
            image_file
            if os.path.isabs(image_file)
            else os.path.join(images_folder, image_file)
        )
        if not os.path.exists(full_image_path):
            print(f"Image not found: {full_image_path}")
            continue

        try:
            embedding = get_image_embedding(
                image=Image.open(pathlib.Path(full_image_path)),
                processor=processor,
                model=model,
            )
        except Exception as e:
            print(f"Error computing embedding for {full_image_path}: {e}")
            continue

        metadata = {
            "product_name": getattr(row, "product_name", ""),
            "category": getattr(row, "category", ""),
            "product_url": getattr(row, "product_url", ""),
            "ratings": getattr(row, "ratings", 0),
            "reviews": getattr(row, "reviews", 0),
            "price": getattr(row, "price", None),
            "brand": getattr(row, "brand", ""),
            "color": getattr(row, "color", ""),
        }

        # Create a unique id for this image
        img_id = str(getattr(row, "product_id", row.Index))
        vector = {
            "id": img_id,
            "values": embedding,
            "metadata": metadata,
        }
        vectors.append(vector)

    if vectors:
        upsert_embeddings(index, vectors, namespace)
        print(f"Upserted {len(vectors)} image embeddings to Pinecone.")
    else:
        print("No vectors to upsert.")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Image Embeddings Setup Script using Pinecone"
    # )
    # parser.add_argument(
    #     "-f",
    #     "--force",
    #     action="store_true",
    #     help="Force deletion of the existing index before upserting new vectors",
    # )
    # args = parser.parse_args()

    pc = pinecone.Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
    )

    try:
        index = pc.Index(INDEX_NAME)
    except Exception as e:
        print(f"Error connecting to Pinecone index: {e}")

    namespace = "image-search"
    # if args.force:
    #     print(
    #         "Force deletion is not implemented in this setup script. Please clear vectors manually if needed."
    #     )

    csv_path = str(DATA_DIR / "new_dataset" / "foo-amazon-products-merged-updated.csv")
    images_folder = str(DATA_DIR / "images" / "new_raw")
    add_images_with_metadata(csv_path, images_folder, namespace)
