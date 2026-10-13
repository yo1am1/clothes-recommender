"""This script downloads images from URLs in a CSV file, removes duplicates, and updates the dataset.
It includes options for resuming downloads and validating existing images.
It uses asynchronous requests for efficient downloading and handles duplicate images by checking their MD5 hashes.
It also provides command-line arguments for flexibility in usage.
"""

__author__ = "Yehor Trepalin"

import argparse
import asyncio
import aiohttp
import os
import pathlib
import pandas as pd
import hashlib

# --- Configuration ---
IMAGE_DIR = pathlib.Path(__file__).parent.parent / "data" / "images" / "new_raw"
DATASET_DIR = pathlib.Path(__file__).parent.parent / "data" / "new_dataset"
DATASET_CSV = os.path.join(DATASET_DIR, "foo-amazon-products-merged-updated.csv")
UPDATED_CSV = os.path.join(DATASET_DIR, "foo-second-amazon-products-merged-updated.csv")

os.makedirs(IMAGE_DIR, exist_ok=True)

df = pd.read_csv(DATASET_CSV)
if "image_path" not in df.columns:
    df["image_path"] = ""
if "image_downloaded" not in df.columns:
    df["image_downloaded"] = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download images and update dataset with duplicate removal, resume, and validation options."
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Only remove duplicates and clear missing paths in dataset (no new downloads).",
    )
    parser.add_argument(
        "--continue",
        dest="resume",
        action="store_true",
        help="Resume loading from the last image that was downloaded (skip already downloaded images).",
    )

    return parser.parse_args()


def clear_image_directory(folder: str) -> None:
    """Clear the image directory by removing all files in it.

    Args:
        folder (str): Path to the directory to clear.
    """
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Cleared images in {folder}")
    else:
        print(f"Folder {folder} does not exist.")


async def download_image(session, url, save_path) -> bool:
    """Download an image from a URL and save it to a specified path.

    Args:
        session: aiohttp session for making requests.
        url (str): URL of the image to download.
        save_path (str): Path to save the downloaded image.

    Returns:
        bool: True if the download was successful, False otherwise.
    """
    try:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                with open(save_path, "wb") as f:
                    f.write(content)
                return True
            else:
                print(f"Failed to download {url}. HTTP Status: {response.status}")
                return False
    except Exception as e:
        print(f"Exception while downloading {url}: {e}")
        return False


async def download_all_images(dataframe, resume=False) -> pd.DataFrame:
    """Download all images from the URLs in the dataframe.

    Args:
        dataframe (pd.DataFrame): DataFrame containing image URLs and metadata.
        resume (bool): If True, skip already downloaded images.

    Returns:
        pd.DataFrame: Updated DataFrame with image paths and download status.
    """
    async with aiohttp.ClientSession() as session:
        tasks = []
        for idx, row in dataframe.iterrows():
            # Only process specific categories.
            if row["category"].lower().strip() in [
                "jeans",
                "shoes",
                "t-shirts & polos",
                "western wear",
                "lingerie & nightwear",
            ]:
                # If resuming, skip rows that already have a downloaded image.
                if (
                    resume
                    and row["image_downloaded"]
                    and row["image_path"]
                    and os.path.exists(row["image_path"])
                ):
                    continue

                url = row["image_url"]
                category = row["category"].replace(
                    " ", "_"
                )  # Replace spaces for filename
                filename = f"{category}_{idx}.jpeg"
                file_path = os.path.join(IMAGE_DIR, filename)

                task = asyncio.create_task(download_image(session, url, file_path))
                tasks.append((task, idx, file_path))

        # Wait for each download task and update the dataframe accordingly.
        for task, idx, file_path in tasks:
            success = await task
            if success:
                dataframe.at[idx, "image_path"] = file_path
                dataframe.at[idx, "image_downloaded"] = True

    return dataframe


def remove_duplicate_images(folder) -> list:
    """Remove duplicate images from a specified folder based on their MD5 hash.

    Args:
        folder (str): Path to the folder containing images.

    Returns:
        list: List of removed duplicate image file paths.
    """
    seen_hashes = {}
    removed_files = []

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        if os.path.isfile(file_path):
            with open(file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            if file_hash in seen_hashes:
                print(f"Removing duplicate image: {file_path}")
                os.remove(file_path)
                removed_files.append(file_path)
            else:
                seen_hashes[file_hash] = file_path
    return removed_files


def update_dataframe_after_duplicate_removal(dataframe, removed_files) -> pd.DataFrame:
    """Update the dataframe to remove rows that reference images that were removed.

    Args:
        dataframe (pd.DataFrame): DataFrame containing image paths and metadata.
        removed_files (list): List of removed duplicate image file paths.

    Returns:
        pd.DataFrame: Updated DataFrame with rows referencing removed images removed.
    """
    indices_to_remove = []
    for idx, row in dataframe.iterrows():
        if row["image_downloaded"] and row["image_path"] in removed_files:
            indices_to_remove.append(idx)
    if indices_to_remove:
        print(
            f"Removing {len(indices_to_remove)} rows from the dataframe due to duplicate images."
        )
        dataframe.drop(indices_to_remove, inplace=True)
    return dataframe


# --- Main Routine ---
async def main(args) -> None:
    """Main function to handle the image downloading process.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """

    global df

    if args.validate:
        print("Running in validate mode: removing duplicates and updating dataset.")
        removed_files = remove_duplicate_images(IMAGE_DIR)
        df = update_dataframe_after_duplicate_removal(df, removed_files)
        df.to_csv(UPDATED_CSV, index=False)
        print(f"Dataframe updated after duplicate removal and saved to {UPDATED_CSV}.")
        return

    # If not resuming, clear the destination directory.
    if not args.resume:
        clear_image_directory(str(IMAGE_DIR))
    else:
        print("Resuming image downloads. Existing images will be preserved.")

    print("Starting image downloads...")
    df = await download_all_images(df, resume=args.resume)
    df.to_csv(UPDATED_CSV, index=False)
    print(f"Updated dataset saved to {UPDATED_CSV}.")

    print("Removing duplicate images from folder...")
    removed_files = remove_duplicate_images(IMAGE_DIR)
    print("Duplicate removal complete.")

    # Update dataframe to remove rows for images that were deleted.
    df = update_dataframe_after_duplicate_removal(df, removed_files)
    df.to_csv(UPDATED_CSV, index=False)
    print(f"Dataframe updated after duplicate removal and saved to {UPDATED_CSV}.")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
