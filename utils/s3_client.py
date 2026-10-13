"""
S3 client utilities for the recommendation system.
Supports both AWS S3 and LocalStack S3.
"""

import logging
import boto3
from botocore.exceptions import ClientError
from utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_s3_client():
    """
    Create and return an S3 client configured to use either AWS S3 or LocalStack S3.
    The client detects whether to use LocalStack based on environment variables.

    Returns:
        Configured S3 client
    """
    s3_client = boto3.client(
        "s3",
        region_name=Config.AWS_REGION,
        aws_access_key_id=Config.AWS_ACCESS_KEY,
        aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
    )

    return s3_client


def list_buckets() -> list:
    """List all S3 buckets and return their names."""
    s3_client = get_s3_client()
    try:
        response = s3_client.list_buckets()
        bucket_names = [bucket["Name"] for bucket in response.get("Buckets", [])]
        logger.info(f"Found {len(bucket_names)} buckets: {', '.join(bucket_names)}")
        return bucket_names
    except ClientError as e:
        logger.error(f"Error listing buckets: {e}")
        return []


<<<<<<< HEAD
def create_bucket(bucket_name) -> bool:
    """Create an S3 bucket with the specified name.

    Args:
        bucket_name (str): The name of the bucket to create.
    Returns:
        bool: True if the bucket was created successfully, False otherwise.
    """
=======
def create_bucket(bucket_name):
    """Create a new S3 bucket."""
>>>>>>> remotes/origin/main
    s3_client = get_s3_client()
    try:
        s3_client.create_bucket(Bucket=bucket_name)
        logger.info(f"Bucket '{bucket_name}' created successfully.")
        return True
    except ClientError as e:
        logger.error(f"Error creating bucket '{bucket_name}': {e}")
        return False


<<<<<<< HEAD
def delete_bucket(bucket_name) -> bool:
    """Delete an S3 bucket.

    Args:
        bucket_name (str): The name of the bucket to delete.
    Returns:
        bool: True if the bucket was deleted successfully, False otherwise.
    """
=======
def delete_bucket(bucket_name):
    """Delete an S3 bucket."""
>>>>>>> remotes/origin/main
    s3_client = get_s3_client()
    try:
        s3_client.delete_bucket(Bucket=bucket_name)
        logger.info(f"Bucket '{bucket_name}' deleted successfully.")
        return True
    except ClientError as e:
        logger.error(f"Error deleting bucket '{bucket_name}': {e}")
        return False


<<<<<<< HEAD
def list_objects(bucket_name, prefix="") -> list:
    """List objects in a bucket with optional prefix.

    Args:
        bucket_name (str): The name of the bucket.
        prefix (str): Optional prefix to filter objects.
    Returns:
        list: List of object keys in the bucket.
    """
    s3_client = get_s3_client()
    objects = []
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name, Prefix=prefix, MaxKeys=30000
        )
        objects = [obj["Key"] for obj in response.get("Contents", [])]

        while response.get("NextContinuationToken", False):
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix,
                MaxKeys=40000,
                ContinuationToken=response["NextContinuationToken"],
            )
            objects.extend([obj["Key"] for obj in response.get("Contents", [])])

=======
def list_objects(bucket_name, prefix=""):
    """List objects in a bucket with optional prefix."""
    s3_client = get_s3_client()
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        objects = [obj["Key"] for obj in response.get("Contents", [])]
>>>>>>> remotes/origin/main
        logger.info(
            f"Found {len(objects)} objects in bucket '{bucket_name}' with prefix '{prefix}'"
        )
        return objects
    except ClientError as e:
        logger.error(f"Error listing objects in bucket '{bucket_name}': {e}")
        return []


<<<<<<< HEAD
def create_folder(bucket_name, folder_name) -> bool:
    """Create a folder in an S3 bucket.

    Args:
        bucket_name (str): The name of the bucket.
        folder_name (str): The name of the folder to create.
    Returns:
        bool: True if the folder was created successfully, False otherwise.
    """
=======
def create_folder(bucket_name, folder_name):
    """Create a folder in an S3 bucket."""
>>>>>>> remotes/origin/main
    s3_client = get_s3_client()
    try:
        s3_client.put_object(Bucket=bucket_name, Key=folder_name + "/")
        logger.info(f"Folder '{folder_name}' created in bucket '{bucket_name}'.")
        return True
    except ClientError as e:
        logger.error(
            f"Error creating folder '{folder_name}' in bucket '{bucket_name}': {e}"
        )
        return False


<<<<<<< HEAD
def upload_file(file_path, bucket_name, object_key) -> bool:
    """Upload a file to an S3 bucket.

    Args:
        file_path (str): The path to the file to upload.
        bucket_name (str): The name of the bucket.
        object_key (str): The key under which to store the file.
    Returns:
        bool: True if the file was uploaded successfully, False otherwise.
    """
=======
def upload_file(file_path, bucket_name, object_key):
    """Upload a file to an S3 bucket."""

>>>>>>> remotes/origin/main
    s3_client = get_s3_client()
    try:
        s3_client.upload_file(file_path, bucket_name, object_key)
        logger.info(f"Successfully uploaded {file_path} to {bucket_name}")
        return True
    except ClientError as e:
        logger.error(f"Error uploading file {file_path} to {bucket_name}: {e}")
        return False


<<<<<<< HEAD
def remove_file(bucket_name, object_key) -> bool:
    """Remove a file from an S3 bucket.

    Args:
        bucket_name (str): The name of the bucket.
        object_key (str): The key of the object to remove.
    Returns:
        bool: True if the file was removed successfully, False otherwise.
    """
=======
def remove_file(bucket_name, object_key):
    """Remove a file from an S3 bucket."""
>>>>>>> remotes/origin/main
    s3_client = get_s3_client()
    try:
        s3_client.delete_object(Bucket=bucket_name, Key=object_key)
        logger.info(f"Successfully removed {bucket_name}/{object_key}")
        return True
    except ClientError as e:
        logger.error(f"Error removing file {bucket_name}/{object_key}: {e}")
        return False


<<<<<<< HEAD
def download_file(bucket_name, object_key, file_path) -> bool:
    """Download a file from an S3 bucket.

    Args:
        bucket_name (str): The name of the bucket.
        object_key (str): The key of the object to download.
        file_path (str): The path where the file will be saved.
    Returns:
        bool: True if the file was downloaded successfully, False otherwise.
    """
=======
def download_file(bucket_name, object_key, file_path):
    """Download a file from an S3 bucket."""
>>>>>>> remotes/origin/main
    s3_client = get_s3_client()
    try:
        s3_client.download_file(bucket_name, object_key, file_path)
        logger.info(
            f"Successfully downloaded {bucket_name}/{object_key} to {file_path}"
        )
        return True
    except ClientError as e:
        logger.error(
            f"Error downloading file {bucket_name}/{object_key} to {file_path}: {e}"
        )
        return False


def get_object_bytes(bucket_name, object_key) -> bytes | None:
    """Get an object's bytes directly from S3 without saving to disk.

    Args:
        bucket_name (str): The name of the bucket.
        object_key (str): The key of the object to retrieve.
    Returns:
        bytes | None: The bytes of the object if successful, None otherwise.
    """
    s3_client = get_s3_client()
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        file_bytes = response["Body"].read()
        logger.info(f"Successfully retrieved {object_key} from {bucket_name}")
        return file_bytes
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            logger.warning(f"Object {object_key} not found in bucket {bucket_name}")
        else:
            logger.error(
                f"Error getting object {object_key} from bucket {bucket_name}: {e}"
            )
        return None


def object_exists(bucket_name, object_key) -> bool:
    """Check if an object exists in the bucket.

    Args:
        bucket_name (str): The name of the bucket.
        object_key (str): The key of the object to check.
    Returns:
        bool: True if the object exists, False otherwise.
    """
    s3_client = get_s3_client()
    try:
        s3_client.head_object(Bucket=bucket_name, Key=object_key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            logger.debug(f"Object {object_key} not found in bucket {bucket_name}")
        else:
            logger.error(
                f"Error checking object {object_key} in bucket {bucket_name}: {e}"
            )
        return False


if __name__ == "__main__":
    raise NotImplementedError(
        "This script is not intended to be run directly. Use the functions in your application."
    )
