#!/bin/bash
set -e

echo "Setting up LocalStack S3 + Lambda..."

LOCALSTACK_CONTAINER="localstack-main"  # Ensure this matches your container name

# Wait for LocalStack to be ready using the correct health endpoint
until curl -s http://localhost:4566/_localstack/health | grep '"s3": *"running"' > /dev/null; do
  echo "Waiting for LocalStack S3 service to be ready..."
  sleep 2
done

# Create bucket (if it doesn't already exist)
docker exec -i $LOCALSTACK_CONTAINER awslocal s3 mb s3://recommend-uploads || true

# See all buckets
docker exec -i $LOCALSTACK_CONTAINER awslocal s3 ls

# Send all data to the bucket from local directory
docker exec -i $LOCALSTACK_CONTAINER awslocal s3 sync ../data/new_dataset s3://recommend-uploads/

echo "✅ LocalStack setup completed."
