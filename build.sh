#!/bin/bash
set -e

IMAGE_TAG="asia-south1-docker.pkg.dev/ecom-review-app/jobs/monthly-report/main/v1"

echo "Building Docker image: ${IMAGE_TAG}"
docker build -t "${IMAGE_TAG}" -f Dockerfile .

echo "Pushing Docker image: ${IMAGE_TAG}"
docker push "${IMAGE_TAG}"

echo "Done! Image pushed: ${IMAGE_TAG}"
