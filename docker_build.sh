#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Base algorithm name
algorithm_name="demo-sagemaker-multimodel"

# The names of the different stages
preprocessing_name="${algorithm_name}-preprocessing"
training_name="${algorithm_name}-training"
model_registry_name="${algorithm_name}-model-registry"

# AWS account ID and region
account=$(aws sts get-caller-identity --query Account --output text)
region=$(aws configure get region)
region=${region:-us-east-1}

# Ensure repositories exist in ECR
create_repository_if_not_exists() {
    local repository_name=$1
    aws ecr describe-repositories --repository-names "${repository_name}" --region ${region} > /dev/null 2>&1 || \
    aws ecr create-repository --repository-name "${repository_name}" --region ${region} > /dev/null
}

# Create repositories if they don't exist
create_repository_if_not_exists ${preprocessing_name}
create_repository_if_not_exists ${training_name}
create_repository_if_not_exists ${model_registry_name}

# Create and use buildx builder
docker buildx create --use || true  # Create a buildx builder if it doesn't already exist

# Authenticate Docker to ECR
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com

# Function to build and push an image to ECR
build_and_push_image() {
    local algorithm_name=$1
    local dockerfile=$2
    local tag=$3

    fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:${tag}"

    # Build and push the Docker image using buildx for cross-compilation
    docker buildx build --platform linux/amd64 -t ${fullname} -f ${dockerfile} --push .
}

# Build and push preprocessing image
build_and_push_image ${preprocessing_name} Dockerfile.preprocessing latest

# Build and push training image
build_and_push_image ${training_name} Dockerfile.training latest

# Build and push model registry image
build_and_push_image ${model_registry_name} Dockerfile.model_registry latest

echo "All images have been successfully built and pushed to ECR."
