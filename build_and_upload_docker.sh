#!/bin/bash

# Build and push docker image to AWS ECR.

set -eo pipefail

if [ -z "$1" ]; then
  echo "Must supply AWS account ID"
  exit 1;
fi

if [ -z "$2" ]; then
  echo "Must supply the image tag"
  exit 1;
fi

IMAGE_TAG=$2
ACCOUNT=$1
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.us-west-2.amazonaws.com
DOCKER_BUILDKIT=1 docker build -t $ACCOUNT.dkr.ecr.us-west-2.amazonaws.com/lightllm:$IMAGE_TAG .
docker push $ACCOUNT.dkr.ecr.us-west-2.amazonaws.com/lightllm:$IMAGE_TAG
