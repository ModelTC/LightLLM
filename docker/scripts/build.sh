#!/usr/bin/env bash
set -euo pipefail

# Notes:
# - All toggles can be configured via CLI flags or environment variables.
# - Default behavior matches the old build_default.sh: enable both DEEPEP and NIXL, and enable cache.
#
# Examples:
#   ./docker/scripts/build.sh
#   ./docker/scripts/build.sh --lite
#   ./docker/scripts/build.sh --no-deepep --no-cache
#   ./docker/scripts/build.sh --no-nixl
#   ./docker/scripts/build.sh --cuda-version 12.4.1 --image-prefix myrepo/lightllm
#   IMAGE_TAG=custom-cuda12 ./docker/scripts/build.sh
#
# Options:
#   --no-deepep               Disable DEEPEP (default: enabled)
#   --no-nixl                 Disable NIXL (default: enabled)
#   --no-cache                Disable cache (default: enabled)
#   --lite                    Disable DEEPEP, NIXL and cache in one shot
#   --cuda-version <ver>      CUDA version (default: 12.8.0)
#   --image-prefix <name>     Image prefix (default: lightllm)
#   --image-tag <tag>         Image tag (default: generated from enabled features)
#   -h / --help               Show help

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

IMAGE_PREFIX="${IMAGE_PREFIX:-lightllm}"
CUDA_VERSION="${CUDA_VERSION:-12.8.0}"
IMAGE_TAG="${IMAGE_TAG:-}"

ENABLE_DEEPEP="${ENABLE_DEEPEP:-1}"
ENABLE_NIXL="${ENABLE_NIXL:-1}"
ENABLE_CACHE="${ENABLE_CACHE:-1}"

print_help() {
  sed -n '1,80p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-deepep) ENABLE_DEEPEP=0 ;;
    --no-nixl) ENABLE_NIXL=0 ;;
    --no-cache) ENABLE_CACHE=0 ;;
    --lite)
      ENABLE_DEEPEP=0
      ENABLE_NIXL=0
      ENABLE_CACHE=0
      ;;
    --cuda-version)
      CUDA_VERSION="${2:-}"
      shift
      ;;
    --image-prefix)
      IMAGE_PREFIX="${2:-}"
      shift
      ;;
    --image-tag)
      IMAGE_TAG="${2:-}"
      shift
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      print_help >&2
      exit 1
      ;;
  esac
  shift
done

# Generate default image tag based on enabled features:
# - All on: cuda${CUDA_VERSION} (same as old build_default.sh)
# - Other combos: composed from enabled feature names
if [[ -z "${IMAGE_TAG}" ]]; then
  tag_parts=()
  if [[ "${ENABLE_NIXL}" -eq 1 ]]; then
    tag_parts+=("nixl")
  fi
  if [[ "${ENABLE_DEEPEP}" -eq 1 ]]; then
    tag_parts+=("deepep")
  fi
  if [[ "${ENABLE_NIXL}" -eq 1 && "${ENABLE_DEEPEP}" -eq 1 && "${ENABLE_CACHE}" -eq 1 ]]; then
    IMAGE_TAG="cuda${CUDA_VERSION}"
  else
    prefix=""
    if [[ ${#tag_parts[@]} -gt 0 ]]; then
      prefix="$(IFS='.'; echo "${tag_parts[*]}")-"
    fi
    IMAGE_TAG="${prefix}cuda${CUDA_VERSION}"
  fi
fi

DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
  --build-arg CUDA_VERSION="${CUDA_VERSION}" \
  --build-arg ENABLE_DEEPEP="${ENABLE_DEEPEP}" \
  --build-arg ENABLE_NIXL="${ENABLE_NIXL}" \
  --build-arg ENABLE_CACHE="${ENABLE_CACHE}" \
  -t "${IMAGE_PREFIX}:${IMAGE_TAG}" .

