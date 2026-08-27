#!/usr/bin/env bash
# Layer Tony's SM121 Dockerfiles onto the official glm53-flash ARM64 image.
# Run ON THE HEAD (aarch64 / GB10). Skips the NaN-debug v2 layer.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "$HERE/lib.sh"
require_cluster

DOCKER_DIR="$REPO_ROOT/docker"
cd "$DOCKER_DIR"

echo "base $HF_BASE_IMAGE must already be pulled"
docker image inspect "$HF_BASE_IMAGE" >/dev/null

# v1 tags as radixark/... because later Dockerfiles FROM those names.
echo "===== v1 NoPE MLA SM121 ====="
docker build -f Dockerfile.glm53-sm121 -t radixark/vllm-glm53-flash:sm121-nope-mla "$REPO_ROOT"

echo "===== v3 FlashInfer 0.6.18 ====="
docker build -f Dockerfile.glm53-sm121-v3 -t radixark/vllm-glm53-flash:sm121-fi618 "$DOCKER_DIR"

echo "===== v4 NCCL 2.30.7 ====="
docker build -f Dockerfile.glm53-sm121-v4 -t radixark/vllm-glm53-flash:sm121-fi618-nccl "$DOCKER_DIR"

echo "===== v5 cutlass-dsl 4.6.2 ====="
docker build -f Dockerfile.glm53-sm121-v5 -t radixark/vllm-glm53-flash:sm121-final "$DOCKER_DIR"

echo "===== v6 PDL off ====="
docker build -f Dockerfile.glm53-sm121-v6 -t radixark/vllm-glm53-flash:sm121-v6 "$DOCKER_DIR"

echo "===== v7 indexer ====="
docker build -f Dockerfile.glm53-sm121-v7 -t radixark/vllm-glm53-flash:sm121-v7 "$DOCKER_DIR"

echo "===== v8 fp8 KV CTA ====="
docker build -f Dockerfile.glm53-sm121-v8 -t radixark/vllm-glm53-flash:sm121-v8 "$DOCKER_DIR"

docker tag radixark/vllm-glm53-flash:sm121-v8 "$IMAGE"
echo "tagged $IMAGE"
docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep -E 'glm53|sm121' || true
