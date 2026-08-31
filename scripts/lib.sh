#!/usr/bin/env bash
# Shared loader. Source this from every script.
# Looks for cluster.env next to this file, then $GLM53_ENV, then CWD.

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_repo="$(cd "$_here/.." && pwd)"

load_cluster_env() {
  local f
  for f in \
    "${GLM53_ENV:-}" \
    "$_here/cluster.env" \
    "$_repo/cluster.env" \
    "$PWD/cluster.env"
  do
    [ -n "$f" ] && [ -f "$f" ] && { # shellcheck disable=SC1090
      set -a
      # shellcheck source=/dev/null
      . "$f"
      set +a
      GLM53_ENV_FILE="$f"
      return 0
    }
  done
  echo "Missing cluster.env." >&2
  echo "  cp scripts/cluster.env.example scripts/cluster.env" >&2
  echo "  edit SSH_USER, HEAD_IP, WORKER_IPS, NCCL_* to match your fabric" >&2
  echo "  see docs/SETUP.md" >&2
  return 1
}

require_cluster() {
  load_cluster_env || exit 2
  : "${SSH_USER:?set SSH_USER in cluster.env}"
  : "${HEAD_IP:?set HEAD_IP in cluster.env}"
  : "${WORKER_IPS:?set WORKER_IPS in cluster.env}"
  # Any host path works; set WEIGHTS in cluster.env. Download the CURRENT
  # revision of HF_REPO (see docs/SETUP.md step 3 for the revision check).
  : "${WEIGHTS:=/var/tmp/models/glm53-flash-fp8}"
  : "${CACHE:=/var/tmp/models/glm53-cache}"
  : "${IMAGE:=glm53-flash:sm121-v8}"
  : "${NAME:=vllm_glm53}"
  : "${PORT:=8000}"
  : "${MASTER_PORT:=29500}"
  : "${HF_REPO:=dealignai/GLM-5.3-Flash-UNCENSORED-FP8}"
  : "${HF_BASE_IMAGE:=vllm/vllm-openai:glm53-flash-arm64-cu130}"
  : "${NCCL_IB_HCA:=rocep1s0f0,roceP2p1s0f0}"
  : "${NCCL_SOCKET_IFNAME:=enp1s0f0np0,enP2p1s0f0np0}"
  : "${GLOO_SOCKET_IFNAME:=enp1s0f0np0}"
  # NODES: head first
  # shellcheck disable=SC2206
  NODES=($HEAD_IP $WORKER_IPS)
  NNODES="${#NODES[@]}"
  if [ "$NNODES" -lt 2 ]; then
    echo "Need HEAD_IP plus at least one worker in WORKER_IPS." >&2
    exit 2
  fi
  REPO_ROOT="$_repo"
  SCRIPT_DIR="$_here"
  ssh_to() {
    ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
      -o ConnectTimeout=8 "$SSH_USER@$1" "$@"
  }
}
