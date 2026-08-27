#!/usr/bin/env bash
# Launch GLM-5.3-Flash FP8 TP=N across the nodes in cluster.env.
# Workers first (headless), then the head. Does not pull weights.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "$HERE/lib.sh"
require_cluster

IMAGE="${GLM53_IMAGE:-$IMAGE}"
NAME="${GLM53_NAME:-$NAME}"
PORT="${GLM53_PORT:-$PORT}"
MASTER_PORT="${GLM53_MASTER_PORT:-$MASTER_PORT}"
WEIGHTS_HOST="${GLM53_WEIGHTS:-$WEIGHTS}"
CACHE_HOST="${GLM53_CACHE:-$CACHE}"
ENTRYPOINT_HOST="${GLM53_ENTRYPOINT:-$REPO_ROOT/scripts/glm53-container-entrypoint.sh}"
PATCH_HOST="${GLM53_PATCHES:-$REPO_ROOT/patches}"

# Occupancy lanes, not peak-full-window math. At gmu=0.85 the KV pool is
# ~2.1–2.4M tokens. 15×200K / 5×500K / 3×1M would not all fit if every
# seq were full. Mixed occupancy is the design. Squeeze gmu 0.885 is a
# README note for stripped OS, not the default.
LANE="${GLM53_LANE:-500k}"
case "$LANE" in
  200k|200K) _LEN=200000;  _SEQS=15 ;;
  500k|500K) _LEN=500000;  _SEQS=5  ;;
  1m|1M|1000k|1000K) _LEN=1000000; _SEQS=3 ;;
  *) printf 'unknown GLM53_LANE=%s (use 200k|500k|1m)\n' "$LANE" >&2; exit 2 ;;
esac
MAX_MODEL_LEN="${GLM53_MAXLEN:-$_LEN}"
MAX_NUM_SEQS="${GLM53_SEQS:-$_SEQS}"
MAX_BATCHED="${GLM53_BATCHED:-8192}"
GMU="${GLM53_GMU:-0.85}"
SERVED_NAME="${GLM53_SERVED_NAME:-glm-5.3-flash}"
TP="${#NODES[@]}"

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
die()  { printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

DRYRUN=0; STOP=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRYRUN=1 ;;
    --stop)    STOP=1 ;;
    *) die "unknown arg: $a (use --dry-run or --stop)" ;;
  esac
done

HEAD="$HEAD_IP"

if [ "$STOP" = 1 ]; then
  say "stopping '$NAME' on all ${NNODES} nodes"
  for ip in "${NODES[@]}"; do
    ssh_to "$ip" "docker rm -f $NAME 2>/dev/null" \
      && printf '   stopped on %s\n' "$ip"
  done
  exit 0
fi

ENVV=(
  -e "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800"
  -e "VLLM_ENGINE_READY_TIMEOUT_S=3600"
  -e "HF_HOME=/cache/huggingface"
  -e "TRITON_CACHE_DIR=/cache/huggingface/.tritoncache"
  -e "HF_HUB_OFFLINE=1"
  -e "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
  -e "TORCH_CUDA_ARCH_LIST=12.1a"
  -e "NCCL_NET=IB"
  -e "NCCL_IB_DISABLE=0"
  -e "NCCL_IB_HCA=$NCCL_IB_HCA"
  -e "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
  -e "GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
  -e "NCCL_MAX_NCHANNELS=4"
  -e "NCCL_MIN_NCHANNELS=4"
  -e "NCCL_CROSS_NIC=1"
  -e "NCCL_CUMEM_ENABLE=0"
  -e "NCCL_IGNORE_CPU_AFFINITY=1"
  -e "NCCL_DEBUG=WARN"
  -e "NCCL_IB_TC=106"
  -e "NCCL_NET_PLUGIN=none"
  -e "NCCL_IB_MERGE_NICS=0"
  -e "NCCL_IB_SUBNET_AWARE_ROUTING=1"
  -e "FLASHINFER_CUDA_ARCH_LIST=12.1a"
  -e "FLASHINFER_DISABLE_VERSION_CHECK=1"
  -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  -e "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0"
)

VLLM_LAYERS="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers"

BASE=(
  --cap-add IPC_LOCK --ulimit memlock=-1:-1
  --network host --ipc host --shm-size 10gb --gpus all
  --device /dev/infiniband:/dev/infiniband
  -v "$WEIGHTS_HOST:/model:ro"
  -v "$CACHE_HOST:/cache/huggingface"
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro
  -v "$ENTRYPOINT_HOST:/glm53-container-entrypoint.sh:ro"
  -v "$PATCH_HOST/sparse_attn_indexer.py:$VLLM_LAYERS/sparse_attn_indexer.py:ro"
  -v "$PATCH_HOST/sparse_attn_indexer_kpool.py:$VLLM_LAYERS/sparse_attn_indexer_kpool.py:ro"
)

# Native FP8 + DeepGEMM. Official image ENTRYPOINT is ["vllm","serve"] —
# override to the GID wrapper, then pass `vllm serve ...` as args.
SERVE=(
  vllm serve /model
  --served-model-name "$SERVED_NAME" --host 0.0.0.0 --port "$PORT"
  --trust-remote-code
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  --tensor-parallel-size "$TP" --pipeline-parallel-size 1
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_BATCHED"
  --gpu-memory-utilization "$GMU"
  --block-size 2304
  --kv-cache-dtype fp8_e4m3
  --async-scheduling
  --distributed-executor-backend mp
)

docker_run_cmd() {
  local rank="$1" headless="$2"
  local cmd=(docker run -d --name "$NAME" "${BASE[@]}" "${ENVV[@]}"
             --entrypoint /glm53-container-entrypoint.sh
             -e "NODE_RANK=$rank" -e "MASTER_ADDR=$HEAD"
             "$IMAGE" "${SERVE[@]}"
             --nnodes "$NNODES" --node-rank "$rank" --master-addr "$HEAD" --master-port "$MASTER_PORT")
  [ "$headless" = 1 ] && cmd+=(--headless)
  local out="" t
  for t in "${cmd[@]}"; do out+=" $(printf '%q' "$t")"; done
  printf '%s' "${out# }"
}

say "GLM-5.3-Flash FP8 launch: ${NNODES} nodes TP=$TP head=$HEAD:$PORT image=$IMAGE"
say "weights=$WEIGHTS_HOST served-as=$SERVED_NAME lane=$LANE ctx=$MAX_MODEL_LEN seqs=$MAX_NUM_SEQS gmu=$GMU"
[ "$DRYRUN" = 1 ] && echo "   (dry-run — nothing will be executed)"

for ((rank=1; rank<NNODES; rank++)); do
  w="${NODES[$rank]}"
  run="$(docker_run_cmd "$rank" 1)"
  shell="docker rm -f $NAME 2>/dev/null; mkdir -p $CACHE_HOST; $run"
  if [ "$DRYRUN" = 1 ]; then
    printf '\n# worker %s (rank %d, headless)\nssh %s@%s %q\n' "$w" "$rank" "$SSH_USER" "$w" "$shell"
  else
    printf '   worker %s rank=%d (headless)\n' "$w" "$rank"
    ssh_to "$w" "$shell" || die "worker launch failed on $w"
  fi
done

run="$(docker_run_cmd 0 0)"
shell="docker rm -f $NAME 2>/dev/null; mkdir -p $CACHE_HOST; $run"
if [ "$DRYRUN" = 1 ]; then
  printf '\n# head %s (rank 0)\n%s\n' "$HEAD" "$shell"
  exit 0
fi
printf '   head %s rank=0\n' "$HEAD"
bash -c "$shell" || die "head launch failed"

say "launched"
echo "   poll:  curl -s http://${HEAD}:$PORT/v1/models"
echo "   logs:  docker logs -f $NAME   (on the head node)"
echo "   stop:  $0 --stop"
