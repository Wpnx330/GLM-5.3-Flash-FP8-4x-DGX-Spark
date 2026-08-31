#!/usr/bin/env bash
# Wrapper. Preflight (clock check, cache drop) then glm53-node-launch.sh.
#
#   GLM53_LANE=500k ./scripts/glm53-serve.sh           # default: 5 seqs × 500K
#   GLM53_LANE=200k ./scripts/glm53-serve.sh           # 15 seqs × 200K
#   GLM53_LANE=1m   ./scripts/glm53-serve.sh           # 3 seqs × 1M
#   GLM53_GMU=0.885 GLM53_LANE=200k ./scripts/glm53-serve.sh   # bare-metal squeeze
#   ./scripts/glm53-serve.sh stop|status|logs|dry-run
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "$HERE/lib.sh"
require_cluster

LAUNCH="$HERE/glm53-node-launch.sh"
IMAGE="${GLM53_IMAGE:-$IMAGE}"
NAME="${GLM53_NAME:-$NAME}"
PORT="${GLM53_PORT:-$PORT}"

case "${1:-start}" in
  stop)
    "$LAUNCH" --stop
    exit 0
    ;;
  status)
    echo "--- container on all nodes ---"
    for ip in "${NODES[@]}"; do
      echo -n "${ip}: "
      ssh_to "$ip" "docker ps --filter name=$NAME --format '{{.Status}}'" 2>/dev/null || echo unreachable
    done
    echo
    echo "--- /v1/models on $HEAD_IP:$PORT ---"
    curl -s --max-time 5 "http://${HEAD_IP}:${PORT}/v1/models" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "(API not ready yet)"
    exit 0
    ;;
  logs)
    docker logs -f "$NAME"
    exit 0
    ;;
  dry-run)
    "$LAUNCH" --dry-run
    exit 0
    ;;
  start)
    ;;
  *)
    echo "Usage: $0 {start|stop|status|logs|dry-run}"
    exit 1
    ;;
esac

echo "[guard] FP8 weights present at $WEIGHTS"
for ip in "${NODES[@]}"; do
  if ! ssh_to "$ip" "test -f $WEIGHTS/config.json"; then
    echo "Error: missing $WEIGHTS/config.json on $ip"
    echo "       Download on the head, then scripts/copy-weights.sh"
    exit 1
  fi
done

echo "[guard] image $IMAGE present"
for ip in "${NODES[@]}"; do
  if ! ssh_to "$ip" "docker image inspect $IMAGE > /dev/null 2>&1"; then
    echo "Error: image $IMAGE missing on $ip"
    echo "       Build on the head (docs/SETUP.md), then scripts/copy-image.sh"
    exit 1
  fi
done

echo "[guard] patches + entrypoint present on every node"
for ip in "${NODES[@]}"; do
  if ! ssh_to "$ip" "test -f $REPO_ROOT/patches/sparse_attn_indexer.py && test -f $REPO_ROOT/scripts/glm53-container-entrypoint.sh"; then
    echo "Error: clone this repo to the same path on $ip (or set GLM53_PATCHES / GLM53_ENTRYPOINT)"
    echo "       Default path: $REPO_ROOT"
    exit 1
  fi
done

if [ "${GLM_SKIP_CLOCK_CHECK:-0}" != 1 ]; then
  echo "[preflight] GPU clock-health check (5s burn) ..."
  clock_bad=0
  for ip in "${NODES[@]}"; do
    mhz=$(ssh_to "$ip" "docker run --rm --gpus all --entrypoint python3 $IMAGE -c '
import torch,time,subprocess
a=torch.randn(8192,8192,device=\"cuda\",dtype=torch.bfloat16)
t=time.time()
while time.time()-t<5:(a@a).sum().item()
print(subprocess.run([\"nvidia-smi\",\"--query-gpu=clocks.current.sm\",\"--format=csv,noheader,nounits\"],capture_output=True,text=True).stdout.split()[0])
' 2>/dev/null" 2>/dev/null || true)
    if [ -z "$mhz" ]; then echo "  $ip: clock probe failed (skipped)"
    elif [ "$mhz" -lt 1500 ]; then echo "  $ip: WEDGED — SM ${mhz} MHz"; clock_bad=1
    else echo "  $ip: SM ${mhz} MHz — healthy"; fi
  done
  if [ "$clock_bad" = 1 ]; then
    echo "A GPU is stuck at low clock. Cold power cycle that node, or GLM_SKIP_CLOCK_CHECK=1."
    exit 1
  fi
fi

echo "[preflight] dropping page cache ..."
for ip in "${NODES[@]}"; do
  ssh_to "$ip" "docker run --rm --privileged -v /proc:/host_proc alpine sh -c 'sync && echo 3 > /host_proc/sys/vm/drop_caches' >/dev/null 2>&1 && echo '  $ip: cache dropped'" || echo "  $ip: skip"
done

# Single source of truth: whatever WEIGHTS resolved to here (cluster.env or
# environment) is exactly what the node launcher mounts. No per-script copies.
export GLM53_WEIGHTS="$WEIGHTS"

"$LAUNCH"

echo
echo "Cold boot is ~12–15 min (weight load + graphs)."
echo "  Poll:    curl -s http://${HEAD_IP}:${PORT}/v1/models"
echo "  Status:  $0 status"
echo "  Logs:    $0 logs"
echo "  Stop:    $0 stop"
