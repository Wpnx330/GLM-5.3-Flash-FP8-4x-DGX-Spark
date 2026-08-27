#!/usr/bin/env bash
# rsync weights from the head to every worker over the IPs in cluster.env.
# Run this ON THE HEAD after hf download finishes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "$HERE/lib.sh"
require_cluster

echo "rsync $WEIGHTS -> workers: $WORKER_IPS  (user=$SSH_USER bind=$HEAD_IP)"
pids=()
for w in $WORKER_IPS; do
  echo "==> $w"
  ssh_to "$w" "mkdir -p $WEIGHTS"
  rsync -aH --info=progress2 --partial \
    -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 -b ${HEAD_IP}" \
    "${WEIGHTS}/" "${SSH_USER}@${w}:${WEIGHTS}/" \
    > "/tmp/glm53-rsync-${w}.log" 2>&1 &
  pids+=($!)
done
fail=0
i=0
for w in $WORKER_IPS; do
  if wait "${pids[$i]}"; then echo "ok $w"; else echo "FAIL $w (see /tmp/glm53-rsync-${w}.log)"; fail=1; fi
  i=$((i+1))
done
[ "$fail" = 0 ] || exit 1
echo "done"
