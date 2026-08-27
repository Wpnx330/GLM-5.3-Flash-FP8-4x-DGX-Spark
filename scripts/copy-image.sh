#!/usr/bin/env bash
# docker save on the head, rsync tar, docker load on workers.
# Run ON THE HEAD after scripts/build-image.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "$HERE/lib.sh"
require_cluster

TAR="${GLM53_IMAGE_TAR:-/var/tmp/glm53-sm121-v8.tar}"
echo "saving $IMAGE -> $TAR"
docker save -o "$TAR" "$IMAGE"
echo "copy + load on workers"
for w in $WORKER_IPS; do
  echo "==> $w"
  rsync -aH --info=progress2 \
    -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 -b ${HEAD_IP}" \
    "$TAR" "${SSH_USER}@${w}:${TAR}"
  ssh_to "$w" "docker load -i $TAR && docker image inspect $IMAGE >/dev/null"
  echo "ok $w"
done
echo "done"
