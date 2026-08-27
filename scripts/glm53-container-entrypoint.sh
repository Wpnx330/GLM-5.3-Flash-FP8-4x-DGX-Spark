#!/usr/bin/env bash
# IN-CONTAINER ENTRYPOINT — pick a RoCEv2 IPv4 GID, then exec vllm.
# Bind-mounted read-only from the host. Official image ENTRYPOINT is
# ["vllm","serve"]; we replace it with this wrapper and pass `vllm serve ...`.
set -eu

HCA="${NCCL_IB_HCA%%,*}"
if [ -n "${HCA:-}" ] && [ -d "/sys/class/infiniband/$HCA/ports/1" ]; then
  for i in $(seq 0 15); do
    t=$(cat "/sys/class/infiniband/$HCA/ports/1/gid_attrs/types/$i" 2>/dev/null || true)
    g=$(cat "/sys/class/infiniband/$HCA/ports/1/gids/$i" 2>/dev/null || true)
    case "$t" in
      *"RoCE v2"*)
        case "$g" in
          *"0000:0000:0000:0000:0000:ffff:"*)
            export NCCL_IB_GID_INDEX=$i
            echo "[glm53-entrypoint] HCA=$HCA NCCL_IB_GID_INDEX=$i gid=$g"
            break
            ;;
        esac
        ;;
    esac
  done
fi
if [ -z "${NCCL_IB_GID_INDEX:-}" ]; then
  echo "[glm53-entrypoint] WARNING: no RoCEv2 IPv4 GID for HCA=$HCA; NCCL will auto-select" >&2
fi

exec "$@"
