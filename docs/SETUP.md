# Setup — 4× DGX Spark, GLM-5.3-Flash FP8

One file drives the whole recipe: `scripts/cluster.env`.

```bash
cp scripts/cluster.env.example scripts/cluster.env
# edit that file, then every script picks it up
```

Do not put IPs or usernames in the launch scripts. `cluster.env` is gitignored.

## What you need

- 4× NVIDIA DGX Spark (GB10, SM121, ARM64, ~128 GiB unified each)
- Docker with NVIDIA runtime on every node
- SSH from the head to the three workers, same user, key auth, no password
- A RoCE / InfiniBand fabric between the four (the 200G QSFP-DD ports, not the 10G management NIC)
- Hugging Face access for the weights (~306 GiB) and the base image (~21 GiB)
- This repo cloned to the **same path** on all four nodes (the launch bind-mounts `patches/` and the GID entrypoint from that path)

This FP8 card is ~78 GiB of weights **per rank**. Tensor parallel 4 is how it fits. See [2-Spark notes](#two-sparks) at the bottom if you only have two boxes.

## 1. Fill `cluster.env`

| Variable | What to set |
|---|---|
| `SSH_USER` | Linux user that can `docker` and SSH between nodes |
| `HEAD_IP` | Fabric IP of rank 0 |
| `WORKER_IPS` | Space-separated fabric IPs of ranks 1–3 |
| `NCCL_IB_HCA` | InfiniBand devices, comma-separated. Dual-rail Spark default is in the example. Single-rail: one device. |
| `NCCL_SOCKET_IFNAME` | Matching Ethernet/RoCE ifaces |
| `GLOO_SOCKET_IFNAME` | One iface for Gloo (usually rail 1) |
| `WEIGHTS` / `CACHE` | Host paths. Defaults under `/var/tmp/models/` are fine. `WEIGHTS` is the single source of truth: serve.sh exports it and every node mounts exactly it. |
| `IMAGE` | Local tag after `build-image.sh` (default `glm53-flash:sm121-v8`) |
| `HF_REPO` | Weight card. Default is `dealignai/GLM-5.3-Flash-UNCENSORED-FP8` |
| `HF_BASE_IMAGE` | `vllm/vllm-openai:glm53-flash-arm64-cu130` |

How to find the fabric IPs and HCAs on a Spark:

```bash
ip -4 addr show | grep -E 'enp1s0f0np0|enP2p1s0f0np0'
ls /sys/class/infiniband
ibstat 2>/dev/null | head
```

Use those IPs for `HEAD_IP` / `WORKER_IPS`. Do not use the 1/10G management NIC for NCCL. Rank 0 must be able to SSH to each worker:

```bash
for ip in $WORKER_IPS; do ssh "$SSH_USER@$ip" hostname; done
```

Optional: `export GLM53_ENV=/abs/path/to/cluster.env` if the file is not next to the scripts.

## 2. Clone this repo on every node

```bash
git clone https://github.com/Wpnx330/GLM-5.3-Flash-FP8-4x-DGX-Spark.git
# same path on all four, e.g. ~/GLM-5.3-Flash-FP8-4x-DGX-Spark
```

Copy `cluster.env` to the workers too (or keep it only on the head if you always launch from the head — workers still need `patches/` and `scripts/glm53-container-entrypoint.sh` at that path).

## 3. Download weights on the head (WAN once)

```bash
mkdir -p "$WEIGHTS"
# from the head, with a Hugging Face token in the environment
hf download dealignai/GLM-5.3-Flash-UNCENSORED-FP8 \
  --local-dir "$WEIGHTS"
# expect 62 shards, ~306 GiB
ls "$WEIGHTS"/model-*-of-00062.safetensors | wc -l
```

**Important — weights get updated upstream.** If you downloaded this card before, re-check the current revision; an older release had a repetition-loop bug later fixed in the weights. Two ways to confirm you have the current files:

```bash
# cheap: compare sizes against the card's file listing (Settings → Files)
# exact: spot-check a shard hash after download
sha256sum "$WEIGHTS"/model-00022-of-00062.safetensors
# compare against the SHA-256 shown on the HF file page for that blob
```

If the card announces a fixed revision, either re-download the changed shards or the whole card into a fresh directory, then point `WEIGHTS` at it in `cluster.env`. The engine mounts whatever `WEIGHTS` says — one variable, every script follows.

Then copy over the fabric:

```bash
./scripts/copy-weights.sh
```

That binds SSH/rsync to `HEAD_IP` so the kernel does not pick the management default route.

## 4. Pull the stock image, build SM121 layers, copy

On the **head only** (ARM64):

```bash
docker pull vllm/vllm-openai:glm53-flash-arm64-cu130
./scripts/build-image.sh
./scripts/copy-image.sh
```

`build-image.sh` layers Tony's SM121 Dockerfiles (v1, v3–v8; skips the NaN-debug v2) onto that base. The last tag is `radixark/vllm-glm53-flash:sm121-v8`, then retagged to `IMAGE` from `cluster.env`. FlashInfer 0.6.18 is the long WAN step.

You still need the bind-mount patches in `patches/` (GB10 SMEM / `persistent_topk`). Those are applied at **run** time, not bake time.

## 5. Pick a lane and start

From the head:

```bash
# daily driver
GLM53_LANE=500k ./scripts/glm53-serve.sh

# volume
GLM53_LANE=200k ./scripts/glm53-serve.sh

# long context
GLM53_LANE=1m ./scripts/glm53-serve.sh
```

| Lane | `max-model-len` | `max-num-seqs` | When to pick it |
|---|---|---|---|
| `200k` | 200,000 | 15 | Many short/medium chats. Your peak of ~5 concurrent jobs is nowhere near 15. |
| `500k` | 500,000 | 5 | Default. Compaction sweet spot. Agent sessions that hold a repo. |
| `1m` | 1,000,000 | 3 | Long dumps. Engine print at gmu 0.85 is ~2.15× full 1M windows. Three jobs that are not all maxed is occupancy; three simultaneous 1M dumps will preempt and slow down. |

Default `gpu-memory-utilization=0.85`. If you stripped the desktop (no GUI), `GLM53_GMU=0.885` is the squeeze. That is not the published default. See README.

Clock check: a GB10 can sit at ~660 MHz and look idle. The serve script burns 5s per node. Override with `GLM_SKIP_CLOCK_CHECK=1` only if you just stopped a live engine (the burn hangs while the GPU is owned).

Cold boot is ~12–15 minutes. Poll:

```bash
./scripts/glm53-serve.sh status
curl -s http://$HEAD_IP:8000/v1/models
```

Stop:

```bash
./scripts/glm53-serve.sh stop
```

Switching lanes is a full restart. One engine, one `max-model-len`.

## 6. Smoke

```bash
curl -s http://$HEAD_IP:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash","messages":[{"role":"user","content":"Reply with exactly: PING-OK"}],"max_tokens":16,"chat_template_kwargs":{"enable_thinking":false}}'
```

Image and video use OpenAI-style `image_url` / `video_url` with a data URL. The tower is loaded. MTP draft is text-only on multimodal requests.

## Lanes vs real occupancy

These seq counts are how we run it, not a promise that 15 full 200K windows fit in the 0.85 KV pool. They do not. Mixed occupancy (some chats at 4K, one at 80K) is what the numbers are for. If you dump 15×200K at once you get preemption and a crawl, not a clean 15-wide decode.

## Two Sparks

This recipe is TP=4 because native FP8 GLM-5.3-Flash is ~78 GiB/rank after load. Two Sparks is TP=2: weights per rank roughly double, KV collapses, 1M is gone, 500K is ugly.

If you only have two boxes:

- Do **not** use these occupancy lanes as-is.
- Tony's NVFP4 recipe on 2 Sparks is the working TP=2 path: [tonyd2wild/GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark) (Marlin, 262K). NVFP4 on GB10 is a dequant path, usually slower decode than FP8+DeepGEMM.
- If you still want this FP8 card on two nodes, you would set `WORKER_IPS` to one IP, accept TP=2 from `NNODES`, drop `max-model-len` hard (try 64K–128K, `max-num-seqs=1` or `2`, `gmu=0.80`), and expect OOM until you tune. We did not ship or bench that. No 2-Spark lane files in this repo.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `pe_dim must be 64 for fp8_ds_mla` | Stock image, not the SM121 layers. Rebuild v1–v8. |
| `compute capability not supported` | Forced a non-SM90/SM120 sparse backend. Leave backend on auto after the image patches. |
| `persistent_topk` / 128KB SMEM | Graphs on without `patches/sparse_attn_indexer*.py` bind-mounts. |
| Clock check hangs | Live container still owns the GPU. `--stop` first or `GLM_SKIP_CLOCK_CHECK=1`. |
| API empty for 10+ min | Weight load. 62 shards, ~10 min, then graph capture. |
| NCCL timeout | Wrong `NCCL_*` in `cluster.env`, or SSH/rsync used the management NIC. |
| Head OOM / `NV_ERR_NO_MEMORY` | gmu too high. Rank 0 also holds the API. Stay at 0.85 unless the OS is stripped. |

Design notes and the CUDA-graph fix: [docs/DESIGN.md](DESIGN.md). Numbers: [docs/BENCH.md](BENCH.md).
