# GLM-5.3-Flash-uncensored FP8 | 4× DGX Spark | 34 tok/s @ 1m cxt

Native FP8, uncensored, multimodal, tensor-parallel 4, 1m context at 28-34 tok/s.

Single-stream decode **28–34 tok/s** on a tiny prompt and **~43 tok/s** after a ~50k-token prefill. Prefill on that 50k prompt is **~2,100 tok/s**. Short-chat speed does not change when you pick the 200k, 500k, or 1M lane. Image and video towers load and answer.

Stock `vllm/vllm-openai:glm53-flash-arm64-cu130` does not boot this model on GB10. This repo is the recipe that does: SM121 image layers, occupancy lanes, CUDA graphs.

**Step by step:** [docs/SETUP.md](docs/SETUP.md)  
**Numbers:** [docs/BENCH.md](docs/BENCH.md)  
**Why it broke and what we changed:** [docs/DESIGN.md](docs/DESIGN.md)

Weights are not in this repo. Pull [dealignai/GLM-5.3-Flash-UNCENSORED-FP8](https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-FP8) yourself (~306 GiB, 62 shards).

## Headline numbers (2026-08-27, 4× GB10, `gmu=0.85`)

| | Tiny ping (~20 tok) | ~50k unique prompt | 1280×640 PNG |
|---|---|---|---|
| Decode | 28–34 tok/s | ~43 tok/s | 32–39 tok/s |
| Prefill | — | ~2,100 tok/s | encoder + 150–700 tok/s |
| TTFT | 0.31–0.33 s | ~25 s | ~1.5 s warm / ~7 s cold encoder |
| E2E | ~1.3 s | ~25 s | ~6–11 s |

Five concurrent ~50k unique prefills: per-stream decode drops to ~8–10 tok/s, TTFT ~75 s. That is HBM sharing, not the 1M flag.

CUDA graphs are on. Tiny-prompt E2E was ~2.4 s with `--enforce-eager` and ~1.3 s after graphs captured.

## Lanes

One engine, one `max-model-len`. Restart to switch. Default `gpu-memory-utilization=0.85`.

```bash
cp scripts/cluster.env.example scripts/cluster.env   # IPs, user, fabric. One file.
GLM53_LANE=500k ./scripts/glm53-serve.sh             # default
GLM53_LANE=200k ./scripts/glm53-serve.sh
GLM53_LANE=1m   ./scripts/glm53-serve.sh
```

| Lane | Context | Seq slots | Pick this when |
|---|---|---|---|
| `200k` | 200,000 | 15 | Many short and medium chats |
| `500k` | 500,000 | 5 | Daily driver. Agent sessions that hold a repo |
| `1m` | 1,000,000 | 3 | Long dumps. Not three simultaneous full 1M windows |

These seq counts are **occupancy**, the way people actually use a box. They are not a promise that 15 full 200K windows fit in the 0.85 KV pool. They do not. Engine print at 0.85 is ~2.1–2.4M KV tokens: about 11× at 200K, 4.7× at 500K, **2.15× at 1M**. Mixed chats (4K, 4K, 80K) are what 15/5/3 are sized for. Dump every slot at max length and you get preemption and a crawl.

If you stripped the desktop, `GLM53_GMU=0.885` is the squeeze. Leave 0.85 if the GUI is still there. Rank 0 also holds the API process.

How to fill `cluster.env`, download weights, build the image, and copy over RoCE: [docs/SETUP.md](docs/SETUP.md).

## How we tested

Streaming `/v1/chat/completions` on the live cluster. TTFT = time to first **content** token. Decode tok/s = `completion_tokens / (E2E − TTFT)`.

Four scenarios × concurrency 1 through 5, on each lane:

1. Tiny ping (`PING-OK`, thinking off)
2. A short reasoning prompt (thinking on)
3. A ~50k-token unique pad (salted every call so prefix cache cannot cheat)
4. The Pharos CLI PNG (`image_url`, 1085 prompt tokens every run)

Video was checked earlier (17s 1080p MP4, named the CLI and the commands). It is not in the 1–5 matrix.

Thinking-on decode tok/s is inflated: thinking tokens flush into `completion_tokens` after the first content token. Quote E2E / TTFT for those rows, not tok/s.

We did **not** run 15 concurrent jobs. Do not read 15 @ 200K as 15 × 30 tok/s.

Full tables: [docs/BENCH.md](docs/BENCH.md). Harness: `scripts/bench_lanes.py`.

## What had to change to boot

1. **Stock image dies after 62/62 shards.** SM120 sparse MLA hard-wires `fp8_ds_mla` and wants `pe_dim=64`. GLM-5.3 is NoPE (`qk_rope_head_dim=0`). Other backends: `compute capability not supported`. Tony's SM121 Dockerfiles extend the SM90 NoPE backend, bump FlashInfer to 0.6.18, and re-pin NCCL / cutlass-dsl. Those files are in `docker/`.
2. **`--kv-cache-dtype fp8` is the wrong flag.** It selected `fp8_ds_mla`. We serve `fp8_e4m3`.
3. **Official image ENTRYPOINT is already `vllm serve`.** Passing it again exits immediately. GID wrapper is `--entrypoint`.
4. **CUDA graphs.** Tony's 5.3 writeup stays on `--enforce-eager`. Dropping it here hit `persistent_topk` wanting 128 KB SMEM. GB10 has 101 KB. Bind-mounts in `patches/` skip that kernel on SM12x. Graphs then captured (PIECEWISE 8/8, FULL 3/3). `--async-scheduling` stays on.
5. **FP8, not NVFP4.** NVFP4 on GB10 is Marlin dequant, usually slower decode.
6. **No DCP.** This card does not need it. `decode_context_parallel_size=1`.
7. **MTP k=3** (k=4 A/B'd 2026-08-28: 4th draft acceptance ~0.5 → net slower), batched
   8192, `--block-size 2304`. Draft length is a knob: `GLM53_MTP_K=4 ./scripts/glm53-serve.sh`.
   That A/B ran on an older weights revision; re-testing k=4 on newer cards is fair game.
   Next-up: Inco's **DFlash2** drafter (SGLang-only today).

Longer version: [docs/DESIGN.md](docs/DESIGN.md).

## Two Sparks

This recipe is TP=4 because native FP8 is ~78 GiB of weights per rank. Two boxes means TP=2, twice the weights per GPU, almost no KV, no 1M lane.

Tony's NVFP4 + Marlin recipe is the working 2-Spark path: [GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark) (262K, ~36 tok/s on TP4 NVFP4). We did not ship 2-Spark occupancy files for this FP8 card. Hints only in SETUP.md.

## Credits

- [tonyd2wild](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark) — SM121 day-0 bugs, Dockerfiles v1–v8, NVIDIA forum writeup. Image layers in `docker/` are his.
- [dealignai](https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-FP8) — FP8 uncensored checkpoint.
- Zhipu / zai-org — GLM-5.3-Flash.

## License

MIT for the scripts and patches in this repo. Model weights and the vLLM image have their own licenses. Tony's Dockerfiles remain under whatever license his repo ships.
