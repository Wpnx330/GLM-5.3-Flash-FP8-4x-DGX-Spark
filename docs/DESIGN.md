# Design choices, failures, and the CUDA-graph fix

This is the 4× Spark FP8 recipe. Tony's 2× Spark NVFP4 work is what made GB10 even possible. We took his SM121 image layers, kept native FP8 + DeepGEMM, and ran occupancy lanes on four boxes.

## What we did not copy from GLM-5.2

5.2 on this cluster was QuantTrio int4-int8mix, DCP, B12X sparse MLA, `fp8_ds_mla` KV, a custom eldritch image. That stack is for a different checkpoint.

5.3 Flash is `Glm5NextForConditionalGeneration`: 320B / 18B-active, hybrid KDA + sparse MLA, **NoPE** (`qk_rope_head_dim=0`). Native FP8. DCP is off (`decode_context_parallel_size=1`). We did not graft 5.2 kernels onto it.

## Weights: FP8, not NVFP4

NVFP4 on GB10 is a Marlin dequant path, not native FP4 tensor cores. Smaller on disk, usually slower at decode. We already measured that on 5.2. `dealignai/GLM-5.3-Flash-UNCENSORED-FP8` is the card this recipe serves.

## Stock image does not boot on GB10

`vllm/vllm-openai:glm53-flash-arm64-cu130` resolves `FLASHINFER_MLA_SPARSE_SM120`, which hard-wires `fp8_ds_mla` and asserts `pe_dim == 64` (DeepSeek rope). GLM-5.3 has `pe_dim=0`. Result: 62/62 shards load, then die in KV init.

Forcing any other sparse backend: `compute capability not supported`. There is no CLI way out.

Tony's layered Dockerfiles extend the SM90 **NoPE** sparse-MLA backend (plain cache, not `fp8_ds_mla`) to SM121, bump FlashInfer to 0.6.18 (0.6.17 FA2 NaNs on SM121), re-pin NCCL 2.30.7 and cutlass-dsl 4.6.2 (the FlashInfer pip silently downgrades both), and gate PDL off on SM12x. We skip his NaN-debug v2 layer. Dockerfiles live in `docker/` and are his; credit in the README.

After that image, attention is `FLASHINFER_MLA_SPARSE_SM90`. KV is `fp8_e4m3`. MoE is DeepGEMM FP8. `--block-size 2304` is required for the SM121 DeepGEMM kpool.

## Official ENTRYPOINT is already `vllm serve`

If you pass `vllm serve` again you double the command and the container exits immediately. The GID wrapper is `--entrypoint`. Args start with `vllm serve /model ...`.

## CUDA graphs: not the 5.2 miss

Tony's 5.3 recipe stays on `--enforce-eager` and lists graphs as backlog. 5.2 graphs on this cluster were a `max_cudagraph_capture_size` / profiler-estimate problem.

Dropping `--enforce-eager` here died later, at KV profile:

```
launch_persistent_topk ... FilteredTopK fallback requires >=128KB smem
```

GB10 SMEM is **101,376 B**. Hopper-shaped top-k wants 128 KB. Eager never launched that kernel. Graphs did.

Same class of bug as Tony's fp8-KV CTA tile (Hopper 32 / 228 KB vs GB10 16 / 101 KB).

Fix: skip `persistent_topk` on SM12x, fall back to `top_k_per_row_decode`. Bind-mounts:

- `patches/sparse_attn_indexer.py`
- `patches/sparse_attn_indexer_kpool.py`

plus `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`. After that, capture completed: PIECEWISE 8/8, FULL 3/3, plus speculator graphs. `--async-scheduling` stays on.

Tiny-prompt E2E went from ~2.4 s (eager) to ~1.3 s (graphs). Single-stream decode ~30 tok/s tiny, ~43 tok/s after a ~50k prefill.

## Occupancy lanes, not peak-full-window

At `gmu=0.85` the engine prints ~2.1–2.4M KV tokens. Full-window math:

| Lane | seqs × ctx | Engine print (this hardware) |
|---|---|---|
| 200k | 15 × 200K | ~11× at 200K (not 15 full windows) |
| 500k | 5 × 500K | ~4.7× at 500K |
| 1m | 3 × 1M | **2.15× at 1M** |

We still ship `seqs=15/5/3` because real traffic is mixed occupancy. GLM-5.2 ran 5 @ 200K the same way: five full parallel 200K jobs would have preempted. They almost never arrived that way.

`max-model-len` for the 1M lane is **1,000,000**, not 1,048,576. Native max is 1,048,576; at 0.85 that print is 2.87×, so "3 @ 1M" would be a lie.

Default gmu stays 0.85 so a box that still has a desktop session does not OOM rank 0 (API + EngineCore live there, ~5 GB extra vs workers). Squeeze `0.885` is documented for stripped OS. 0.89+ is how you get `NV_ERR_NO_MEMORY` on the head.

## Decode stack (order we turned things on)

1. SM121 NoPE image (required to boot)
2. 5 × 500K occupancy (daily)
3. MTP k=3 (Tony's TP4 used k=4; position 4 was free-riding)
4. `--max-num-batched-tokens 8192` (4096 is too small with MTP draft slots)
5. Drop eager, add async-scheduling, bind-mount the SMEM patch

Not used: DCP, DeepEP (duplicate NCCL, failed import), NVFP4+Marlin, InstantTensor.

## Multimodal

Vision + video towers load. Encoder cache ~32k tokens, profiled with 1 video item. Image (1280×640 PNG) and a 17s 1080p MP4 both returned real content (product name, badges, CLI commands). MTP drafter does not take vision embeddings.

## Clock wedge

A GB10 can report P0 and sit at ~660 MHz. One wedged rank gates TP=4. Warm reboot does not always clear it. The serve script burns 5s per node before launch.
