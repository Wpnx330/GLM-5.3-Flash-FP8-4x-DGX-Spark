# Bench notes (2026-08-27)

**Hardware:** 4× DGX Spark, GB10 SM121, TP=4, RoCE fabric  
**Weights:** `dealignai/GLM-5.3-Flash-UNCENSORED-FP8`  
**Image:** SM121 v8 + `persistent_topk` bind-mounts  
**Serve:** MTP k=3, `fp8_e4m3` KV, async scheduling, CUDA graphs on, `gmu=0.85`

Method: streaming `/v1/chat/completions`. TTFT = time to first **content** token. Decode tok/s = completion_tokens / (E2E − TTFT). Large prompts are uniquely salted so prefix cache cannot cheat. Conc 1–5 on each lane. We did not run 15-wide.

Harness: `scripts/bench_lanes.py`.

## Headline (safe to quote)

Single-stream, thinking off. Lane does **not** change short-chat speed.

| Workload | Prefill | Decode | TTFT | E2E |
|---|---|---|---|---|
| Tiny ping (~20 tok) | — | 28–34 tok/s | 0.31–0.33 s | ~1.3 s |
| ~50k unique prompt | ~2,100 tok/s | ~43 tok/s | ~25 s | ~25 s |
| 1280×640 PNG (1085 vision tok) | encoder + 150–700 tok/s | 32–39 tok/s | ~1.5 s warm / ~7 s cold encoder | ~6–11 s |

Five-wide ~50k unique prefills: per-stream decode ~8–10 tok/s, TTFT ~75 s.

## 200k lane (15 seqs × 200K, KV 2.28M, 11.39× at 200K)

### Small

| Conc | TTFT s | E2E s | Decode tok/s | ok |
|---|---|---|---|---|
| 1 | 0.312 | 1.452 | 28.1 | 1/1 |
| 2 | 0.327 | 1.591 | 20.5 | 2/2 |
| 3 | 0.326 | 1.693 | 20.3 | 3/3 |
| 4 | 0.321 | 1.943 | 18.5 | 4/4 |
| 5 | 0.395 | 1.586 | 20.2 | 5/5 |

### Reasoning (thinking on)

Do not quote decode tok/s here. TTFT is first content token; thinking tokens then land in `completion_tokens` and inflate the rate. Quote E2E / TTFT.

| Conc | TTFT s | E2E s | ok |
|---|---|---|---|
| 1 | 4.82 | 7.34 | 1/1 |
| 2 | 6.81 | 9.27 | 2/2 |
| 3 | 16.44 | 20.79 | 3/3 |
| 4 | 12.11 | 17.60 | 4/4 |
| 5 | 15.04 | 21.66 | 5/5 |

### Image (1085 prompt tokens every run)

| Conc | TTFT s | E2E s | Decode tok/s | ok |
|---|---|---|---|---|
| 1 | 7.20 (cold encoder) | 10.51 | 38.7 | 1/1 |
| 2 | 1.50 | 6.77 | 24.4 | 2/2 |
| 3 | 2.05 | 8.45 | 19.3 | 3/3 |
| 4 | 2.40 | 10.36 | 15.9 | 4/4 |
| 5 | 2.94 | 12.08 | 14.1 | 5/5 |

### Large (~52k unique tokens)

| Conc | TTFT s | E2E s | Prefill tok/s | Decode tok/s |
|---|---|---|---|---|
| 1 | 25.5 | 25.7 | 2026 | 42.5 |
| 2 | 38.9 | 42.4 | 1414 | 19.1 |
| 3 | 49.9 | 65.2 | 1267 | 10.1 |
| 4 | 62.6 | 72.4 | 1050 | 11.8 |
| 5 | 74.5 | 93.8 | 937 | 7.8 |

## 500k lane (5 seqs × 500K, KV 2.37M, 4.73× at 500K)

### Small

| Conc | TTFT s | E2E s | Decode tok/s | ok |
|---|---|---|---|---|
| 1 | 0.325 | 1.327 | 31.9 | 1/1 |
| 2 | 0.524 | 2.131 | 15.6 | 2/2 |
| 3 | 0.339 | 1.635 | 18.7 | 3/3 |
| 4 | 0.491 | 2.527 | 13.1 | 4/4 |
| 5 | 5.801 | 7.164 | 17.6 | 5/5 |

c=5 small TTFT 5.8 s is an outlier. 200k c=5 stayed at 0.40 s. Do not average 5.8 into "500k is slower."

### Reasoning (thinking on)

| Conc | TTFT s | E2E s | ok |
|---|---|---|---|
| 1 | 4.55 | 6.34 | 1/1 |
| 2 | 6.96 | 9.65 | 2/2 |
| 3 | 16.06 | 19.90 | 2/3 |
| 4 | 12.16 | 17.67 | 4/4 |
| 5 | 13.53 | 22.31 | 5/5 |

### Image

| Conc | TTFT s | E2E s | Decode tok/s | ok |
|---|---|---|---|---|
| 1 | 1.55 | 5.60 | 31.6 | 1/1 |
| 2 | 1.49 | 6.73 | 24.4 | 2/2 |
| 3 | 1.96 | 9.57 | 16.5 | 3/3 |
| 4 | 2.49 | 10.17 | 16.5 | 4/4 |
| 5 | 2.80 | 14.78 | 10.6 | 5/5 |

### Large (~52k unique)

| Conc | TTFT s | E2E s | Prefill tok/s | Decode tok/s |
|---|---|---|---|---|
| 1 | 24.5 | 24.7 | 2143 | 43.4 |
| 2 | 38.8 | 40.5 | 1430 | 21.6 |
| 3 | 51.1 | 60.6 | 1174 | 13.8 |
| 4 | 62.8 | 69.4 | 1012 | 10.0 |
| 5 | 74.7 | 85.9 | 916 | 9.1 |

## 1m lane (3 seqs × 1,000,000, KV 2.15M, 2.15× at 1M)

Conc 4–5 queues behind `max-num-seqs=3`.

### Small

| Conc | TTFT s | E2E s | Decode tok/s | ok |
|---|---|---|---|---|
| 1 | 0.326 | 1.259 | 34.3 | 1/1 |
| 2 | 0.798 | 2.332 | 18.9 | 2/2 |
| 3 | 0.311 | 2.209 | 14.4 | 3/3 |
| 4 | 1.343 | 3.490 | 23.0 | 4/4 |
| 5 | 1.199 | 3.062 | 14.3 | 5/5 |

### Reasoning (thinking on)

| Conc | TTFT s | E2E s | ok |
|---|---|---|---|
| 1 | 4.83 | 7.58 | 1/1 |
| 2 | 11.17 | 14.34 | 2/2 |
| 3 | 13.92 | 21.72 | 3/3 |
| 4 | 10.03 | 16.49 | 3/4 |
| 5 | 14.14 | 18.99 | 5/5 |

### Image

| Conc | TTFT s | E2E s | Decode tok/s | ok |
|---|---|---|---|---|
| 1 | 6.95 (cold encoder) | 10.56 | 35.5 | 1/1 |
| 2 | 1.48 | 6.22 | 27.0 | 2/2 |
| 3 | 1.92 | 10.23 | 14.4 | 3/3 |
| 4 | 4.58 | 13.09 | 17.9 | 4/4 |
| 5 | 6.48 | 15.35 | 15.8 | 5/5 |

### Large (~52k unique)

| Conc | TTFT s | E2E s | Prefill tok/s | Decode tok/s |
|---|---|---|---|---|
| 1 | 24.7 | 24.9 | 2128 | 43.8 |
| 2 | 37.6 | 49.6 | 1554 | 37.9 |
| 3 | 50.9 | 64.0 | 1216 | 15.3 |
| 4 | 63.8 | 82.2 | 1067 | 9.1 |
| 5 | 75.0 | 82.7 | 917 | 9.8 |

## What we will not claim

- 15 @ 200K at 30 tok/s each. Unmeasured past conc 5.
- 3 full 1M windows concurrent. Engine print is 2.15×. `seqs=3` is occupancy.
- Thinking-on decode tok/s as model speed.
- Prefix-cached large-prefill rates (first 500k large pass hit cache; discarded).
