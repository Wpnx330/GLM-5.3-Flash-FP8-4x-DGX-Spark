# SM121 image layers

These Dockerfiles are from [tonyd2wild/GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark) (`docker/`). They patch the stock `vllm/vllm-openai:glm53-flash-arm64-cu130` image so GLM-5.3-Flash (NoPE MLA) can run on GB10.

We skip `Dockerfile.glm53-sm121-v2` (NaN debug). `scripts/build-image.sh` builds v1, then v3 through v8, and tags the result as `IMAGE` from `cluster.env`.

Run-time bind-mounts in `../patches/` are ours: they gate `persistent_topk` off on SM12x so CUDA graphs can capture. Tony's published 5.3 recipe still uses `--enforce-eager`.
