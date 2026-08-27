#!/usr/bin/env bash
# Streaming lane bench. Measures TTFT (prefill), decode tok/s, E2E.
#   GLM53_URL=http://HEAD:8000/v1/chat/completions \
#     python3 scripts/bench_lanes.py --lane 500k --out bench/500k.jsonl
from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

URL = os.environ.get("GLM53_URL", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = os.environ.get("GLM53_MODEL", "glm-5.3-flash")
IMG = Path(os.environ.get("GLM53_IMAGE_PATH", ""))

FILLER = (
    "Pharos is a Go-powered CLI for the MCP server registry. "
    "Search, install, publish, and manage Model Context Protocol servers from a terminal. "
    "This paragraph is padding so the prefill path has real tokens to chew. "
)


def large_prompt(target_chars: int = 200_000, salt: str = "0") -> str:
    chunks = []
    n = 0
    i = 0
    while n < target_chars:
        block = f"[{salt}:{i:05d}] {FILLER}"
        chunks.append(block)
        n += len(block)
        i += 1
    return (
        "Read this document. Then reply with exactly: LARGE-OK. "
        "Do not summarize.\n\n" + "".join(chunks)
    )


def payloads(img_b64: str | None) -> dict:
    return {
        "small": {
            "name": "small",
            "body": {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Reply with exactly: PING-OK"}],
                "max_tokens": 32,
                "chat_template_kwargs": {"enable_thinking": False},
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        },
        "reasoning": {
            "name": "reasoning",
            "body": {
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "A factory makes 17 widgets in 3 days at 8 hours/day. "
                            "How many widgets in 12 days at 6 hours/day, same rate? "
                            "Show the arithmetic, then the integer answer on the last line."
                        ),
                    }
                ],
                "max_tokens": 256,
                "chat_template_kwargs": {"enable_thinking": True},
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        },
        "large": {
            "name": "large",
            "body": {
                "model": MODEL,
                "messages": [{"role": "user", "content": large_prompt()}],
                "max_tokens": 48,
                "chat_template_kwargs": {"enable_thinking": False},
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        },
        "image": {
            "name": "image",
            "body": {
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Name the product headline and the four badges. One short paragraph.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_b64 or ''}"},
                            },
                        ],
                    }
                ],
                "max_tokens": 128,
                "chat_template_kwargs": {"enable_thinking": False},
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        },
    }


def one_call(body: dict, timeout: int) -> dict:
    t0 = time.perf_counter()
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    first = None
    usage = {}
    text = []
    err = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    if first is None:
                        first = time.perf_counter()
                    text.append(piece)
    except Exception as e:
        err = str(e)
        if hasattr(e, "read"):
            try:
                err = e.read().decode()[:800]
            except Exception:
                pass
    t1 = time.perf_counter()
    e2e = t1 - t0
    ttft = (first - t0) if first else None
    decode_s = (t1 - first) if first else None
    prompt_t = usage.get("prompt_tokens")
    comp_t = usage.get("completion_tokens")
    decode_tps = (comp_t / decode_s) if decode_s and decode_s > 0 and comp_t else None
    prefill_tps = (prompt_t / ttft) if ttft and ttft > 0 and prompt_t else None
    return {
        "ok": err is None and first is not None,
        "error": err,
        "e2e_s": round(e2e, 3),
        "ttft_s": round(ttft, 3) if ttft else None,
        "decode_s": round(decode_s, 3) if decode_s else None,
        "prompt_tokens": prompt_t,
        "completion_tokens": comp_t,
        "prefill_tps": round(prefill_tps, 2) if prefill_tps else None,
        "decode_tps": round(decode_tps, 2) if decode_tps else None,
        "chars": len("".join(text)),
    }


def run_wave(spec: dict, conc: int, timeout: int) -> list[dict]:
    bodies = []
    for i in range(conc):
        body = json.loads(json.dumps(spec["body"]))
        if spec["name"] == "large":
            body["messages"][0]["content"] = large_prompt(salt=f"{time.time_ns()}-{i}")
        bodies.append(body)
    rows = []
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(one_call, b, timeout) for b in bodies]
        for f in as_completed(futs):
            rows.append(f.result())
    return rows


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("ok")]

    def avg(key):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "n": len(rows),
        "ok": len(ok),
        "ttft_s_avg": avg("ttft_s"),
        "e2e_s_avg": avg("e2e_s"),
        "prefill_tps_avg": avg("prefill_tps"),
        "decode_tps_avg": avg("decode_tps"),
        "prompt_tokens_avg": avg("prompt_tokens"),
        "completion_tokens_avg": avg("completion_tokens"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", default="500k")
    ap.add_argument("--max-conc", type=int, default=5)
    ap.add_argument("--scenarios", default="small,reasoning,large,image")
    ap.add_argument("--out", default="")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    img_b64 = None
    if "image" in names:
        if not IMG.is_file():
            raise SystemExit("image scenario needs GLM53_IMAGE_PATH pointing at a PNG")
        img_b64 = base64.b64encode(IMG.read_bytes()).decode()
    specs = payloads(img_b64)
    out_path = Path(args.out or f"bench/{args.lane}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"endpoint {URL} lane={args.lane} conc=1..{args.max_conc} scenarios={names}")
    print("warmup...")
    one_call(specs["small"]["body"], timeout=120)

    with out_path.open("w") as fh:
        for name in names:
            spec = specs[name]
            for conc in range(1, args.max_conc + 1):
                t0 = time.time()
                rows = run_wave(spec, conc, args.timeout)
                summ = summarize(rows)
                rec = {
                    "lane": args.lane,
                    "scenario": name,
                    "concurrency": conc,
                    "summary": summ,
                    "rows": rows,
                    "wall_s": round(time.time() - t0, 3),
                }
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                s = summ
                print(
                    f"{name:10} c={conc} ok={s['ok']}/{s['n']} "
                    f"ttft={s['ttft_s_avg']}s e2e={s['e2e_s_avg']}s "
                    f"prefill={s['prefill_tps_avg']} tok/s decode={s['decode_tps_avg']} tok/s "
                    f"prompt={s['prompt_tokens_avg']}"
                )
    print("wrote", out_path)


if __name__ == "__main__":
    main()
