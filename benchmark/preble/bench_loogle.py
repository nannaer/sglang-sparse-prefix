# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Replay LooGLE short-dependency QA against Preble or round-robin workers.

Uses Preble's prompt template, random question prefix and head/tail truncation.
The loader's answer-derived 35-token cap can be overridden with
--max-new-tokens 45 to match its end-to-end config. Groups the public JSONL by doc_id.
"""

import argparse
import asyncio
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import aiohttp
import numpy as np
from transformers import AutoTokenizer


def prepare(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    groups = defaultdict(list)
    for line in Path(args.dataset).read_text().splitlines():
        row = json.loads(line)
        groups[row["doc_id"]].append(row)
    if not groups:
        raise ValueError("Dataset contains no documents")
    rng = random.Random(args.seed)
    docs = rng.sample(sorted(groups), min(args.patterns, len(groups)))
    scale = args.requests / sum(len(groups[doc]) for doc in docs)
    workload = []
    for doc in docs:
        rows = groups[doc]
        for index in range(math.ceil(len(rows) * scale)):
            row = rows[index % len(rows)]
            nonce = f"{rng.getrandbits(128):032x}"
            prompt = (
                "Please answer the question based on the long texts below. \n"
                f"{row['context']}\nQuestion: {nonce}{row['question']}\nAnswer: "
            )
            tokens = tokenizer.encode(prompt)
            if len(tokens) > args.max_input_tokens:
                half = args.max_input_tokens // 2
                prompt = tokenizer.decode(tokens[:half]) + tokenizer.decode(
                    tokens[-half:]
                )
                tokens = tokenizer.encode(prompt)
            workload.append(
                {
                    "doc_id": doc,
                    "input_ids": tokens,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": (
                            args.max_new_tokens
                            if args.max_new_tokens is not None
                            else min(35, len(tokenizer.encode(row["answer"])))
                        ),
                    },
                }
            )
    rng.shuffle(workload)
    workload = workload[: args.requests]
    target = Path(args.workload)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row) + "\n" for row in workload))
    manifest = vars(args).copy()
    manifest.update(
        actual_requests=len(workload),
        actual_patterns=len({row["doc_id"] for row in workload}),
        dataset_sha256=hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        workload_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    )
    target.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"Prepared {len(workload)} requests / {manifest['actual_patterns']} documents",
        flush=True,
    )


async def replay(args):
    workload = [
        json.loads(line) for line in Path(args.workload).read_text().splitlines()
    ]
    workload = workload[: args.requests]
    if not workload:
        raise ValueError("Workload contains no requests")
    if args.arrivals:
        arrivals = json.loads(Path(args.arrivals).read_text())
        if (
            not isinstance(arrivals, list)
            or len(arrivals) != len(workload)
            or any(
                type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in arrivals
            )
            or any(left > right for left, right in zip(arrivals, arrivals[1:]))
        ):
            raise ValueError(
                "Arrivals must contain one finite, nonnegative, ordered time per request"
            )
    else:
        rng = random.Random(args.seed)
        arrivals = []
        elapsed = 0.0
        for _ in workload:
            arrivals.append(elapsed)
            if args.rate > 0:
                elapsed += rng.expovariate(args.rate)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    semaphore = asyncio.Semaphore(args.concurrency)
    start = time.perf_counter()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10800),
        connector=aiohttp.TCPConnector(limit=0),
    ) as session:

        async def send(index, row):
            await asyncio.sleep(max(0, start + arrivals[index] - time.perf_counter()))
            arrived = time.perf_counter()
            result = {
                "index": index,
                "doc_id": row["doc_id"],
                "prompt_tokens": len(row["input_ids"]),
                "arrival": arrivals[index],
                "success": False,
            }
            url = (
                args.router_url
                if args.policy == "preble"
                else args.worker_urls[index % len(args.worker_urls)]
            )
            payload = {
                "input_ids": row["input_ids"],
                "sampling_params": row["sampling_params"],
                "stream": True,
            }
            async with semaphore:
                result["client_queue_seconds"] = time.perf_counter() - arrived
                ttft = None
                last = None
                done = False
                try:
                    async with session.post(
                        url.rstrip("/") + "/generate", json=payload
                    ) as response:
                        result["worker"] = response.headers.get(
                            "X-Preble-Worker", str(index % len(args.worker_urls))
                        )
                        if response.status != 200:
                            raise RuntimeError(
                                f"HTTP {response.status}: {(await response.text())[:200]}"
                            )
                        async for line in response.content:
                            if not line.startswith(b"data:"):
                                continue
                            data = line[5:].strip()
                            if data == b"[DONE]":
                                done = True
                                break
                            if not data:
                                continue
                            last = json.loads(data)
                            if last.get("error") is not None:
                                raise RuntimeError(f"SSE error: {last['error']}")
                            if (
                                ttft is None
                                and last.get("meta_info", {}).get(
                                    "completion_tokens", 0
                                )
                                > 0
                            ):
                                ttft = time.perf_counter() - arrived
                    if last is None or not done:
                        raise RuntimeError("Incomplete SSE generation")
                    meta = last["meta_info"]
                    reason = meta.get("finish_reason") or {}
                    if reason.get("type") not in ("stop", "length"):
                        raise RuntimeError(
                            f"Generation did not finish successfully: {reason}"
                        )
                    output_tokens = meta["completion_tokens"]
                    cached_tokens = meta.get("cached_tokens", 0)
                    if type(output_tokens) is not int or output_tokens < 0:
                        raise RuntimeError("Invalid completion_tokens")
                    if type(cached_tokens) is not int or cached_tokens < 0:
                        raise RuntimeError("Invalid cached_tokens")
                    result.update(
                        success=True,
                        ttft=ttft,
                        latency=time.perf_counter() - arrived,
                        output_tokens=output_tokens,
                        cached_tokens=cached_tokens,
                        finish_reason=reason,
                        output_sha256=hashlib.sha256(
                            last.get("text", "").encode()
                        ).hexdigest(),
                    )
                except Exception as exc:
                    result["success"] = False
                    result["error"] = str(exc)
            results.append(result)
            if len(results) % 25 == 0:
                print(f"Completed {len(results)}/{len(workload)}", flush=True)

        await asyncio.gather(*(send(index, row) for index, row in enumerate(workload)))
    duration = time.perf_counter() - start
    successful = [row for row in results if row["success"]]
    summary = {
        "config": vars(args),
        "requests": len(results),
        "successful": len(successful),
        "duration_seconds": duration,
        "completed_rps": len(successful) / duration,
        "output_tokens_per_second": sum(row["output_tokens"] for row in successful)
        / duration,
        "cached_token_fraction": sum(row["cached_tokens"] for row in successful)
        / max(1, sum(row["prompt_tokens"] for row in successful)),
        "workload_sha256": hashlib.sha256(Path(args.workload).read_bytes()).hexdigest(),
        "arrivals_sha256": (
            hashlib.sha256(Path(args.arrivals).read_bytes()).hexdigest()
            if args.arrivals
            else None
        ),
    }
    for metric in ("ttft", "latency"):
        values = [row[metric] for row in successful if row[metric] is not None]
        summary[metric + "_observations"] = len(values)
        if values:
            summary[metric + "_mean"] = float(np.mean(values))
            for percentile in (50, 95, 99):
                summary[f"{metric}_p{percentile}"] = float(
                    np.percentile(values, percentile)
                )
    results.sort(key=lambda row: row["index"])
    (output / "requests.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in results)
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if len(successful) != len(results):
        raise SystemExit("Some requests failed; inspect requests.jsonl")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--model-path")
    parser.add_argument("--patterns", type=int, default=105)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Preparation: override the answer cap (45 matches Preble's end-to-end config)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--arrivals",
        help="Replay archived arrival times: a JSON list of elapsed seconds, overrides --rate",
    )
    parser.add_argument("--policy", choices=["preble", "round-robin"], default="preble")
    parser.add_argument("--router-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--worker-urls",
        nargs="+",
        default=["http://127.0.0.1:31000", "http://127.0.0.1:31001"],
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=2,
        help="Poisson requests/sec; zero sends all requests immediately",
    )
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--output")
    args = parser.parse_args()
    if (
        min(args.patterns, args.requests, args.concurrency) < 1
        or args.max_input_tokens < 2
        or args.rate < 0
        or (args.max_new_tokens is not None and args.max_new_tokens < 0)
    ):
        parser.error("Invalid workload or concurrency parameters")
    if args.prepare_only:
        if not args.dataset or not args.model_path:
            parser.error("Preparation needs --dataset and --model-path")
        prepare(args)
    else:
        if args.max_new_tokens is not None:
            parser.error("--max-new-tokens applies to --prepare-only")
        if not args.output:
            parser.error("Replay needs --output")
        asyncio.run(replay(args))


if __name__ == "__main__":
    main()
