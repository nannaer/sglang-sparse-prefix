# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Warm up, then freeze a hardware scale for Preble's reference prefill model.

Use idle workers with the benchmark's model and backend settings. Each pass runs
six fixed profiles serially per worker, with independent workers in parallel.
This measures a new calibration; it does not replace the archived frozen values.
"""

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import statistics
import time
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp

from sglang.srt.routing.preble_cost import a6000_prefill_time

PROFILES = (
    (8192, 0),
    (16384, 0),
    (32722, 0),
    (16896, 16384),
    (18432, 16384),
    (32722, 30674),
)
REFERENCE_CALIBRATION_WORKLOAD_SHA256 = (
    "b9fbf417c3f588e7b92dd134fff42eead61b7bf3aa10ff3c7020db2f96b74d68"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_prompt(path):
    raw = path.read_bytes()
    longest, selected = [], None
    for index, line in enumerate(raw.splitlines()):
        row = json.loads(line)
        tokens = row.get("input_ids") if isinstance(row, dict) else None
        require(
            isinstance(tokens, list)
            and bool(tokens)
            and all(type(token) is int and token >= 0 for token in tokens),
            f"Workload row {index} needs a nonempty list of nonnegative token IDs",
        )
        if len(tokens) > len(longest):
            longest, selected = tokens, index
    require(
        len(longest) >= max(n for n, _ in PROFILES),
        "Longest prompt needs at least 32722 tokens",
    )
    return longest, {
        "workload": str(path.resolve()),
        "workload_sha256": hashlib.sha256(raw).hexdigest(),
        "selected_request_index": selected,
        "selected_prompt_tokens": len(longest),
        "selected_input_ids_sha256": hashlib.sha256(
            json.dumps(longest, separators=(",", ":")).encode()
        ).hexdigest(),
        "input_ids_hash_encoding": "UTF-8 JSON with separators=(',', ':')",
        "selection": "First longest input_ids in workload order",
        "reference_calibration_workload_sha256": REFERENCE_CALIBRATION_WORKLOAD_SHA256,
    }


async def request(session, url, tokens, budget):
    started = time.perf_counter()
    ttft, last, done = None, None, False
    async with session.post(
        url + "/generate",
        json={
            "input_ids": tokens,
            "sampling_params": {"temperature": 0, "max_new_tokens": budget},
            "stream": True,
        },
    ) as response:
        response.raise_for_status()
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
            require(isinstance(last, dict), "SSE data must be an object")
            require(last.get("error") is None, f"SSE error: {last.get('error')}")
            meta = last.get("meta_info") or {}
            count = meta.get("completion_tokens", 0)
            require(type(count) is int and count >= 0, "Invalid completion_tokens")
            if ttft is None and count > 0:
                ttft = time.perf_counter() - started
    elapsed = time.perf_counter() - started
    require(done and last is not None and ttft is not None, "Incomplete SSE generation")
    meta = last["meta_info"]
    reason = meta.get("finish_reason") or {}
    require(
        reason.get("type") in ("stop", "length"),
        "Generation did not finish successfully",
    )
    count = meta["completion_tokens"]
    cached = meta.get("cached_tokens", 0)
    require(
        type(count) is int and 0 < count <= budget, "Invalid final completion_tokens"
    )
    require(type(cached) is int and 0 <= cached <= len(tokens), "Invalid cached_tokens")
    return {
        "prompt_tokens": len(tokens),
        "cached_tokens": cached,
        "output_tokens": count,
        "ttft": ttft,
        "latency": elapsed,
        "tpot": (elapsed - ttft) / (count - 1) if count > 1 else None,
    }


async def worker_pass(session, worker, url, tokens, budget):
    samples = []
    for n, prefix in PROFILES:
        async with session.post(url + "/flush_cache?timeout=30") as response:
            response.raise_for_status()
            require(
                (await response.text()).startswith("Cache flushed."),
                "Cache flush failed",
            )
        if prefix:
            prime = await request(session, url, tokens[:prefix], 1)
            require(prime["cached_tokens"] == 0, "Prefix prime was not cold")
        result = await request(session, url, tokens[:n], budget)
        require(
            (
                result["cached_tokens"] == 0
                if not prefix
                else result["cached_tokens"] >= prefix - 1
            ),
            f"Worker {worker}: unexpected cache reuse for profile {(n, prefix)}",
        )
        result.update(worker=worker, intended_prefix=prefix)
        result["new_prefill_tokens"] = n - result["cached_tokens"]
        result["reference_prediction"] = a6000_prefill_time(
            result["new_prefill_tokens"], n
        )
        result["ratio"] = result["ttft"] / result["reference_prediction"]
        samples.append(result)
    return samples


async def calibrate(args, tokens, provenance):
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600)
    ) as session:

        async def run_pass():
            batches = await asyncio.gather(
                *(
                    worker_pass(session, worker, url, tokens, args.max_new_tokens)
                    for worker, url in enumerate(args.worker_urls)
                )
            )
            return [sample for batch in batches for sample in batch]

        warmup = await run_pass()
        warmup_completed = dt.datetime.now(dt.timezone.utc).isoformat()
        samples = await run_pass()
    scale = statistics.median(sample["ratio"] for sample in samples)
    tpots = [sample["tpot"] for sample in samples if sample["tpot"] is not None]
    require(
        bool(tpots), "Measured generations need at least one multi-token completion"
    )
    for sample in samples:
        sample["scaled_prediction"] = scale * sample["reference_prediction"]
        sample["relative_prediction_error"] = (
            sample["scaled_prediction"] / sample["ttft"] - 1
        )
    result = {
        "schema_version": 1,
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": "median of measured TTFT / reference T(new_prefill_tokens, full_context_tokens)",
        "calibration_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        **provenance,
        "worker_urls": args.worker_urls,
        "profiles": [
            {"prompt_tokens": n, "intended_prefix": prefix} for n, prefix in PROFILES
        ],
        "max_new_tokens": args.max_new_tokens,
        "warmup_completed_at": warmup_completed,
        "warmup_samples": warmup,
        "samples": samples,
        "prefill_time_scale": scale,
        "initial_tpot": statistics.median(tpots),
    }
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "utc",
                    "workload_sha256",
                    "prefill_time_scale",
                    "initial_tpot",
                )
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--worker-urls", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=45)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("Output already exists; choose a new file")
    if not args.output.parent.is_dir():
        parser.error("Output parent directory does not exist")
    if args.max_new_tokens < 2:
        parser.error("max-new-tokens must be at least 2 to measure TPOT")
    args.worker_urls = [url.rstrip("/") for url in args.worker_urls]
    if len(set(args.worker_urls)) != len(args.worker_urls):
        parser.error("Worker URLs must be distinct")
    for url in args.worker_urls:
        parsed = urlsplit(url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.netloc
            or parsed.query
            or parsed.fragment
        ):
            parser.error(f"Expected an HTTP(S) worker base URL: {url}")
    try:
        tokens, provenance = load_prompt(args.workload)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    asyncio.run(calibrate(args, tokens, provenance))


if __name__ == "__main__":
    main()
