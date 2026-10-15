# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Preble frontend for identical, cumulative-streaming SGLang /generate workers."""

import argparse
import asyncio
import json
import math
import time
import uuid
from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
from transformers import AutoTokenizer

from sglang.srt.routing.preble import PrebleScheduler
from sglang.srt.routing.preble_cost import a6000_prefill_time


def create_app(worker_urls, tokenizer, scheduler):
    urls = [url.rstrip("/") for url in worker_urls]
    if len(urls) != scheduler.num_workers or len(set(urls)) != len(urls):
        raise ValueError("Worker URLs must be unique and match the scheduler pool")
    vocab_size = len(tokenizer)
    unavailable_until = [0.0] * len(urls)

    @asynccontextmanager
    async def lifespan(app):
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10800),
            connector=aiohttp.TCPConnector(limit=0),
        ) as session:
            app.state.session = session

            async def check_worker(url):
                async with session.get(
                    url + "/server_info", timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    response.raise_for_status()
                    info = await response.json()
                if info.get("incremental_streaming_output", False):
                    raise ValueError(
                        f"Preble requires cumulative streaming output: {url}"
                    )

            await asyncio.gather(*(check_worker(url) for url in urls))
            yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok", "workers": urls}

    @app.get("/routing_stats")
    async def routing_stats():
        return {
            "scheduled": scheduler.scheduled,
            "migrations": scheduler.migrations,
            "replications": scheduler.replications,
            "nodes": len(scheduler.nodes),
            "active": [
                sum(n.ref_counter[w] for n in scheduler.root.children.values())
                for w in range(len(urls))
            ],
        }

    @app.post("/generate")
    async def generate(request: Request):
        try:
            payload = await request.json()
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(400, "Invalid JSON")
        if not isinstance(payload, dict):
            raise HTTPException(400, "Expected one generation object")
        if any(
            payload.get(k) is not None
            for k in (
                "image_data",
                "audio_data",
                "video_data",
                "input_embeds",
                "lora_path",
                "session_params",
            )
        ):
            raise HTTPException(
                400, "Preble supports stateless text requests to one model"
            )
        if any(payload.get(k) is not None for k in ("extra_key", "cache_salt")):
            raise HTTPException(
                400, "Preble does not support cache namespaces (extra_key/cache_salt)"
            )
        tokens = payload.get("input_ids")
        if tokens is None:
            text = payload.get("text")
            if not isinstance(text, str) or not text:
                raise HTTPException(
                    400, "Provide nonempty text or input_ids; batches are unsupported"
                )
            tokens = await asyncio.to_thread(tokenizer.encode, text)
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(type(t) is not int or t < 0 for t in tokens)
        ):
            raise HTTPException(400, "input_ids must be one nonempty token list")
        if any(t >= vocab_size for t in tokens):
            raise HTTPException(400, "Token ID outside model vocabulary")
        params = payload.get("sampling_params", {})
        if not isinstance(params, dict):
            raise HTTPException(400, "sampling_params must be an object")
        if type(params.get("n", 1)) is not int or params.get("n", 1) != 1:
            raise HTTPException(400, "Preble supports one sample per request (n=1)")
        budget = params.get("max_new_tokens", 16)
        if type(budget) is not int or budget < 0:
            raise HTTPException(400, "max_new_tokens must be nonnegative")
        # The reference frontend defaults to 16; current workers default to 128.
        payload["sampling_params"] = {**params, "max_new_tokens": budget}
        if type(payload.get("stream", False)) is not bool:
            raise HTTPException(400, "stream must be a boolean")
        available = [
            w for w, until in enumerate(unavailable_until) if until <= time.monotonic()
        ]
        if not available:
            raise HTTPException(503, "All workers temporarily unavailable")
        started = time.monotonic()
        assignment = scheduler.select(tokens, budget, available)
        url = urls[assignment.worker]
        stream = payload.get("stream", False)
        # Forward exactly the tokens used for routing; avoid independent retokenization.
        payload.pop("text", None)
        payload["input_ids"] = tokens
        payload["rid"] = uuid.uuid4().hex
        payload["stream"] = True
        headers = {}
        if "authorization" in request.headers:
            headers["Authorization"] = request.headers["authorization"]
        session = app.state.session
        response = None
        released = False

        async def abort():
            try:
                async with session.post(
                    url + "/abort_request",
                    json={"rid": payload["rid"]},
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as result:
                    await result.read()
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass

        async def release(done=False, output_tokens=0, ttft=0):
            nonlocal released
            if released:
                return
            released = True
            if response is not None:
                response.close()
            scheduler.finish(
                assignment,
                output_tokens if done else 0,
                ttft if done else 0,
                time.monotonic() - started,
            )
            if not done:
                await asyncio.shield(asyncio.create_task(abort()))

        async def with_disconnect(operation):
            # Nonstream HTTP handlers keep running after a socket disconnect.
            stopped = asyncio.Event()

            async def disconnected():
                while not stopped.is_set() and not await request.is_disconnected():
                    try:
                        await asyncio.wait_for(stopped.wait(), timeout=0.1)
                    except asyncio.TimeoutError:
                        pass

            pending = asyncio.ensure_future(operation)
            monitor = asyncio.create_task(disconnected())
            try:
                done, _ = await asyncio.wait(
                    (pending, monitor), return_when=asyncio.FIRST_COMPLETED
                )
                if monitor in done:
                    raise HTTPException(499, "Client disconnected")
                return await pending
            finally:
                stopped.set()
                if not pending.done():
                    pending.cancel()
                await asyncio.gather(pending, monitor, return_exceptions=True)

        async def open_response():
            nonlocal response
            response = await session.post(
                url + "/generate", json=payload, headers=headers
            )

        try:
            await with_disconnect(open_response())
        except (aiohttp.ClientError, asyncio.TimeoutError):
            await release()
            unavailable_until[assignment.worker] = time.monotonic() + 5
            raise HTTPException(502, "Worker connection failed")
        except BaseException:
            await release()
            raise
        if response.status != 200:
            try:
                body = await with_disconnect(response.read())
                return Response(
                    body, status_code=response.status, media_type="application/json"
                )
            finally:
                await release(done=True)

        async def chunks():
            ttft = 0.0
            output_tokens = 0
            done = False
            try:
                async for line in response.content:
                    if line.startswith(b"data:"):
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            done = True
                        elif data:
                            obj = json.loads(data)
                            count = obj.get("meta_info", {}).get("completion_tokens", 0)
                            if count > 0 and ttft == 0:
                                ttft = time.monotonic() - started
                            output_tokens = count
                    yield line
            finally:
                await release(done, output_tokens, ttft)

        routing_headers = {"X-Preble-Worker": str(assignment.worker)}
        if stream:
            return StreamingResponse(
                chunks(),
                media_type="text/event-stream",
                headers=routing_headers,
                background=BackgroundTask(release),
            )

        async def collect():
            last = None
            async for line in chunks():
                if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                    last = json.loads(line[5:])
            return last

        try:
            last = await with_disconnect(collect())
        finally:
            await release()
        if last is None:
            raise HTTPException(502, "Worker returned no generation")
        error = last.get("error")
        reason = last.get("meta_info", {}).get("finish_reason") or {}
        status = 200
        if error is not None or reason.get("type") == "abort":
            code = error.get("code") if isinstance(error, dict) else None
            code = code or reason.get("status_code")
            status = code if type(code) is int and 400 <= code < 600 else 502
        return JSONResponse(last, status_code=status, headers=routing_headers)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-urls", nargs="+", required=True)
    parser.add_argument(
        "--model-path",
        required=True,
        help="The identical tokenizer used by every worker",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--window-seconds", type=float, default=180)
    parser.add_argument("--disable-rebalancing", action="store_true")
    parser.add_argument("--disable-eviction", action="store_true")
    parser.add_argument(
        "--max-tokens-per-worker",
        nargs="+",
        type=int,
        help="Virtual KV capacities in tokens; reference default is 198516 per worker",
    )
    prefill_model = parser.add_mutually_exclusive_group()
    prefill_model.add_argument(
        "--prefill-tokens-per-second",
        type=float,
        help="Optional linear prefill model; otherwise use the reference A6000 model",
    )
    prefill_model.add_argument(
        "--prefill-time-scale",
        type=float,
        default=1.0,
        help="Hardware calibration scale for the reference two-argument prefill model",
    )
    parser.add_argument(
        "--initial-tpot",
        type=float,
        default=0.15,
        help="Measured initial decode seconds per output token",
    )
    args = parser.parse_args()
    if args.prefill_tokens_per_second is not None and (
        not math.isfinite(args.prefill_tokens_per_second)
        or args.prefill_tokens_per_second <= 0
    ):
        parser.error("Prefill rate must be finite and positive")
    if not math.isfinite(args.prefill_time_scale) or args.prefill_time_scale <= 0:
        parser.error("Prefill time scale must be finite and positive")
    if not math.isfinite(args.initial_tpot) or args.initial_tpot <= 0:
        parser.error("Initial TPOT must be finite and positive")
    prefill_kwargs = {}
    if args.prefill_tokens_per_second is not None:
        prefill_kwargs["prefill_time"] = (
            lambda tokens, context: tokens / args.prefill_tokens_per_second
        )
    elif args.prefill_time_scale != 1:
        prefill_kwargs["prefill_time"] = (
            lambda tokens, context: args.prefill_time_scale
            * a6000_prefill_time(tokens, context)
        )
    scheduler = PrebleScheduler(
        len(args.worker_urls),
        window_seconds=args.window_seconds,
        initial_tpot=args.initial_tpot,
        enable_rebalancing=not args.disable_rebalancing,
        enable_eviction=not args.disable_eviction,
        max_tokens_gpu=args.max_tokens_per_worker,
        **prefill_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    uvicorn.run(
        create_app(args.worker_urls, tokenizer, scheduler),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
