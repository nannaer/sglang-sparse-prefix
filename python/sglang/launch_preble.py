# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Preble frontend for a fixed pool of identical SGLang /generate workers."""

import argparse
import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from transformers import AutoTokenizer

from sglang.srt.routing.preble import PrebleScheduler


def create_app(worker_urls, tokenizer, scheduler):
    urls = [url.rstrip("/") for url in worker_urls]
    if len(urls) != scheduler.num_workers or len(set(urls)) != len(urls):
        raise ValueError("Worker URLs must be unique and match the scheduler pool")
    unavailable_until = [0.0] * len(urls)

    @asynccontextmanager
    async def lifespan(app):
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10800),
            connector=aiohttp.TCPConnector(limit=0),
        ) as session:
            app.state.session = session
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
                sum(n.active[w] for n in scheduler.root.children.values())
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
        if any(t >= len(tokenizer) for t in tokens):
            raise HTTPException(400, "Token ID outside model vocabulary")
        params = payload.get("sampling_params", {})
        if not isinstance(params, dict):
            raise HTTPException(400, "sampling_params must be an object")
        budget = params.get("max_new_tokens", 16)
        if type(budget) is not int or budget < 0:
            raise HTTPException(400, "max_new_tokens must be nonnegative")
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
        try:
            response = await session.post(
                url + "/generate", json=payload, headers=headers
            )
        except (aiohttp.ClientError, asyncio.TimeoutError):
            scheduler.finish(assignment)
            unavailable_until[assignment.worker] = time.monotonic() + 5
            raise HTTPException(502, "Worker connection failed")
        except BaseException:
            scheduler.finish(assignment)
            raise
        if response.status != 200:
            try:
                body = await response.read()
                return Response(
                    body, status_code=response.status, media_type="application/json"
                )
            finally:
                response.release()
                scheduler.finish(assignment)

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
                response.close()
                scheduler.finish(
                    assignment,
                    output_tokens if done else 0,
                    ttft if done else 0,
                    time.monotonic() - started,
                )
                if not done:
                    # Release backend resources on frontend cancellation as well.
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

                    task = asyncio.create_task(abort())
                    await asyncio.shield(task)

        routing_headers = {"X-Preble-Worker": str(assignment.worker)}
        if stream:
            return StreamingResponse(
                chunks(), media_type="text/event-stream", headers=routing_headers
            )
        last = None
        async for line in chunks():
            if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                last = json.loads(line[5:])
        if last is None:
            raise HTTPException(502, "Worker returned no generation")
        return JSONResponse(last, headers=routing_headers)

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
    parser.add_argument(
        "--prefill-tokens-per-second",
        type=float,
        required=True,
        help="Measured prefill rate for this model and worker hardware",
    )
    parser.add_argument(
        "--initial-tpot",
        type=float,
        required=True,
        help="Measured initial decode seconds per output token",
    )
    args = parser.parse_args()
    if args.prefill_tokens_per_second <= 0:
        parser.error("Prefill rate must be positive")
    scheduler = PrebleScheduler(
        len(args.worker_urls),
        window_seconds=args.window_seconds,
        prefill_time=lambda tokens, context: tokens / args.prefill_tokens_per_second,
        initial_tpot=args.initial_tpot,
        enable_rebalancing=not args.disable_rebalancing,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    uvicorn.run(
        create_app(args.worker_urls, tokenizer, scheduler),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
