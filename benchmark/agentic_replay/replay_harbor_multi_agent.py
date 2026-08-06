#!/usr/bin/env python3
"""Replay one Harbor multi-agent serving trace directly against SGLang."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


TRACE_FORMAT = "harbor_model_relay_turn_v2"


@dataclass(frozen=True)
class AgentSequence:
    agent_id: str
    rows: list[dict[str, Any]]
    start_offset_seconds: float


@dataclass(frozen=True)
class MultiAgentTrace:
    trace_dir: Path
    traj_id: str
    rows: list[dict[str, Any]]
    agents: list[AgentSequence]


class CacheBranchRegistry:
    """Track replay-created Radix branches across concurrent agents."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._branches: list[list[int]] = []

    def add(self, tokens: list[int]) -> None:
        with self._lock:
            self._branches.append(list(tokens))

    def shape_and_reserve(
        self,
        input_ids: list[int],
        row: dict[str, Any],
        previous_request_tokens: list[int] | None,
    ) -> tuple[list[int], int]:
        with self._lock:
            shaped, adjustments = shape_recorded_cache_prefix(
                input_ids,
                row,
                previous_request_tokens,
                cached_branches=self._branches,
            )
            # Reserve the branch before sending so concurrently prepared requests
            # cannot accidentally choose the same cache-shaping boundary.
            self._branches.append(list(shaped))
        return shaped, adjustments


def common_prefix_length(left: list[int], right: list[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def _resolve_trace_dir(source: Path) -> Path:
    source = source.resolve()
    if source.is_file():
        if source.name != "model_relay_trace.jsonl":
            raise ValueError(f"expected model_relay_trace.jsonl, got {source}")
        return source.parent
    if (source / "model_relay_trace.jsonl").is_file():
        return source
    if (source / "agent" / "model_relay_trace.jsonl").is_file():
        return source / "agent"
    raise ValueError(f"no model_relay_trace.jsonl under {source}")


def load_trace(source: Path) -> MultiAgentTrace:
    trace_dir = _resolve_trace_dir(source)
    relay_path = trace_dir / "model_relay_trace.jsonl"
    multi_agent_path = trace_dir / "multi_agent_trace.json"
    if not multi_agent_path.is_file():
        raise ValueError(f"missing multi-agent metadata: {multi_agent_path}")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        relay_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        status_code = int(row.get("status_code", 200) or 0)
        if row.get("trace_format") != TRACE_FORMAT or not 200 <= status_code < 300:
            continue
        if not row.get("token_ids_complete"):
            raise ValueError(f"{relay_path}:{line_number}: token IDs are incomplete")
        input_ids = row.get("input_ids")
        output_ids = row.get("output_ids")
        if not isinstance(input_ids, list) or not isinstance(output_ids, list):
            raise ValueError(f"{relay_path}:{line_number}: token IDs are missing")
        if len(input_ids) != int(row["input_tokens"]):
            raise ValueError(f"{relay_path}:{line_number}: input length mismatch")
        if len(output_ids) != int(row["output_tokens"]):
            raise ValueError(f"{relay_path}:{line_number}: output length mismatch")
        rows.append(row)
    rows.sort(key=lambda row: int(row["turn"]))
    if not rows:
        raise ValueError(f"no successful {TRACE_FORMAT} rows in {relay_path}")

    metadata = json.loads(multi_agent_path.read_text(encoding="utf-8"))
    calls = metadata.get("model_calls")
    if not isinstance(calls, list):
        raise ValueError(f"{multi_agent_path}: model_calls must be a list")
    by_request_id = {
        str(call["request_id"]): str(call["agent_id"])
        for call in calls
        if call.get("request_id") is not None and call.get("agent_id") is not None
    }
    by_turn = {
        int(call["turn"]): str(call["agent_id"])
        for call in calls
        if call.get("turn") is not None and call.get("agent_id") is not None
    }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        agent_id = by_request_id.get(str(row.get("request_id")))
        if agent_id is None:
            agent_id = by_turn.get(int(row["turn"]))
        if agent_id is None:
            raise ValueError(f"no agent mapping for global turn {row['turn']}")
        row["agent_id"] = agent_id
        grouped[agent_id].append(row)

    earliest_start = min(float(row["request_started_at"]) for row in rows)
    agents = []
    for agent_id, agent_rows in grouped.items():
        agent_rows.sort(key=lambda row: float(row["request_started_at"]))
        for agent_turn, row in enumerate(agent_rows):
            row["agent_turn"] = agent_turn
        agents.append(
            AgentSequence(
                agent_id=agent_id,
                rows=agent_rows,
                start_offset_seconds=max(
                    0.0,
                    float(agent_rows[0]["request_started_at"]) - earliest_start,
                ),
            )
        )
    agents.sort(key=lambda agent: agent.start_offset_seconds)

    traj_ids = {str(row.get("traj_id")) for row in rows}
    if len(traj_ids) != 1 or "None" in traj_ids:
        raise ValueError(f"expected one traj_id, got {traj_ids}")
    return MultiAgentTrace(trace_dir, traj_ids.pop(), rows, agents)


def rewrite_replayed_outputs(
    rows: list[dict[str, Any]],
    agent_turn: int,
    generated_outputs: list[list[int]],
) -> tuple[list[int], list[dict[str, int]]]:
    recorded_input = [int(token) for token in rows[agent_turn]["input_ids"]]
    replay_input = recorded_input.copy()
    spans: list[dict[str, int]] = []
    for source_agent_turn, generated_output in enumerate(generated_outputs):
        source_row = rows[source_agent_turn]
        source_input = [int(token) for token in source_row["input_ids"]]
        if recorded_input[: len(source_input)] != source_input:
            continue
        source_output = [int(token) for token in source_row["output_ids"]]
        start = len(source_input)
        retained = common_prefix_length(
            source_output,
            recorded_input[start : start + len(source_output)],
        )
        if retained == 0:
            continue
        if len(generated_output) < retained:
            raise ValueError(
                f"agent turn {agent_turn}: generated output {source_agent_turn} "
                f"is shorter than retained span {retained}"
            )
        replay_input[start : start + retained] = generated_output[:retained]
        spans.append(
            {
                "source_global_turn": int(source_row["turn"]),
                "source_agent_turn": source_agent_turn,
                "start": start,
                "length": retained,
            }
        )
    return replay_input, spans


def shape_recorded_cache_prefix(
    input_ids: list[int],
    row: dict[str, Any],
    previous_request_tokens: list[int] | None,
    *,
    cached_branches: list[list[int]] | None = None,
) -> tuple[list[int], int]:
    target = row.get("recorded_cached_tokens")
    if previous_request_tokens is None or not isinstance(target, int):
        return input_ids, 0
    if target < 0 or target > min(len(input_ids), len(previous_request_tokens)):
        raise ValueError(
            f"global turn {row['turn']}: invalid cache target {target} for "
            f"input={len(input_ids)}, previous={len(previous_request_tokens)}"
        )
    shaped = list(input_ids)
    changed = sum(
        left != right
        for left, right in zip(shaped[:target], previous_request_tokens[:target])
    )
    shaped[:target] = previous_request_tokens[:target]
    if target < len(shaped):
        branches = [previous_request_tokens, *(cached_branches or [])]
        forbidden = {
            int(branch[target])
            for branch in branches
            if branch is not None
            and target < len(branch)
            and branch[:target] == shaped[:target]
        }
        if shaped[target] in forbidden:
            replacement = next(
                (int(token) for token in input_ids if int(token) not in forbidden),
                None,
            )
            if replacement is None:
                raise ValueError(
                    f"global turn {row['turn']}: no valid token can break cache "
                    f"at target {target}"
                )
            shaped[target] = replacement
            changed += 1
    return shaped, changed


def build_cache_warmup(row: dict[str, Any]) -> list[int] | None:
    target = row.get("recorded_cached_tokens")
    input_ids = row.get("input_ids")
    if not isinstance(target, int) or target <= 0 or not isinstance(input_ids, list):
        return None
    if target > len(input_ids):
        raise ValueError(
            f"global turn {row['turn']}: warmup target {target} exceeds input length"
        )
    recorded_next = int(input_ids[target]) if target < len(input_ids) else None
    break_token = next(
        token for token in range(4) if recorded_next is None or token != recorded_next
    )
    return [*[int(token) for token in input_ids[:target]], break_token]


def _generate_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return f"{root}/generate"


def flush_cache(base_url: str, timeout: float) -> None:
    url = _generate_url(base_url).removesuffix("/generate") + "/flush_cache"
    response = requests.post(url, timeout=timeout)
    response.raise_for_status()


def extract_output_ids(payload: dict[str, Any]) -> list[int]:
    output_ids = payload.get("output_ids")
    if isinstance(output_ids, list):
        return [int(token) for token in output_ids]
    logprobs = (payload.get("meta_info") or {}).get("output_token_logprobs")
    if not isinstance(logprobs, list):
        raise ValueError("SGLang response has no output token IDs")
    result = []
    for item in logprobs:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            result.append(int(item[1]))
        elif isinstance(item, dict) and item.get("token_id") is not None:
            result.append(int(item["token_id"]))
        else:
            raise ValueError(f"unexpected output logprob entry: {item!r}")
    return result


def call_generate(
    session: requests.Session,
    *,
    base_url: str,
    request_id: str,
    input_ids: list[int],
    output_tokens: int,
    timeout: float,
) -> tuple[list[int], dict[str, Any], float, float]:
    started_at = time.time()
    response = session.post(
        _generate_url(base_url),
        json={
            "rid": request_id,
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": output_tokens,
                "min_new_tokens": output_tokens,
                "temperature": 0,
                "ignore_eos": True,
                "stop": None,
                "stop_token_ids": None,
            },
            "return_logprob": True,
            "top_logprobs_num": 0,
            "return_text_in_logprobs": False,
            "stream": False,
            "log_metrics": True,
        },
        timeout=timeout,
    )
    finished_at = time.time()
    response.raise_for_status()
    payload = response.json()
    return extract_output_ids(payload), payload.get("meta_info") or {}, started_at, finished_at


def warm_first_request_prefixes(
    trace: MultiAgentTrace,
    *,
    base_url: str,
    timeout: float,
    cache_branches: CacheBranchRegistry,
) -> list[dict[str, Any]]:
    results = []
    with requests.Session() as session:
        session.trust_env = False
        for agent in trace.agents:
            row = agent.rows[0]
            warmup_input = build_cache_warmup(row)
            if warmup_input is None:
                continue
            output_ids, meta, started_at, finished_at = call_generate(
                session,
                base_url=base_url,
                request_id=f"harbor-ma-warm-{trace.traj_id}-{agent.agent_id}",
                input_ids=warmup_input,
                output_tokens=1,
                timeout=timeout,
            )
            cache_branches.add([*warmup_input, *output_ids])
            results.append(
                {
                    "agent_id": agent.agent_id,
                    "target_cached_tokens": int(row["recorded_cached_tokens"]),
                    "warmup_input_tokens": len(warmup_input),
                    "warmup_cached_tokens": int(meta.get("cached_tokens", 0) or 0),
                    "latency_seconds": finished_at - started_at,
                }
            )
    return results


def replay_agent(
    trace: MultiAgentTrace,
    agent: AgentSequence,
    *,
    base_url: str,
    timeout: float,
    sleep_delays: bool,
    time_scale: float,
    shape_cache: bool,
    replay_epoch: float,
    cache_branches: CacheBranchRegistry,
    output_file: Any,
    output_lock: threading.Lock,
) -> list[dict[str, Any]]:
    initial_delay = agent.start_offset_seconds * time_scale
    if sleep_delays and initial_delay > 0:
        time.sleep(initial_delay)

    generated_outputs: list[list[int]] = []
    replay_requests: list[tuple[list[int], list[int]]] = []
    results = []
    with requests.Session() as session:
        session.trust_env = False
        for agent_turn, row in enumerate(agent.rows):
            inter_request_delay = 0.0
            if agent_turn > 0:
                previous = agent.rows[agent_turn - 1]
                inter_request_delay = max(
                    0.0,
                    float(row["request_started_at"])
                    - float(previous["request_finished_at"]),
                )
                if sleep_delays and inter_request_delay > 0:
                    time.sleep(inter_request_delay * time_scale)

            input_ids, rewrite_spans = rewrite_replayed_outputs(
                agent.rows, agent_turn, generated_outputs
            )
            unshaped_input = input_ids
            cache_shape_adjustments = 0
            previous_request_tokens = None
            if replay_requests:
                previous_request_tokens = [*replay_requests[-1][0], *replay_requests[-1][1]]
            if shape_cache:
                input_ids, cache_shape_adjustments = cache_branches.shape_and_reserve(
                    input_ids, row, previous_request_tokens
                )
            else:
                cache_branches.add(input_ids)

            target_output_tokens = len(row["output_ids"])
            output_ids, meta, started_at, finished_at = call_generate(
                session,
                base_url=base_url,
                request_id=(
                    f"harbor-ma-replay-{trace.traj_id}-{agent.agent_id}-"
                    f"{agent_turn:04d}"
                ),
                input_ids=input_ids,
                output_tokens=target_output_tokens,
                timeout=timeout,
            )
            if len(output_ids) != target_output_tokens:
                raise ValueError(
                    f"{agent.agent_id} turn {agent_turn}: output length "
                    f"{len(output_ids)} != {target_output_tokens}"
                )

            recorded_cached = int(row.get("recorded_cached_tokens", 0) or 0)
            replay_cached = int(meta.get("cached_tokens", 0) or 0)
            source_lcp = 0
            replay_lcp = 0
            if agent_turn > 0:
                previous_source = [
                    *agent.rows[agent_turn - 1]["input_ids"],
                    *agent.rows[agent_turn - 1]["output_ids"],
                ]
                source_lcp = common_prefix_length(previous_source, row["input_ids"])
                replay_lcp = common_prefix_length(previous_request_tokens or [], input_ids)
            result = {
                "trace_format": "harbor_multi_agent_replay_turn_v1",
                "traj_id": trace.traj_id,
                "agent_id": agent.agent_id,
                "agent_turn": agent_turn,
                "global_turn": int(row["turn"]),
                "input_tokens": len(input_ids),
                "recorded_input_tokens": len(row["input_ids"]),
                "output_tokens": len(output_ids),
                "recorded_output_tokens": target_output_tokens,
                "recorded_cached_tokens": recorded_cached,
                "replay_cached_tokens": replay_cached,
                "cached_token_delta": replay_cached - recorded_cached,
                "recorded_cache_hit_rate": recorded_cached / len(row["input_ids"]),
                "replay_cache_hit_rate": replay_cached / len(input_ids),
                "source_same_agent_lcp": source_lcp,
                "replay_same_agent_lcp": replay_lcp,
                "cache_shape_adjustments": cache_shape_adjustments,
                "rewrite_spans": rewrite_spans,
                "source_agent_start_offset_seconds": agent.start_offset_seconds,
                "source_inter_request_delay_seconds": inter_request_delay,
                "replay_request_start_offset_seconds": started_at - replay_epoch,
                "request_started_at": started_at,
                "request_finished_at": finished_at,
                "client_latency_seconds": finished_at - started_at,
                "unshaped_input_sha256": hashlib.sha256(
                    json.dumps(unshaped_input, separators=(",", ":")).encode()
                ).hexdigest(),
                "shaped_input_sha256": hashlib.sha256(
                    json.dumps(input_ids, separators=(",", ":")).encode()
                ).hexdigest(),
            }
            with output_lock:
                output_file.write(json.dumps(result, separators=(",", ":")) + "\n")
                output_file.flush()
            generated_outputs.append(output_ids)
            replay_requests.append((input_ids, output_ids))
            cache_branches.add([*input_ids, *output_ids])
            results.append(result)
    return results


def replay(args: argparse.Namespace) -> dict[str, Any]:
    trace = load_trace(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    turns_path = args.output_dir / "replay_turns.jsonl"
    summary_path = args.output_dir / "summary.json"

    if args.flush_cache:
        flush_cache(args.base_url, args.request_timeout)
    cache_branches = CacheBranchRegistry()
    warmups = []
    if not args.no_cache_shaping:
        warmups = warm_first_request_prefixes(
            trace,
            base_url=args.base_url,
            timeout=args.request_timeout,
            cache_branches=cache_branches,
        )

    output_lock = threading.Lock()
    replay_epoch = time.time()
    results: list[dict[str, Any]] = []
    errors = []
    with turns_path.open("w", encoding="utf-8") as output_file:
        with ThreadPoolExecutor(max_workers=len(trace.agents)) as pool:
            futures = {
                pool.submit(
                    replay_agent,
                    trace,
                    agent,
                    base_url=args.base_url,
                    timeout=args.request_timeout,
                    sleep_delays=not args.no_sleep,
                    time_scale=args.time_scale,
                    shape_cache=not args.no_cache_shaping,
                    replay_epoch=replay_epoch,
                    cache_branches=cache_branches,
                    output_file=output_file,
                    output_lock=output_lock,
                ): agent
                for agent in trace.agents
            }
            for future in as_completed(futures):
                agent = futures[future]
                try:
                    results.extend(future.result())
                except Exception as error:
                    errors.append({"agent_id": agent.agent_id, "error": repr(error)})
    results.sort(key=lambda row: int(row["global_turn"]))
    cache_deltas = [int(row["cached_token_delta"]) for row in results]
    summary = {
        "source": str(trace.trace_dir),
        "traj_id": trace.traj_id,
        "base_url": args.base_url,
        "agents": len(trace.agents),
        "source_requests": len(trace.rows),
        "completed_requests": len(results),
        "errors": errors,
        "duration_seconds": time.time() - replay_epoch,
        "sleep_delays": not args.no_sleep,
        "time_scale": args.time_scale,
        "cache_shaping": not args.no_cache_shaping,
        "cache_warmups": warmups,
        "input_lengths_match": all(
            row["input_tokens"] == row["recorded_input_tokens"] for row in results
        ),
        "output_lengths_match": all(
            row["output_tokens"] == row["recorded_output_tokens"] for row in results
        ),
        "cache_exact_matches": sum(delta == 0 for delta in cache_deltas),
        "cache_compared_requests": len(cache_deltas),
        "cache_all_exact": bool(cache_deltas) and all(delta == 0 for delta in cache_deltas),
        "cache_max_abs_delta_tokens": max(map(abs, cache_deltas), default=None),
        "turns": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one Harbor multi-agent token trace against SGLang."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--flush-cache", action="store_true")
    parser.add_argument("--no-sleep", action="store_true")
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--no-cache-shaping", action="store_true")
    args = parser.parse_args()
    if args.time_scale < 0:
        parser.error("--time-scale must be non-negative")
    return args


def main() -> None:
    print(json.dumps(replay(parse_args()), indent=2))


if __name__ == "__main__":
    main()
