from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


_TRACER = None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _safe_attn_tp_rank() -> int:
    try:
        from sglang.srt.runtime_context import get_parallel

        return int(get_parallel().attn_tp_rank)
    except Exception:
        return int(os.getenv("SGLANG_TP_RANK", os.getenv("LOCAL_RANK", "0")))


def _seq_lens_to_list(forward_batch: Any) -> list[int]:
    seq_lens = getattr(forward_batch, "seq_lens_cpu", None)
    if seq_lens is None:
        seq_lens = getattr(forward_batch, "seq_lens", None)
    if seq_lens is None:
        return []
    if isinstance(seq_lens, torch.Tensor):
        return [int(x) for x in seq_lens.detach().to("cpu").tolist()]
    return [int(x) for x in seq_lens]


def _req_pool_indices_to_list(forward_batch: Any) -> list[int]:
    req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
    if req_pool_indices is None:
        return []
    if isinstance(req_pool_indices, torch.Tensor):
        return [int(x) for x in req_pool_indices.detach().to("cpu").tolist()]
    return [int(x) for x in req_pool_indices]


def _percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


class HiSparseTopkOverlapTracer:
    def __init__(self) -> None:
        trace_dir = os.getenv("SGLANG_HISPARSE_TOPK_TRACE_DIR", "")
        self.enabled = bool(trace_dir)
        self.shared_prefix_tokens = _env_int(
            "SGLANG_HISPARSE_TOPK_TRACE_SHARED_PREFIX_TOKENS", 0
        )
        self.max_steps = _env_int("SGLANG_HISPARSE_TOPK_TRACE_MAX_STEPS", 128)
        self.target_rank = _env_int("SGLANG_HISPARSE_TOPK_TRACE_RANK", 0)
        self.rank = _safe_attn_tp_rank()
        self.layer_steps: dict[int, int] = defaultdict(int)
        self._file = None

        if not self.enabled or self.rank != self.target_rank:
            self.enabled = False
            return
        if self.shared_prefix_tokens <= 0:
            raise ValueError(
                "SGLANG_HISPARSE_TOPK_TRACE_SHARED_PREFIX_TOKENS must be set "
                "when SGLANG_HISPARSE_TOPK_TRACE_DIR is enabled"
            )

        out_dir = Path(trace_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / f"hisparse_topk_overlap.tp{self.rank}.jsonl"
        self._file = self.path.open("w", encoding="utf-8", buffering=1)
        metadata = {
            "record_type": "metadata",
            "schema_version": 1,
            "created_at_unix": time.time(),
            "shared_prefix_tokens": self.shared_prefix_tokens,
            "max_steps": self.max_steps,
            "rank": self.rank,
            "target_rank": self.target_rank,
        }
        self._file.write(json.dumps(metadata, sort_keys=True) + "\n")

    def trace(self, layer_id: int, topk_indices: torch.Tensor, forward_batch: Any) -> None:
        if not self.enabled:
            return
        forward_mode = getattr(forward_batch, "forward_mode", None)
        if forward_mode is None or not forward_mode.is_decode_or_idle():
            return

        decode_step = self.layer_steps[layer_id]
        self.layer_steps[layer_id] += 1
        if decode_step >= self.max_steps:
            return

        row = self._summarize(layer_id, decode_step, topk_indices, forward_batch)
        self._file.write(json.dumps(row, sort_keys=True) + "\n")

    def _summarize(
        self, layer_id: int, decode_step: int, topk_indices: torch.Tensor, forward_batch: Any
    ) -> dict[str, Any]:
        arr = topk_indices.detach().to(device="cpu", dtype=torch.int32).numpy()
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        valid_mask = arr >= 0
        prefix_mask = valid_mask & (arr < self.shared_prefix_tokens)
        valid_counts = valid_mask.sum(axis=1).astype(np.int64)
        prefix_counts = prefix_mask.sum(axis=1).astype(np.int64)

        prefix_sets: list[set[int]] = []
        intra_unique_ratios: list[float] = []
        for row, row_mask, prefix_count in zip(arr, prefix_mask, prefix_counts):
            values = row[row_mask]
            value_set = set(int(x) for x in values.tolist())
            prefix_sets.append(value_set)
            if prefix_count > 0:
                intra_unique_ratios.append(len(value_set) / float(prefix_count))

        flat_prefix = arr[prefix_mask]
        sum_prefix_topk = int(prefix_counts.sum())
        unique_prefix_tokens = 0
        duplicate_fraction = None
        redundancy_factor = None
        shared_by_ge2_ratio = None
        shared_by_ge4_ratio = None
        shared_by_ge8_ratio = None
        max_request_coverage = 0
        mean_request_coverage = None
        if sum_prefix_topk > 0:
            unique_values, counts = np.unique(flat_prefix, return_counts=True)
            unique_prefix_tokens = int(unique_values.size)
            duplicate_fraction = 1.0 - unique_prefix_tokens / float(sum_prefix_topk)
            redundancy_factor = sum_prefix_topk / float(unique_prefix_tokens)
            shared_by_ge2_ratio = float(np.mean(counts >= 2))
            shared_by_ge4_ratio = float(np.mean(counts >= 4))
            shared_by_ge8_ratio = float(np.mean(counts >= 8))
            max_request_coverage = int(counts.max())
            mean_request_coverage = float(counts.mean())

        pairwise_jaccard: list[float] = []
        pairwise_overlap: list[float] = []
        for i in range(len(prefix_sets)):
            left = prefix_sets[i]
            if not left:
                continue
            for j in range(i + 1, len(prefix_sets)):
                right = prefix_sets[j]
                if not right:
                    continue
                inter = len(left & right)
                union = len(left | right)
                pairwise_jaccard.append(inter / float(union) if union else 0.0)
                pairwise_overlap.append(inter / float(min(len(left), len(right))))

        seq_lens = _seq_lens_to_list(forward_batch)
        req_pool_indices = _req_pool_indices_to_list(forward_batch)
        total_valid_topk = int(valid_counts.sum())
        return {
            "record_type": "topk_overlap",
            "schema_version": 1,
            "layer_id": int(layer_id),
            "decode_step": int(decode_step),
            "batch_size": int(arr.shape[0]),
            "topk": int(arr.shape[1]),
            "shared_prefix_tokens": int(self.shared_prefix_tokens),
            "seq_len_min": min(seq_lens) if seq_lens else None,
            "seq_len_max": max(seq_lens) if seq_lens else None,
            "req_pool_indices": req_pool_indices,
            "total_valid_topk": total_valid_topk,
            "sum_prefix_topk": sum_prefix_topk,
            "unique_prefix_tokens": unique_prefix_tokens,
            "prefix_topk_ratio_mean": float(
                np.mean(prefix_counts / np.maximum(valid_counts, 1))
            ),
            "prefix_topk_count_mean": float(np.mean(prefix_counts)),
            "intra_request_unique_ratio_mean": (
                float(np.mean(intra_unique_ratios)) if intra_unique_ratios else None
            ),
            "duplicate_fraction": duplicate_fraction,
            "redundancy_factor": redundancy_factor,
            "shared_by_ge2_ratio": shared_by_ge2_ratio,
            "shared_by_ge4_ratio": shared_by_ge4_ratio,
            "shared_by_ge8_ratio": shared_by_ge8_ratio,
            "max_request_coverage": max_request_coverage,
            "mean_request_coverage": mean_request_coverage,
            "pairwise_jaccard_mean": (
                float(np.mean(pairwise_jaccard)) if pairwise_jaccard else None
            ),
            "pairwise_jaccard_p50": _percentile(pairwise_jaccard, 50),
            "pairwise_jaccard_p90": _percentile(pairwise_jaccard, 90),
            "pairwise_overlap_mean": (
                float(np.mean(pairwise_overlap)) if pairwise_overlap else None
            ),
            "pairwise_overlap_p50": _percentile(pairwise_overlap, 50),
            "pairwise_overlap_p90": _percentile(pairwise_overlap, 90),
        }


def maybe_trace_hisparse_topk_overlap(
    layer_id: int, topk_indices: Optional[torch.Tensor], forward_batch: Any
) -> None:
    global _TRACER
    if topk_indices is None:
        return
    if _TRACER is None:
        _TRACER = HiSparseTopkOverlapTracer()
    _TRACER.trace(layer_id, topk_indices, forward_batch)
