#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METRICS = [
    "prefix_topk_ratio_mean",
    "prefix_topk_count_mean",
    "pairwise_jaccard_mean",
    "pairwise_overlap_mean",
    "sum_prefix_topk",
    "unique_prefix_tokens",
    "duplicate_fraction",
    "redundancy_factor",
    "shared_by_ge2_ratio",
    "shared_by_ge4_ratio",
    "shared_by_ge8_ratio",
    "mean_request_coverage",
    "max_request_coverage",
]


def _iter_trace_rows(paths: list[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("record_type") == "topk_overlap":
                    row["_trace_path"] = str(path)
                    yield row


def _trace_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for value in args.trace:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.glob("hisparse_topk_overlap.tp*.jsonl")))
        else:
            paths.append(path)
    paths = [path for path in paths if path.exists()]
    if not paths:
        raise FileNotFoundError("No trace JSONL files found")
    return paths


def _finite(values: list[Any]) -> list[float]:
    out = []
    for value in values:
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value):
            out.append(value)
    return out


def _summary(values: list[float]) -> dict[str, float | int | None]:
    vals = _finite(values)
    if not vals:
        return {"mean": None, "p10": None, "p50": None, "p90": None, "count": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "count": int(arr.size),
    }


def _mean(values: list[Any]) -> float | None:
    vals = _finite(values)
    if not vals:
        return None
    return float(np.asarray(vals, dtype=np.float64).mean())


def _format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    value = float(value)
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.3f}"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _print_metric_table(title: str, metric: str, summary: dict[str, Any]) -> None:
    print(f"\n{title}")
    print("| group | mean | p10 | p50 | p90 |")
    print("|---|---:|---:|---:|---:|")
    row = summary[metric]
    print(
        "| full_indexer | "
        + " | ".join(
            _format_value(row[key]) for key in ("mean", "p10", "p50", "p90")
        )
        + " |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize HiSparse cross-request prefix topK overlap trace."
    )
    parser.add_argument(
        "--trace",
        action="append",
        required=True,
        help="Trace JSONL file or directory. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="hisparse_topk_overlap")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument(
        "--min-decode-step",
        type=int,
        default=1,
        help="Drop decode steps smaller than this value. Default excludes step 0.",
    )
    args = parser.parse_args()

    paths = _trace_paths(args)
    rows = []
    for row in _iter_trace_rows(paths):
        if args.batch_size and int(row["batch_size"]) != args.batch_size:
            continue
        if int(row["decode_step"]) < args.min_decode_step:
            continue
        rows.append(row)
    if not rows:
        raise RuntimeError(
            "No rows left after filtering. Check --batch-size and --min-decode-step."
        )

    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_layer[int(row["layer_id"])].append(row)

    layer_rows: list[dict[str, Any]] = []
    for layer_id in sorted(by_layer):
        layer_input = by_layer[layer_id]
        out: dict[str, Any] = {
            "layer_id": layer_id,
            "num_steps": len(layer_input),
            "batch_size_min": min(int(row["batch_size"]) for row in layer_input),
            "batch_size_max": max(int(row["batch_size"]) for row in layer_input),
            "seq_len_min": min(
                int(row["seq_len_min"])
                for row in layer_input
                if row.get("seq_len_min") is not None
            ),
            "seq_len_max": max(
                int(row["seq_len_max"])
                for row in layer_input
                if row.get("seq_len_max") is not None
            ),
        }
        for metric in METRICS:
            out[metric] = _mean([row.get(metric) for row in layer_input])
        layer_rows.append(out)

    summary = {
        metric: _summary([row.get(metric) for row in layer_rows])
        for metric in METRICS
    }
    summary_meta = {
        "trace_files": [str(path) for path in paths],
        "row_count": len(rows),
        "layer_count": len(layer_rows),
        "batch_size_filter": args.batch_size,
        "min_decode_step": args.min_decode_step,
        "decode_step_min": min(int(row["decode_step"]) for row in rows),
        "decode_step_max": max(int(row["decode_step"]) for row in rows),
        "batch_size_min": min(int(row["batch_size"]) for row in rows),
        "batch_size_max": max(int(row["batch_size"]) for row in rows),
        "seq_len_min": min(
            int(row["seq_len_min"]) for row in rows if row.get("seq_len_min") is not None
        ),
        "seq_len_max": max(
            int(row["seq_len_max"]) for row in rows if row.get("seq_len_max") is not None
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_csv = output_dir / f"{args.output_prefix}.layers.csv"
    summary_json = output_dir / f"{args.output_prefix}.summary.json"
    _write_csv(
        layer_csv,
        layer_rows,
        [
            "layer_id",
            "num_steps",
            "batch_size_min",
            "batch_size_max",
            "seq_len_min",
            "seq_len_max",
            *METRICS,
        ],
    )
    summary_json.write_text(
        json.dumps({"metadata": summary_meta, "metrics": summary}, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary_meta, indent=2))
    _print_metric_table("Prefix TopK Ratio", "prefix_topk_ratio_mean", summary)
    _print_metric_table("Pairwise Jaccard", "pairwise_jaccard_mean", summary)
    _print_metric_table("Pairwise Overlap", "pairwise_overlap_mean", summary)
    _print_metric_table("Duplicate Fraction", "duplicate_fraction", summary)
    _print_metric_table("Redundancy Factor", "redundancy_factor", summary)
    _print_metric_table("Unique Prefix Tokens", "unique_prefix_tokens", summary)
    _print_metric_table("Sum Prefix TopK", "sum_prefix_topk", summary)
    _print_metric_table("Shared By >=2 Requests", "shared_by_ge2_ratio", summary)
    _print_metric_table("Shared By >=4 Requests", "shared_by_ge4_ratio", summary)
    _print_metric_table("Mean Request Coverage", "mean_request_coverage", summary)
    print(f"\nlayer_csv={layer_csv}")
    print(f"summary_json={summary_json}")


if __name__ == "__main__":
    main()
