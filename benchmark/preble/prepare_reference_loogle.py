# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Archive the original Preble LooGLE generator on pinned legacy data.

This script executes selected AST definitions from the pinned
reference checkout. It does not import or modify that old SGLang installation.
Only dataset I/O, UUID entropy, the tokenizer, and explicit RNG boundaries are
adapted. The original prompt construction, proportional QA replication, both
shuffles, tokenization, and head/tail retokenization execute unchanged.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import random
import re
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace

import datasets
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

REFERENCE_COMMIT = "1a35eae20c416f7062ca81775aab72890323fd97"
SOURCE_SHA256 = {
    "preble/benchmarks/benchmark_workload_gen.py": "d4a37bcce25d56d73af7e364f88321f1aacf3dec7db3986b329be1c0c6465170",
    "preble/benchmarks/multi_exp_configs/multi_exp_utils.py": "dfc0f0ac220ccbf436c5c9cf7b84d5a61ce3a00ff46454f78913c4d81268b1df",
}
DATASET_REVISION = "bdc356116a09a6a3d552e28a9f330878915b1696"
DATASET_SHA256 = "065e2036eada3513c02835137a4c83e483fbfbf8922740b7f14c6a22542dbc8e"
DATASET_URL = (
    "https://huggingface.co/datasets/bigai-nlco/LooGLE/resolve/"
    + DATASET_REVISION
    + "/shortdep_qa/test-00000-of-00001.parquet"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def execute_definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    found = {node.name for node in definitions}
    if found != set(names):
        raise RuntimeError(
            f"Missing source definitions in {path}: {set(names) - found}"
        )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *definitions,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return {
        node.name: hashlib.sha256(
            ast.dump(node, include_attributes=False).encode()
        ).hexdigest()
        for node in definitions
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help=f"Local WukLab/preble checkout at {REFERENCE_COMMIT}",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help=f"Legacy shortdep_qa parquet from revision {DATASET_REVISION}",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patterns", type=int, default=24)
    parser.add_argument("--requests", type=int, default=393)
    parser.add_argument("--max-new-tokens", type=int, default=45)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--uuid-seed", type=int, default=10)
    parser.add_argument("--arrival-seed", type=int, default=10)
    parser.add_argument("--rates", nargs="+", type=float, default=[0.7, 1.4])
    args = parser.parse_args()
    if min(args.patterns, args.requests, args.max_new_tokens) < 1:
        parser.error("Patterns, requests, and output cap must be positive")
    if args.max_new_tokens >= 32768:
        parser.error("Output cap must leave room in the 32768-token context")
    if any(not math.isfinite(rate) or rate <= 0 for rate in args.rates):
        parser.error("Arrival rates must be finite and positive")
    if len({f"{rate:g}" for rate in args.rates}) != len(args.rates):
        parser.error("Rates must produce distinct arrival filenames")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new directory")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.source_root, text=True
    ).strip()
    if commit != REFERENCE_COMMIT:
        raise RuntimeError(f"Unexpected reference commit: {commit}")
    for relative_path, expected_sha256 in SOURCE_SHA256.items():
        if sha256(args.source_root / relative_path) != expected_sha256:
            raise RuntimeError(f"Pinned source SHA256 does not match: {relative_path}")
    if sha256(args.dataset) != DATASET_SHA256:
        raise RuntimeError("Pinned dataset SHA256 does not match")

    dataset = datasets.Dataset.from_parquet(str(args.dataset))
    if len(dataset) != 105 or set(dataset.column_names) != {
        "input",
        "title",
        "qa_pairs",
        "output",
    }:
        raise RuntimeError("Unexpected legacy LooGLE schema")
    legacy_rows = list(dataset)
    source_indices = {
        hashlib.sha256(row["input"].encode()).hexdigest(): index
        for index, row in enumerate(legacy_rows)
    }
    if len(source_indices) != len(legacy_rows):
        raise RuntimeError("Legacy contexts are not unique")
    selected_rows = []

    class DatasetInput:
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def shuffle(self, *positional, **keyword):
            return DatasetInput(self.data.shuffle(*positional, **keyword))

        def select(self, *positional, **keyword):
            selected = self.data.select(*positional, **keyword)
            selected_rows[:] = list(selected)
            return selected

    def load_dataset(*positional, **keyword):
        if positional != ("bigainlco/LooGLE", "shortdep_qa") or keyword != {
            "split": "test"
        }:
            raise RuntimeError("Unexpected reference dataset request")
        return DatasetInput(dataset)

    nonce_rng = random.Random(args.uuid_seed)
    nonces = []

    def deterministic_uuid4():
        value = uuid.UUID(int=nonce_rng.getrandbits(128), version=4)
        nonces.append(value.hex)
        return value

    namespace = dict(
        __name__="preble_reference_generator",
        AutoTokenizer=AutoTokenizer,
        Enum=Enum,
        auto=auto,
        LoadDistribution=SimpleNamespace(EVEN="EVEN"),
        ThreadPoolExecutor=ThreadPoolExecutor,
        load_dataset=load_dataset,
        logging=logging,
        math=math,
        np=np,
        random=random,
        tqdm=tqdm,
        uuid=SimpleNamespace(uuid4=deterministic_uuid4),
    )
    generator_path = args.source_root / "preble/benchmarks/benchmark_workload_gen.py"
    helper_path = (
        args.source_root / "preble/benchmarks/multi_exp_configs/multi_exp_utils.py"
    )
    ast_hashes = execute_definitions(
        generator_path,
        ["DataLoader", "LooGLEDatasetType", "LooGLEDataset"],
        namespace,
    )
    ast_hashes.update(
        execute_definitions(
            helper_path, ["calc_send_out_times", "create_loogle_dataset"], namespace
        )
    )
    original_arrivals = namespace["calc_send_out_times"]

    def seeded_arrivals(requests, rate, experiment_time):
        # Keep NumPy's original MT19937/exponential implementation; make the
        # arrival RNG independent of HF dataset shuffle and tokenization.
        np.random.seed(args.arrival_seed)
        return original_arrivals(requests, rate, experiment_time)

    namespace["calc_send_out_times"] = seeded_arrivals
    random.seed(args.seed)
    np.random.seed(args.seed)
    _, requests, first_arrivals = namespace["create_loogle_dataset"](
        [args.patterns, args.requests, args.rates[0]],
        args.model_path,
        float("inf"),
        max_tokens_override=args.max_new_tokens,
    )

    total_qa = sum(len(ast.literal_eval(row["qa_pairs"])) for row in selected_rows)
    scale = args.requests / total_qa
    nonce_metadata = {}
    selection = []
    generated_index = 0
    for row in selected_rows:
        context_sha = hashlib.sha256(row["input"].encode()).hexdigest()
        document_index = source_indices[context_sha]
        pairs = ast.literal_eval(row["qa_pairs"])
        generated_count = math.ceil(len(pairs) * scale)
        selection.append(
            dict(
                source_document_index=document_index,
                context_sha256=context_sha,
                title=row["title"],
                qa_pairs=len(pairs),
                generated_requests=generated_count,
            )
        )
        for index in range(generated_count):
            nonce_metadata[nonces[generated_index]] = dict(
                doc_id=f"legacy-{document_index:03d}-{context_sha[:12]}",
                source_document_index=document_index,
                source_qa_index=index % len(pairs),
                generation_index=generated_index,
            )
            generated_index += 1
    if generated_index != len(nonces) or len(requests) != args.requests:
        raise RuntimeError("Unexpected reference replication/request count")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    workload_path = (
        args.output_dir / f"loogle-{args.patterns}-{args.requests}-reference.jsonl"
    )
    lengths = []
    retained_documents = set()
    with workload_path.open("w") as output:
        for index, request in enumerate(requests):
            matched = [
                value
                for value in re.findall(r"Question: ([0-9a-f]{32})", request["text"])
                if value in nonce_metadata
            ]
            if len(matched) != 1:
                raise RuntimeError(
                    "Cannot recover original question UUID after truncation"
                )
            metadata = nonce_metadata[matched[0]]
            retained_documents.add(metadata["source_document_index"])
            record = dict(
                request_id=index,
                **metadata,
                text=request["text"],
                input_ids=request["input_ids"],
                sampling_params=request["sampling_params"],
            )
            lengths.append(len(record["input_ids"]))
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    arrival_artifacts = []
    for index, rate in enumerate(args.rates):
        # Source appends a dummy last sleep; the replay file contains N actual
        # request deadlines, starts at zero, and has no dummy entry.
        full_times = (
            first_arrivals
            if index == 0
            else seeded_arrivals(requests, rate, float("inf"))
        )
        times = full_times[:-1]
        if len(times) != len(requests) or times[0] != 0:
            raise RuntimeError("Unexpected reference arrival schedule")
        target = args.output_dir / f"arrivals-r{rate:g}.json"
        write_json(target, times)
        arrival_artifacts.append(
            dict(
                rate=rate,
                path=str(target.resolve()),
                sha256=sha256(target),
                requests=len(times),
                first_seconds=times[0],
                last_seconds=times[-1],
            )
        )
    model_path = Path(args.model_path)
    tokenizer_files = {
        name: sha256(model_path / name)
        for name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "tokenizer.model",
            "config.json",
        ]
        if (model_path / name).is_file()
    }
    manifest = dict(
        reference_commit=commit,
        source_files={
            str(path.relative_to(args.source_root)): sha256(path)
            for path in [generator_path, helper_path]
        },
        executed_definitions_ast_sha256=ast_hashes,
        dataset_revision=DATASET_REVISION,
        dataset_url=DATASET_URL,
        dataset_path=str(args.dataset.resolve()),
        dataset_sha256=DATASET_SHA256,
        preparation_script_sha256=sha256(Path(__file__)),
        model_path=args.model_path,
        tokenizer_files_sha256=tokenizer_files,
        config={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        workload_path=str(workload_path.resolve()),
        workload_sha256=sha256(workload_path),
        actual_requests=len(requests),
        actual_patterns=len(retained_documents),
        generated_before_final_slice=generated_index,
        selected_documents=selection,
        prompt_length=dict(
            min=min(lengths), mean=float(np.mean(lengths)), max=max(lengths)
        ),
        arrival_artifacts=arrival_artifacts,
        versions={
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in [
                    "datasets",
                    "numpy",
                    "transformers",
                    "tokenizers",
                    "pyarrow",
                ]
            },
        },
        differences_from_upstream=[
            "Pinned legacy parquet replaces the unversioned HF loader; row contents/schema are preserved.",
            "The configured tokenizer replaces the reference Mistral-7B tokenizer; no model weights are loaded here.",
            "UUID4 entropy uses an independent fixed Python RNG and standard UUIDv4 version/variant bits; upstream uses OS entropy.",
            "Python and NumPy are explicitly seeded immediately before preparation, instead of relying on entrypoint import order.",
            "NumPy MT19937 seed is explicitly reset before arrival generation; upstream shares global NumPy state with dataset shuffle.",
            "Archived arrivals omit only the upstream dummy final sleep entry; there is one deadline per request.",
            "Added document/request metadata and archived text/input_ids do not change generation or ordering.",
            "Current recorded dependency versions replace the original unpinned dependency environment.",
        ],
    )
    manifest_path = args.output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "workload": str(workload_path),
                "manifest": str(manifest_path),
                "requests": len(requests),
                "patterns": manifest["actual_patterns"],
                "workload_sha256": manifest["workload_sha256"],
                "prompt_length": manifest["prompt_length"],
                "arrivals": arrival_artifacts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
