# Preble scheduling on current SGLang

This integration adds Preble's global exploration/exploitation (E2) routing and
local cache-hit priority queues to the current SGLang runtime. It uses this
checkout's native `/generate` API and scheduler interfaces; it does not install
the older SGLang fork shipped with Preble.

The reference is [WukLab/Preble](https://github.com/WukLab/preble) at commit
[`1a35eae20c416f7062ca81775aab72890323fd97`](https://github.com/WukLab/preble/tree/1a35eae20c416f7062ca81775aab72890323fd97),
released with [Preble: Efficient Distributed Prompt Scheduling for LLM Serving
(ICLR 2025)](https://iclr.cc/virtual/2025/poster/28456)
([arXiv preprint](https://arxiv.org/abs/2407.00023)). Benchmark measurements and their
environment details belong in [RESULTS.md](RESULTS.md); the workload below does
not by itself establish reproduction of the paper's performance.

## Implementation

| Component | Reference implementation | Current integration |
| --- | --- | --- |
| Global E2 routing and rebalancing | [global_scheduler_with_time.py](https://github.com/WukLab/preble/blob/1a35eae20c416f7062ca81775aab72890323fd97/preble/global_scheduler_with_time.py) | [PrebleScheduler](../../python/sglang/srt/routing/preble.py) |
| Compressed token radix and request references | [global_lru_cache.py](https://github.com/WukLab/preble/blob/1a35eae20c416f7062ca81775aab72890323fd97/preble/global_lru_cache.py) | `PrefixNode` and `Assignment` in the scheduler |
| TTFT overload detection | [ttft_overload_detector.py](https://github.com/WukLab/preble/blob/1a35eae20c416f7062ca81775aab72890323fd97/preble/ttft_overload_detector.py) | `PrebleScheduler._overloaded` |
| Local multi-priority queues | [model_rpc.py](https://github.com/WukLab/preble/blob/1a35eae20c416f7062ca81775aab72890323fd97/python/sglang/srt/managers/router/model_rpc.py) | [SchedulePolicy](../../python/sglang/srt/managers/schedule_policy.py), selected with `--schedule-policy fcfs-mpq` |
| HTTP forwarding and completion feedback | [server/server.py](https://github.com/WukLab/preble/blob/1a35eae20c416f7062ca81775aab72890323fd97/preble/server/server.py) | [sglang.launch_preble](../../python/sglang/launch_preble.py) |

The frontend inserts prompt token IDs into a compressed radix tree. A node is
large when its segment contains more tokens than all preceding context:
`segment_tokens > context_tokens - segment_tokens`. Each request belongs to its
closest large ancestor, called its important node. Splitting a radix node copies
its assignments and active references, and regroups historical observations
against the updated tree.

E2 exploits locality when the assigned matching prefix is longer than the
unmatched suffix: it chooses the least-cost worker among that prefix's assigned,
available workers. Otherwise it explores all available workers. Estimated worker
cost combines windowed prefix prefill cost with outstanding output-token budgets
multiplied by measured time per output token (TPOT). The frontend forwards exactly
the token IDs used for this decision, so workers do not independently retokenize
the prompt.

| Mechanism | Default behavior |
| --- | --- |
| Prefix request history | 180-second sliding window; `--window-seconds` changes it |
| Prefill estimate | Request count × estimated miss fraction × prefill time, divided among assigned workers |
| Decode estimate | Outstanding requested output-token budgets × each worker's median TPOT |
| TPOT feedback | Up to 200 samples per worker, seeded with the required initial measurement |
| Rebalancing warmup | Starts after `50 × number_of_workers` scheduled requests |
| Rebalancing eligibility | Prefix groups observed at least twice; request-to-important-node chain shorter than three edges |
| Load threshold | Consider rebalancing when the highest estimated load exceeds `1.5 ×` the lowest |
| Group migration | Move smaller groups while the move does not reverse the two workers' estimated load order |
| Hotspot replication | For one indivisible group, add an assignment when recent-half mean TTFT is at least `2 ×` older-half mean TTFT; both halves need observations |

With the default window, the TTFT halves are 90 seconds each. Rebalancing changes
where future requests run. In-flight requests keep their original worker, and an
additional assignment causes that worker to compute/cache the prefix as requests
arrive.

The local `fcfs-mpq` policy uses ten cache-hit-rate groups. Zero-hit requests and
requests with up to 10% hits are in group 0; subsequent 10% bands reach group 9.
The policy visits nonempty groups from 9 down to 0 in a repeating cycle, preserves
FIFO by queue-entry time within each group, and permits up to `10 × (group + 1)`
requests from the selected group per prefill decision. Existing SGLang token,
request and chunked-prefill budgets can reduce admission. Membership is refreshed
from current prefix matches, and a continuing chunked request retains the
runtime's existing admission path.

## Scope and differences from upstream

The global scheduler implements the upstream constructor's
`enable_eviction=False` variant. It does not simulate GPU cache capacity or add
the upstream virtual-LRU victim recomputation penalty to routing cost. The
upstream public serving entry and its two-worker LooGLE simulation config enable
that optional penalty, so this is not a complete port of those configurations.
Each worker still performs normal GPU radix-cache eviction. The frontend's tree
records routing assignments and estimated locality, rather than authoritative
KV residency.

The integration does not move KV tensors between workers and does not add
Mooncake, a shared DRAM store, or a KV connector. Routing migration and replication
refer to assignments, not memory transfers.

The following corrections and adaptations are deliberate:

- Exact repeated prompts exploit an existing assignment. Upstream classifies the
  existing endpoint by its radix segment length and can explore again on a full
  match. Hit estimates here use the selected worker's existing assignment path;
  actual worker `cached_tokens` remain the benchmark's cache measurement.
- Outstanding decode budgets survive history expiry and assignment migration,
  and are released once on completion, errors or cancellation.
- Idle assignments expire with the history window; in-flight paths remain
  protected. Local MPQ groups are refreshed at each scheduling decision, with
  FIFO preservation and admission restricted to the chosen group.
- The frontend requires a measured prefill rate and initial TPOT. The launch CLI
  uses `tokens / measured_rate` for prefill cost; it does not use Preble's
  Mistral-7B/A6000 regression for the 235B model below. This linear estimate does
  not model context-dependent attention cost.

The frontend supports a fixed pool of identical models/tokenizers and one shared
cache namespace. Its inference endpoint is native `POST /generate`, with one
text string or token-ID list and `sampling_params.n=1`. It also exposes `/health`
and `/routing_stats`. OpenAI-compatible endpoints, batch generation, LoRA,
multimodal input, stateful sessions and `extra_key/cache_salt` namespaces are not
supported. Startup queries every worker's `/server_info` and rejects
`incremental_streaming_output=True`; cumulative streaming is required to return
complete nonstream responses. Client disconnects abort the backend request and
release routing references, including disconnects while waiting for HTTP headers.

## Launch two 235B replicas on eight H200 GPUs

Use an environment with the CUDA and kernel dependencies required by this
checkout. For CUDA 12.8 or 12.9, follow the CUDA 12 branches in
[docker/Dockerfile](../../docker/Dockerfile): use the
`sglang-kernel==0.4.4+cu129` wheel and `torch==2.11.0` from the corresponding
CUDA 12 PyTorch index (`cu128` for CUDA 12.8). Match FlashInfer's JIT-cache build
to that CUDA environment and record the installed versions with the results.
Run the commands from the repository root, with a complete BF16
[Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507)
checkpoint. Both replicas, the frontend and the workload generator must use the
same tokenizer directory. These are two independent tensor-parallel replicas,
each using four GPUs.

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export PREBLE_MODEL_PATH=/path/to/Qwen3-235B-A22B-Instruct-2507
```

Run worker 0 in one terminal:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path "$PREBLE_MODEL_PATH" --tp-size 4 \
  --host 127.0.0.1 --port 31000 \
  --mem-fraction-static 0.90 --context-length 32768 \
  --max-running-requests 64 --chunked-prefill-size 1024 \
  --schedule-policy fcfs-mpq --attention-backend triton \
  --moe-runner-backend flashinfer_cutlass --disable-cuda-graph
```

Run worker 1 in another terminal, using the same configuration:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m sglang.launch_server \
  --model-path "$PREBLE_MODEL_PATH" --tp-size 4 \
  --host 127.0.0.1 --port 31001 \
  --mem-fraction-static 0.90 --context-length 32768 \
  --max-running-requests 64 --chunked-prefill-size 1024 \
  --schedule-policy fcfs-mpq --attention-backend triton \
  --moe-runner-backend flashinfer_cutlass --disable-cuda-graph
```

`flashinfer_cutlass` is the BF16 MoE backend here; attention uses Triton. Wait for
both workers' `/health` endpoints to succeed. Measure cold prefill and decode on
this model and worker configuration, after kernel warmup, with an idle request
queue and representative prompt lengths. Use the cold prompt token count divided
by its prefill duration for tokens/second; estimate TPOT from
`(generation_latency - TTFT) / (completion_tokens - 1)` over several requests.
Set `PREBLE_PREFILL_TOKENS_PER_SECOND` and `PREBLE_INITIAL_TPOT` to those positive
measurements, then start the frontend in a third terminal:

```bash
python -m sglang.launch_preble \
  --worker-urls http://127.0.0.1:31000 http://127.0.0.1:31001 \
  --model-path "$PREBLE_MODEL_PATH" --host 127.0.0.1 --port 8000 \
  --prefill-tokens-per-second "${PREBLE_PREFILL_TOKENS_PER_SECOND:?Set the measured prefill rate}" \
  --initial-tpot "${PREBLE_INITIAL_TPOT:?Set the measured decode TPOT}"
```

Use `--disable-rebalancing` for an E2 routing ablation. Example native request:

```bash
curl --fail http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Explain prefix caching.","sampling_params":{"temperature":0,"max_new_tokens":45}}'
```

## LooGLE workload and comparison

The [benchmark driver](../../benchmark/preble/bench_loogle.py) adapts the official
[LooGLEDataset generator](https://github.com/WukLab/preble/blob/1a35eae20c416f7062ca81775aab72890323fd97/preble/benchmarks/benchmark_workload_gen.py#L830)
and [create_loogle_dataset helper](https://github.com/WukLab/preble/blob/1a35eae20c416f7062ca81775aab72890323fd97/preble/benchmarks/multi_exp_configs/multi_exp_utils.py#L69).
The upstream [two-worker LooGLE simulation config](https://github.com/WukLab/preble/blob/1a35eae20c416f7062ca81775aab72890323fd97/preble/benchmarks/multi_exp_configs/e2e_234r_loogle_config.py#L109)
supplies 24 document prefixes, 393 requests, 0.7 requests/second, a 45-token
output cap and a prompt truncation target of `32768 - 45 = 32723` tokens. It sets
`simulate=True` and `load_format='dummy'` for Mistral-7B; the commands below adapt
its workload parameters to two real BF16 235B replicas. Also run 1.4
requests/second with the same workload as a higher-load comparison.

The current public short-dependency QA JSONL has one QA per row, with `doc_id`,
`context`, `question` and `answer`, rather than the older `input` plus stringified
`qa_pairs` schema. The driver groups rows by `doc_id`, samples documents, replicates
their QA pairs in proportion to available QA counts, and inserts a random
128-bit hexadecimal nonce before each question. It preserves the official prompt
template and head/tail decode-and-retokenize truncation. A fixed seed of 42 makes
document selection, nonces, request shuffling and exponential interarrival times
repeatable; this is a deterministic adaptation, not the same random trace as the
upstream unseeded dataset shuffle/UUID generation. Preparation writes dataset and
workload SHA-256 hashes to a manifest.

```bash
mkdir -p preble-bench
curl --fail --location \
  https://huggingface.co/datasets/bigai-nlco/LooGLE/resolve/4b50b4cb9333b48b3f7ddfd307c1d13369be5d1e/data/shortdep_qa.jsonl \
  --output preble-bench/shortdep_qa.jsonl

python benchmark/preble/bench_loogle.py --prepare-only \
  --dataset preble-bench/shortdep_qa.jsonl --model-path "$PREBLE_MODEL_PATH" \
  --workload preble-bench/loogle-24-393.jsonl \
  --patterns 24 --requests 393 --max-input-tokens 32723 \
  --max-new-tokens 45 --seed 42
```

Keep worker settings identical to compare global routing: round-robin sends
directly to alternating workers, while Preble sends through the frontend, so
Preble's frontend overhead is included. Wait for all requests to finish and flush
both worker caches before each trial; restart the frontend before each Preble
trial to clear its history and assignments.

```bash
curl --fail http://127.0.0.1:31000/flush_cache
curl --fail http://127.0.0.1:31001/flush_cache

python benchmark/preble/bench_loogle.py \
  --workload preble-bench/loogle-24-393.jsonl --requests 393 \
  --policy round-robin \
  --worker-urls http://127.0.0.1:31000 http://127.0.0.1:31001 \
  --rate 0.7 --concurrency 256 --seed 42 --output preble-bench/rr-0.7
```

After flushing again and starting a fresh frontend, run:

```bash
python benchmark/preble/bench_loogle.py \
  --workload preble-bench/loogle-24-393.jsonl --requests 393 \
  --policy preble --router-url http://127.0.0.1:8000 \
  --worker-urls http://127.0.0.1:31000 http://127.0.0.1:31001 \
  --rate 0.7 --concurrency 256 --seed 42 --output preble-bench/preble-0.7
```

Repeat both trials with `--rate 1.4` and separate output directories. Each trial
writes `requests.jsonl` and `summary.json`: successful/failed requests, observed
TTFT and latency percentiles, completed requests/second, output tokens/second and
the actual worker-reported cached-token fraction. Client semaphore waiting is
included in TTFT/latency and also recorded separately. A failed or incomplete
generation makes the driver exit unsuccessfully. Compare workload hashes and
record model revision, worker commands, runtime dependencies and warmup/reset
procedure alongside results.
