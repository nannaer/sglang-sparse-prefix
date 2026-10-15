# SGLang Prefix Load Balancing: Qwen3-235B LooGLE measurements

These measurements validate the current SGLang integration on a large BF16 model. They compare the upstream no-eviction-cost E2 variant against round-robin with the same local MPQ policy. They do not reproduce the paper's numerical speedups or its full eviction-aware serving configuration.

## Environment and workload

- SGLang base: `ef11e3904885c8050f167a4a6f805da03e9e9bae`; the integration stays on this checkout.
- Hardware: 8 NVIDIA H200 GPUs (143771 MiB each), two independent TP4 replicas.
- Model: `Qwen/Qwen3-235B-A22B-Instruct-2507`, public revision `ac9c66cc9b46af7306746a9250f23d47083d689e`, unquantized BF16. Config, index and tokenizer vocabulary match the public repository; all 118 weight shards match its public LFS SHA-256. All 36,945 tensor headers and shapes were also checked. Local tokenizer differences concern chat template/model maximum length; this workload uses token IDs without a chat template.
- Dataset: `bigai-nlco/LooGLE` at `4b50b4cb9333b48b3f7ddfd307c1d13369be5d1e`, `data/shortdep_qa.jsonl`; content SHA-256 `8951521e8c8340a4e6aa5ffcb2e8f058cca154251915acf946e721fcf001e8c7`.
- Workload: 24 documents, 393 requests, seed 42; 32723-token head/tail truncation target, 45-token output cap, temperature 0. The request count, document count and 0.7 RPS rate follow the public Preble simulation config (which uses dummy weights); this run uses the real 235B BF16 checkpoint. Actual prompt lengths 14268–32722 (mean 23555.04). Workload SHA-256 `b9fbf417c3f588e7b92dd134fff42eead61b7bf3aa10ff3c7020db2f96b74d68`.
- Workers: `fcfs-mpq`, context 32768, chunked prefill 1024, maximum 64 running requests, static memory fraction 0.9, Triton attention, FlashInfer CUTLASS MoE, CUDA graphs disabled.
- GPU KV capacity: worker 0 327054 tokens, worker 1 328505 tokens.
- Runtime packages: `torch=2.11.0+cu128`, `triton=3.6.0`, `sglang-kernel=0.4.4+cu129`, `apache-tvm-ffi=0.1.9`, `flashinfer-python=0.6.12`, `flashinfer-cubin=0.6.12`, `flashinfer-jit-cache=0.6.12+cu128`, `nvidia-cutlass-dsl=4.5.2`, `nvidia-cutlass-dsl-libs-base=4.5.2`, `nvidia-nccl-cu12=2.28.9`, `transformers=5.8.1`.
- Cold single-request calibration: 3864.04 prefill tokens/s and 0.079823 seconds/output token. The frontend uses these measured values, then updates TPOT from request completions. The prefill estimate is linear and omits context-dependent attention cost.

Each trial starts with idle workers and a successful `/flush_cache` on both. Preble starts a fresh frontend, so its tree/history are also empty. Both policies use identical Poisson arrivals and request order. There are two repetitions per arrival rate; trial order reverses on the second repetition. Round-robin connects directly to workers, while Preble includes its frontend HTTP overhead.

Before calibration and replay, both final-config replicas passed three repeated arithmetic answers (`4`) and three repeated capital answers (`Paris`). Real LooGLE samples at 17,544 and 32,722 prompt tokens produced coherent, identical answers across replicas. This is an inference sanity check, not a dataset accuracy evaluation. Full responses are preserved in `semantic-smoke.json`.

The initial CUDA13 kernel wheel silently left MoE TopK outputs uninitialized on driver 570.133.20. Reading that runtime error state returned error35 (insufficient driver). Replacing it with the same-version official CUDA12 wheel, `0.4.4+cu129`, made the same 1/21/1024-token TopK inputs match Torch (maximum error 8.94e-8) and restored model output. The invalid calibration was discarded; the measurements below use only the final CUDA12 configuration. Raw before/after checks are `topk-0.4.4.json` and `topk-0.4.4+cu129.json`.

## Per-trial results

All durations are seconds and include client semaphore waiting. Cache fraction is worker-reported cached prompt tokens divided by all prompt tokens. Completed RPS is observed completions divided by total replay duration, not a saturation throughput estimate.

| Arrival RPS | Policy | Repeat | Success | TTFT mean | TTFT p95 | TTFT p99 | Latency mean | Latency p95 | Latency p99 | Cache fraction | Completed RPS | Output tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.7 | round-robin | 1 | 393/393 | 23.282 | 66.200 | 79.147 | 57.706 | 105.023 | 136.442 | 63.150% | 0.645 | 28.75 |
| 0.7 | round-robin | 2 | 393/393 | 3.834 | 12.780 | 28.562 | 11.987 | 37.044 | 55.121 | 59.543% | 0.651 | 28.87 |
| 1.4 | round-robin | 1 | 393/393 | 15.291 | 55.394 | 73.763 | 33.057 | 77.362 | 97.524 | 66.000% | 1.234 | 54.84 |
| 1.4 | round-robin | 2 | 393/393 | 15.502 | 55.319 | 73.575 | 34.606 | 77.985 | 97.353 | 65.103% | 1.209 | 53.71 |
| 0.7 | preble | 1 | 393/393 | 2.648 | 8.525 | 12.583 | 9.043 | 24.272 | 37.717 | 69.150% | 0.651 | 28.98 |
| 0.7 | preble | 2 | 393/393 | 2.266 | 7.327 | 12.225 | 8.583 | 23.442 | 37.245 | 71.053% | 0.645 | 28.59 |
| 1.4 | preble | 1 | 393/393 | 7.147 | 21.666 | 28.823 | 21.452 | 52.843 | 60.325 | 69.559% | 1.216 | 54.01 |
| 1.4 | preble | 2 | 393/393 | 6.317 | 21.640 | 28.816 | 19.412 | 52.820 | 60.264 | 71.275% | 1.266 | 56.23 |

## Comparison across repetitions

The following ratios divide round-robin mean duration by Preble mean duration, using the mean of the two trial means. They describe these measured workloads, not the paper's hardware or model.

| Arrival RPS | TTFT ratio (RR/Preble) | Latency ratio (RR/Preble) | Mean cache fraction, RR | Mean cache fraction, Preble |
| --- | --- | --- | --- | --- |
| 0.7 | 5.518× | 3.954× | 61.347% | 70.102% |
| 1.4 | 2.287× | 1.656× | 65.552% | 70.417% |

## Routing feedback

| Arrival RPS | Repeat | Scheduled | Migrations | Replications | Active after replay |
| --- | --- | --- | --- | --- | --- |
| 0.7 | 1 | 393 | 562 | 0 | [0, 0] |
| 0.7 | 2 | 393 | 572 | 0 | [0, 0] |
| 1.4 | 1 | 393 | 267 | 0 | [0, 0] |
| 1.4 | 2 | 393 | 294 | 0 | [0, 0] |

## Evidence and reproduction

Raw artifacts are retained locally in `/mnt/nvme1n1/ml_research/chenjiefei/preble-results` and are not included in this public repository: each `round-robin-r<RATE>-rep<N>` or `preble-r<RATE>-rep<N>` directory contains the exact command, `requests.jsonl`, `summary.json` and replay log. Preble trials also contain `/routing_stats` after completion; active counts must be `[0, 0]`. Model/dataset revisions, calibration samples, worker commands, dependencies and server info are stored alongside them.

Use [README.md](README.md) for the worker/frontend/workload commands. Replay each policy at 0.7 and 1.4 RPS twice with seed 42, flushing both caches between trials and restarting the frontend before each Preble trial. The original comparison harness, `run_comparison.py`, is also retained only in the local artifact directory; the public replay entry point is [bench_loogle.py](../../benchmark/preble/bench_loogle.py).

The global scheduler does not receive actual KV eviction events or estimate eviction penalties. Each worker still uses normal radix LRU; global assignments can therefore overestimate locality after an eviction. This experiment holds local MPQ constant to isolate global routing, whereas the paper's full comparison also changes local scheduling and uses different models/GPUs. Arrival rates, two repetitions and a single dataset are insufficient to establish saturation throughput or a general serving speedup.
