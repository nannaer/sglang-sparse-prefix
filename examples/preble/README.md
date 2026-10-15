# Preble 执行流程与运行方法

参考 [WukLab/Preble `1a35eae20c`](https://github.com/WukLab/preble/tree/1a35eae20c416f7062ca81775aab72890323fd97) 的公开服务默认路径，保留当前 SGLang runtime。全局路由、代价公式与 MPQ 伪代码见[项目主页](../../README.md)，逐项[一致性验证](ALIGNMENT_REAUDIT.md)及[235B 单轮结果](RUNTIME_2296.md)可单独查看。

## 执行流程

```text
客户端 /generate
  → frontend tokenize、插入全局前缀树、估计负载与驱逐代价、选择副本
  → 发送相同 input_ids 到选中 worker
  → worker 持久 MPQ 分组、FIFO 准入、SGLang prefill / decode
  → frontend 转发累计 SSE；完成后释放引用、更新 TTFT / TPOT
```

每个 worker 的实际 KV 在自己的 HBM。全局树维护路由归属与虚拟缓存；不接收真实 eviction 事件，也不完整统计生成输出的实际 KV。migration / replication 修改未来归属，新副本由后续请求计算缓存；这里没有跨 worker KV 传输或 Mooncake。

Frontend 支持同模型、同 tokenizer、固定 worker 池的原生 `/generate`，单条 text 或 input_ids、n=1，默认流式；显式 stream=False 为扩展。不支持 OpenAI 路由入口、batch、LoRA、多模态、session 或 cache_salt。另有 `/health`、`/routing_stats`。

## 启动两个 TP4 副本

使用当前 checkout 的兼容环境。以下配置对应 **8×H200、完整 BF16 Qwen3-235B-A22B-Instruct-2507**；模型权重和模型下载缓存必须在开发目录之外。

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export PREBLE_MODEL_PATH=/models/path/to/Qwen3-235B-A22B-Instruct-2507
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path "$PREBLE_MODEL_PATH" --tp-size 4 --dtype bfloat16 \
  --host 127.0.0.1 --port 31000 --random-seed 659938497 \
  --mem-fraction-static 0.90 --context-length 32768 \
  --max-running-requests 64 --chunked-prefill-size 512 \
  --schedule-policy fcfs-mpq --attention-backend triton \
  --moe-runner-backend flashinfer_cutlass --disable-cuda-graph
```

另一个终端启动 worker1：使用相同命令，将 GPU 改为 `4,5,6,7`、port 改为 `31001`、seed 改为 `556540270`。

**等待两端日志分别出现完整一行 `The server is fired up and ready to roll!`，再执行任何外部 flush / warmup / 校准。** 仅 `/health` 或 `/server_info` 可访问不足以证明内置 warmup 已完成。随后读取两端 `/server_info` 的 `internal_states[0].memory_usage.token_capacity`；TP4 的容量不能乘 4。2296 本次每端实际为 328505 tokens。

```bash
python -m sglang.launch_preble \
  --worker-urls http://127.0.0.1:31000 http://127.0.0.1:31001 \
  --model-path "$PREBLE_MODEL_PATH" --host 127.0.0.1 --port 18000 \
  --max-tokens-per-worker 328505 328505 \
  --prefill-time-scale 2.387698396256729 --initial-tpot 0.06731290138453584
```

容量与时间参数应使用实际部署值。CLI 默认启用 eviction 和 rebalancing；窗口 180 秒。原二维 prefill 模型保持结构，只允许整体 scale 校准；本次沿用正式实验前冻结的参数，不用单轮结果重新拟合。`--disable-eviction`、`--disable-rebalancing` 可作消融。

## 准备与重放 LooGLE

[轨迹准备脚本](../../benchmark/preble/prepare_reference_loogle.py) 校验原源码 / 旧 schema 数据 hash，执行原生成器 AST。固定 seed / UUID 随机源是现代环境复现适配，不等同论文当年的逐请求 trace。

```bash
mkdir -p preble-bench
git clone https://github.com/WukLab/preble.git preble-bench/reference
git -C preble-bench/reference checkout 1a35eae20c416f7062ca81775aab72890323fd97
curl --fail --location \
  https://huggingface.co/datasets/bigai-nlco/LooGLE/resolve/bdc356116a09a6a3d552e28a9f330878915b1696/shortdep_qa/test-00000-of-00001.parquet \
  --output preble-bench/shortdep_qa.parquet
python benchmark/preble/prepare_reference_loogle.py \
  --source-root preble-bench/reference --dataset preble-bench/shortdep_qa.parquet \
  --model-path "$PREBLE_MODEL_PATH" --output-dir preble-bench/reference-trace \
  --patterns 24 --requests 393 --max-new-tokens 45 \
  --seed 10 --uuid-seed 10 --arrival-seed 10 --rates 1.4
```

新部署可在就绪后用 [calibrate.py](../../benchmark/preble/calibrate.py) 对两端执行 cold 8K / 16K / 32K 与部分命中 profile，然后冻结结果；该脚本不自动等待上述 ready 日志。

```bash
python benchmark/preble/calibrate.py \
  --workload preble-bench/reference-trace/loogle-24-393-reference.jsonl \
  --worker-urls http://127.0.0.1:31000 http://127.0.0.1:31001 \
  --max-new-tokens 45 --output preble-bench/calibration.json
```

将输出的 `prefill_time_scale` / `initial_tpot` 传给 frontend 并重启，正式回放前冻结；复用本次部署时沿用上面的归档值即可。

正式回放前等在途归零，对两端 POST `/flush_cache?timeout=30`，确认 `Cache flushed.`，并启动全新 frontend。每轮使用新输出目录：

```bash
python benchmark/preble/bench_loogle.py \
  --workload preble-bench/reference-trace/loogle-24-393-reference.jsonl \
  --arrivals preble-bench/reference-trace/arrivals-r1.4.json \
  --policy preble --router-url http://127.0.0.1:18000 \
  --worker-urls http://127.0.0.1:31000 http://127.0.0.1:31001 \
  --requests 393 --rate 1.4 --concurrency 256 --output preble-bench/preble-r1.4-once
```

核验 393 成功、393 个 TTFT、frontend active=[0,0]，并保存 summary / requests / routing_stats。`--arrivals` 决定计划到达；TTFT / E2E 从 sleep 醒后起钟，包含 semaphore、HTTP、frontend 与计算，不含唤醒迟延。完成 RPS 包含到达等待和尾部排空，不代表饱和吞吐。

2296 本次只运行一轮 Preble + MPQ，没有追加 RR 或重复轮次。旧 ABBA 保留为[历史结果](RUNTIME_RESULTS.md)，其两轮方差的证据与限制见[简短调查结论](VARIANCE_ANALYSIS.md)。

## 源码入口

| 功能 | 当前实现 |
| --- | --- |
| HTTP 与完成反馈 | [launch_preble.py](../../python/sglang/launch_preble.py) |
| E2、重平衡 | [preble.py](../../python/sglang/srt/routing/preble.py) |
| 窗口负载、时间模型、热点检测 | [preble_cost.py](../../python/sglang/srt/routing/preble_cost.py) |
| 全局虚拟 radix / LRU | [preble_cache.py](../../python/sglang/srt/routing/preble_cache.py) |
| MPQ 分组与准入 | [schedule_policy.py](../../python/sglang/srt/managers/schedule_policy.py)、[scheduler.py](../../python/sglang/srt/managers/scheduler.py) |
| MPQ retract | [schedule_batch.py](../../python/sglang/srt/managers/schedule_batch.py) |
