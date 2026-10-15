# 完整 Preble + MPQ：Qwen3-235B 验证

**历史实验。** 以下八轮使用本轮五项 runtime 修正之前的实现，并按策略分阶段执行。原始数据保留；这些结果不作为 `c773f3e560` 中 TPOT、准入扫描、FIFO、chunk 轮转与取消引用修正的 GPU 验证。

本次比较已移植全局驱逐代价的 **Preble + MPQ** 与原版主要基线 **round-robin + LPM**。旧版“不含 eviction cost、两方都用 MPQ”的实验不作为此次对齐证据。

参考实现是 [WukLab/Preble `1a35eae20c416f7062ca81775aab72890323fd97`](https://github.com/WukLab/preble/tree/1a35eae20c416f7062ca81775aab72890323fd97) 的公开服务默认路径。模型、GPU、runtime、随机流与论文不同，以下衡量当前大模型部署的行为与性能，不声称复现论文的数值加速比。

## 实测结果

八轮共 **3144/3144 请求成功**，TTFT / E2E 均完整；所有 Preble 轮结束后 `active=[0,0]`。以下单位为秒，cache 是按 token 数加权的实际 worker 命中比例。

| 到达 RPS | 策略 | 请求数 | TTFT mean | TTFT p95 | E2E mean | E2E p95 | Cache | 完成 RPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.7 | Preble + MPQ | 786 | 5.940 | 33.149 | 17.481 | 88.824 | 88.565% | 0.761 |
| 0.7 | RR + LPM | 786 | 25.851 | 141.529 | 62.956 | 190.543 | 69.515% | 0.647 |
| 1.4 | Preble + MPQ | 786 | 42.364 | 175.520 | 70.438 | 191.755 | 84.719% | 1.113 |
| 1.4 | RR + LPM | 786 | 31.310 | 126.116 | 61.814 | 149.383 | 78.094% | 1.161 |

平均时延的 RR / Preble 比值如下；大于 1 表示 Preble 较快，小于 1 表示 RR 较快。这些是两轮实测的描述性比值。

| 到达 RPS | TTFT：RR / Preble | E2E：RR / Preble |
| --- | --- | --- |
| 0.7 | 4.352× | 3.601× |
| 1.4 | 0.739× | 0.878× |

在本次固定轨迹上，0.7 RPS 档的 Preble 平均 TTFT / E2E 分别为基线的约 1/4.35、1/3.60；1.4 RPS 档则分别比基线高约 35.3%、14.0%，且两种策略的重复间波动都明显。缓存命中率提高没有在高负载档转化为更低时延。因此，本次结果支持低负载档的实测收益，不能支持跨负载稳定加速，也没有复现论文的数值性能。比较只有两次重复，且两策略没有按时间交替，以下保留全部轮次供核查。

### 逐轮波动

| 策略 | 到达 RPS | 重复 | 成功 | TTFT mean | TTFT p95 | E2E mean | Cache |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Preble + MPQ | 0.7 | 1 | 393/393 | 5.898 | 33.026 | 17.193 | 88.359% |
| Preble + MPQ | 0.7 | 2 | 393/393 | 5.981 | 33.051 | 17.769 | 88.771% |
| Preble + MPQ | 1.4 | 1 | 393/393 | 31.293 | 78.446 | 55.028 | 86.475% |
| Preble + MPQ | 1.4 | 2 | 393/393 | 53.435 | 193.860 | 85.847 | 82.964% |
| RR + LPM | 1.4 | 1 | 393/393 | 43.751 | 157.755 | 77.162 | 81.280% |
| RR + LPM | 1.4 | 2 | 393/393 | 18.869 | 69.359 | 46.467 | 74.907% |
| RR + LPM | 0.7 | 1 | 393/393 | 25.477 | 135.484 | 62.293 | 69.449% |
| RR + LPM | 0.7 | 2 | 393/393 | 26.224 | 143.355 | 63.619 | 69.580% |

### 路由反馈

| 到达 RPS | 重复 | Migration | Replication | 请求数：worker 0 / 1 | 最终 active |
| --- | --- | --- | --- | --- | --- |
| 0.7 | 1 | 34 | 0 | 185 / 208 | [0, 0] |
| 0.7 | 2 | 40 | 0 | 184 / 209 | [0, 0] |
| 1.4 | 1 | 35 | 0 | 202 / 191 | [0, 0] |
| 1.4 | 2 | 48 | 0 | 189 / 204 | [0, 0] |

四轮共 157 次 migration；replication 未在本次 GPU 轨迹触发，仅有源码与 CPU 定向验证覆盖。以上操作调整未来路由归属，不搬运 KV tensor。

### 输出与客户端等待

| 到达 RPS | 策略 | 输出 tokens：min / mean / max | Client queue mean | Client queue p95 | Client queue max |
| --- | --- | --- | --- | --- | --- |
| 0.7 | Preble + MPQ | 14 / 44.541 / 45 | 0.000011 | 0.000015 | 0.000239 |
| 0.7 | RR + LPM | 14 / 44.584 / 45 | 0.000015 | 0.000021 | 0.000246 |
| 1.4 | Preble + MPQ | 14 / 44.533 / 45 | 0.000010 | 0.000013 | 0.000046 |
| 1.4 | RR + LPM | 14 / 44.583 / 45 | 0.000015 | 0.000016 | 0.001259 |

下表为按 index 对齐的输出文本 hash 相同比例，每项比较 393 个请求；完整 token 数一致比例在 aggregate 中。它衡量数值输出差异，不衡量答案准确率。

| 到达 RPS | Preble 两重复间 | RR 两重复间 | 跨策略 rep1 | 跨策略 rep2 |
| --- | --- | --- | --- | --- |
| 0.7 | 77.61% | 77.61% | 68.19% | 63.10% |
| 1.4 | 73.54% | 63.36% | 65.39% | 66.67% |

## 固定环境

- 执行位置：3 号机 1727，容器 `sglang_chenjiefei_ds4`。原有 8000 服务保持运行，实验 worker 使用 31000 / 31001，frontend 使用 18000。
- GPU：8×NVIDIA H200，每卡 143771 MiB，driver 595.71.05；两份独立 TP4 副本。
- 模型：`Qwen/Qwen3-235B-A22B-Instruct-2507`，完整 BF16，未量化、非 dummy weights。公共 revision 为 `ac9c66cc9b46af7306746a9250f23d47083d689e`，本地 snapshot 名为 `56e16a623ffb2855ca901a65166a9170e99df127`。
- 模型核验：前轮在 2026-10-14 已检查全部 118 个 shard 的 LFS SHA 与 36945 个 BF16 tensor header；本次复用该 checkpoint，没有把全量 payload hash 再跑一遍。配置、tokenizer 与历史核验报告的 hash 在公开 protocol 中。局部 tokenizer 配置的 chat template / 最大长度差异不用于本次 token-ID 请求。
- SGLang base：`ef11e3904885c8050f167a4a6f805da03e9e9bae`。正式 frontend / replay 使用 commit `1fb127f033`；MPQ worker 启动 commit 为 `24dc393f7e`，LPM 为 `1fb127f033`。这段差异仅涉及 frontend / global router / benchmark，worker 本地 runtime 文件不变；protocol 提供 git blob 对照。
- Worker：context 32768、static memory fraction 0.90、max running 64、chunked prefill 512、Triton attention、FlashInfer CUTLASS MoE、关闭 CUDA graph。策略分别为 `fcfs-mpq` / `lpm`；原生 LPM 保留当前 runtime 的队列长度超过 128 时回退 FCFS 的规则。
- 每个副本的 KV token 池均为 **328506**，weight 109.71 GB、KV 14.72 GB / GPU。TP4 token 容量不能再乘 4。
- 有效随机种子固定为 worker 0 `659938497`、worker 1 `556540270`；LPM 重启时显式传入原 MPQ 自动解析的值。两 phase 的 server_info 静态配置逐项核对，只有 `schedule_policy` 不同。
- Python 3.13.13；torch 2.11.0+cu128、triton 3.6.0、sglang-kernel 0.4.4+cu129、transformers 5.8.1、FlashInfer Python / cubin 0.6.12、JIT cache 0.6.12+cu128、TVM FFI 0.1.9、CUTLASS DSL 4.5.2、NCCL 2.28.9。

两个 phase 都在正式重放前通过四条真实生成语义检查：17×19 → `323`、法国首都 → `Paris`，两端均正确。这验证推理运行正常，不是 LooGLE 答案准确率评价。协议用 localhost fake HTTP 的验证不加载模型；所有真实模型推理均使用上述 235B checkpoint。

## 工作负载与校准

正式 workload 直接执行固定原版生成器的 AST：原 prompt 模板、QA 比例复制、两次 shuffle、head/tail decode 后重新 tokenize 的截断规则均保留。输入为旧 schema 的 `shortdep_qa` parquet，而不是把当前 JSONL 当作原 loader。

| 项目 | 值 |
| --- | --- |
| 数据集 | `bigai-nlco/LooGLE`，revision `bdc356116a09a6a3d552e28a9f330878915b1696` |
| Parquet SHA-256 | `065e2036eada3513c02835137a4c83e483fbfbf8922740b7f14c6a22542dbc8e` |
| 规模 | 24 文档、393 请求；0.7 / 1.4 RPS，每档两次重复 |
| Prompt tokens | min 13559、mean 21103.532、max 32722 |
| 输出 | `max_new_tokens=45`、temperature 0；实际输出数量见结果 |
| Python / NumPy seed | 10；独立 UUIDv4 seed 10；arrival RNG 显式重置 seed 10 |
| Workload SHA-256 | `c7b08adf0d9a9f5763ff26c7e7d2508020e3c5cb2ecc4a4ae0c531cecb50dc54` |
| 0.7 arrival SHA-256 | `512c9c8d3144e47202637b022433bca671e4e843761f15726a28cba8376781cc` |
| 1.4 arrival SHA-256 | `4d828135a0c485bb1c4559d4df00133dc46ddf4b5f93b98c353038730e299c40` |

原公开两 worker LooGLE 配置包含 24 / 393 / 0.7 RPS / 45-token 参数，使用 Mistral-7B、`simulate=True` 和 dummy weights。本次将其工作负载放到真实 235B 上，并增加 1.4 RPS 档。固定 UUID 与 RNG 边界用于当前环境重现，不能恢复论文未归档的逐请求随机 trace。公开准备脚本已在 1727 CPU 上重生成，workload 与两份 arrival 文件逐字节一致。

Prefill 仍是原版二维 `T(new_tokens, full_context)`，只整体乘一个硬件 scale；没有改成线性 tokens/rate。正式实验前于 **04:34:34 UTC** 冻结：

```text
prefill_time_scale = 2.387698396256729
initial_tpot      = 0.06731290138453584 秒 / output token
calibration SHA   = b9a93683d8e47498af24030166e9ec17ab6ab5bc7493798e10c873057a066fc0
```

每端六种 profile：cold 8192 / 16384 / 32722；16384 cached + 512 / 2048 tail；30674 cached + 2048 tail。先完整 warmup 一遍，再独立测量一遍；取 12 个 `TTFT / T(实际未命中 tokens, 完整 context)` 的中位数为 scale，取实际 `(E2E−TTFT)/(output_tokens−1)` 中位数为初始 TPOT。最长 profile 因 context 上限可能只完成 44 个输出 token，统计使用实际值。

**校准输入与正式测试轨迹不同。** 校准取此前 seed42 JSONL 轨迹中第一个最长 prompt，workload SHA 为 `b9fbf417c3f588e7b92dd134fff42eead61b7bf3aa10ff3c7020db2f96b74d68`；该输入来自当前 schema 的 LooGLE revision `4b50b4cb9333b48b3f7ddfd307c1d13369be5d1e`，dataset SHA 为 `8951521e8c8340a4e6aa5ffcb2e8f058cca154251915acf946e721fcf001e8c7`。这是独立校准，不作为正式性能样本；两方正式测试都使用上表 seed10 的同一份原 AST 轨迹。LPM 重启后重做相同六 profile 的 warmup，不重新拟合或覆盖参数。

## 执行与统计方法

每轮从两端空闲开始，POST `/flush_cache?timeout=30` 成功后才发请求；每轮 Preble frontend 都是新进程、空全局树和历史。两方 workload、顺序、固定 arrival 文件相同，客户端 HTTP 并发上限 256。Preble 包含 frontend 转发开销；RR 直接按 `index % 2` 选 worker。

实际顺序固定为：Preble 0.7 rep1 / rep2 → 1.4 rep1 / rep2 → 重启 LPM → 同 profile warmup → RR 1.4 rep1 / rep2 → 0.7 rep1 / rep2。**两策略没有跨时间交替**，阶段效应仍可能影响比较。一次在请求发出前发生的端口预检查 TIME_WAIT 误报已单独归档，修正 SO_REUSEADDR 后继续；有效首轮保留，未把失败预检查当性能样本。phase / execution / runner hash 分别记录，避免掩盖脚本操作版本变化。

TTFT / E2E 从计划到达的 sleep 醒后计时，包含 semaphore 等待、HTTP、排队、计算与 frontend 开销；不包含 sleep 超时的唤醒迟延。客户端等待单独报告，超过 256 个并发时实际 HTTP 提交会受它限制。cache fraction 按 `Σcached_tokens / Σprompt_tokens` 加权。吞吐是成功请求 / 输出 token 数除以整轮时长，分母包括到达等待与尾部排空，不代表饱和吞吐。

0.7 / 1.4 RPS 是指数到达分布的参数；这份有限轨迹的最后到达分别为 512.937 / 256.469 秒，393 条请求对应的实际到达密度约为 0.766 / 1.532 请求每秒。因此，0.7 档的完成 RPS 可以高于 0.7，表格始终使用实际整轮时长。32722-token 的最长 prompt 会被当前 runtime 将输出上限从 45 收紧到 44；分析器按 context、token 池与 page 预算的实际公式检查有效上限，`length` 完成必须恰好达到该上限。

每个 policy / rate 合并两轮的 **786 条原始请求**计算均值与线性插值分位数，没有平均两轮 p95。逐轮数据保留以显示波动；这些请求共享文档前缀和实验条件，不能视为 786 次独立实验。输出 hash / token 数按请求 index 对齐比较；温度零在不同 batch 形状下仍可有数值差异，不将 hash 相同当作生成质量或性能有效性的必要条件。

## 源码验证与边界

- 全局差分：36 traces，2 / 3 / 4 workers、eviction 开 / 关、窗口 expiry；9012 次 select / finish 调用，完整状态与固定原源码一致。9006 次 finish 正常、6 次双方触发同一原断言。migration / replication 定向路径覆盖；保留原缺陷的含义见[实现文档](README.md)。
- 本地 MPQ：5000 次原源码逐决策差分，另有真实 `Req` / radix API 的 CPU 准入验证。
- Frontend：29 项 localhost HTTP 检查通过；固定 arrival 的真实 HTTP 重放与非法输入拒绝通过。公开校准入口的两端真实 HTTP 协议验证通过；未提交单元测试。
- 实际 GPU 运行的 migration / replication 次数见结果。没有触发的分支只声明源码 / CPU 覆盖，不声明已在该 GPU trace 上复现。未做单独 eviction / migration 消融，不能把全部收益归因于某一个模块。
- 原模拟器 hit-rate hint 不在当前公开服务接口范围内；实际 batch 的预算和 chunk 控制沿用现代 SGLang。两方同为 chunk 512，这不是原模拟器全部运行参数的逐项复制。

## 公开证据与复现

[实现与运行方法](README.md) 包含 worker、frontend、固定原生成器与 `--arrivals` 重放命令。独立校准输入可用现有适配生成器重建：

```bash
curl --fail --location \
  https://huggingface.co/datasets/bigai-nlco/LooGLE/resolve/4b50b4cb9333b48b3f7ddfd307c1d13369be5d1e/data/shortdep_qa.jsonl \
  --output preble-bench/calibration-source.jsonl
python benchmark/preble/bench_loogle.py --prepare-only \
  --dataset preble-bench/calibration-source.jsonl --model-path "$PREBLE_MODEL_PATH" \
  --workload preble-bench/calibration-workload.jsonl \
  --patterns 24 --requests 393 --max-input-tokens 32723 --max-new-tokens 45 --seed 42
python benchmark/preble/calibrate.py \
  --workload preble-bench/calibration-workload.jsonl \
  --worker-urls http://127.0.0.1:31000 http://127.0.0.1:31001 \
  --max-new-tokens 45 --output preble-bench/calibration.json
```

新测量会受机器状态影响，不保证逐数值等于本次已冻结标定；复现正式轨迹前应先核验 workload / arrivals 与上表 hash，再冻结自己的参数。

本仓库公开紧凑证据，不包含模型权重或 78 MB 的原 prompt 轨迹：

- [aggregate.json](results/reference-235b/aggregate.json)：逐轮 / 合并指标、原文件 SHA、配置校验、输出一致性与统计边界。
- [requests.jsonl.gz](results/reference-235b/requests.jsonl.gz)：全部逐请求指标及输出 hash，逐行 `{trial, request}`，不含 prompt 或输出全文。
- [protocol.json](results/reference-235b/protocol.json)：版本、命令、环境、模型核验来源、生成器与校准输入溯源。
- [calibration.json](results/reference-235b/calibration.json)：冻结的 12 个样本及参数。
- [validation.json](results/reference-235b/validation.json)：源码差分与 HTTP 验证报告，包含共有原断言失败。

完整原始执行记录保留在本地 `preble-results/reference-1727-20261015`：每轮的 flush、命令、逐请求 JSONL、summary、完成标记，Preble 另有最终 routing_stats；两 phase 的启动、server_info、语义检查与 warmup 分别留档。
