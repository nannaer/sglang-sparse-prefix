# Preble 原版对齐后的 235B 补充实验

本次测量使用 runtime [`c773f3e560`](https://github.com/nannaer/sglang-sparse-prefix/commit/c773f3e5605eb3fe9a2d032c80f0e9def6dfc2d9)，比较 **Preble + MPQ** 与 **RR + LPM**。参考仍固定为 WukLab/Preble `1a35eae20c416f7062ca81775aab72890323fd97` 的公开服务默认路径，SGLang base 未变。

## 结果

四轮共 **1572/1572 请求成功**，两轮 Preble 完成后均 `active=[0,0]`。配置、输入 hash、实际输出上限和最终清理检查全部通过；8 张实验 GPU 已释放。以下每策略合并 786 条原始请求，时延单位为秒。

| 策略 | 请求数 | TTFT mean | TTFT p95 | E2E mean | E2E p95 | Cache | 完成 RPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Preble + MPQ | 786 | 21.346 | 61.101 | 51.428 | 112.332 | 83.301% | 1.325 |
| RR + LPM | 786 | 34.717 | 147.657 | 65.251 | 174.779 | 79.120% | 1.233 |

本次固定 1.4 RPS 轨迹上，Preble 的平均 TTFT / E2E 分别降低 **38.5% / 21.2%**，p95 分别降低 58.6% / 35.7%。TTFT 中位数则为 Preble 14.695 秒、RR 11.673 秒。

这是单一模型、单一到达档和两次重复的描述性结果，不等同论文数值复现，也未证明跨负载稳定收益。Preble 两轮间波动仍明显，全部轮次如下。

| 次序 | 策略 / 重复 | 成功 | TTFT mean | TTFT p95 | E2E mean | E2E p95 | Cache |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | Preble + MPQ / rep1 | 393/393 | 28.390 | 80.105 | 64.747 | 123.919 | 84.525% |
| B1 | RR + LPM / rep1 | 393/393 | 35.062 | 141.736 | 66.466 | 173.259 | 79.341% |
| B2 | RR + LPM / rep2 | 393/393 | 34.372 | 152.464 | 64.036 | 175.076 | 78.899% |
| A2 | Preble + MPQ / rep2 | 393/393 | 14.301 | 42.062 | 38.109 | 92.337 | 82.077% |

A1 / A2 分别有 **58 / 110 次 migration**、0 次 replication；worker 0 / 1 请求数分别为 **169 / 224**、**221 / 172**。Migration 调整未来请求归属，不搬运 KV；本次 GPU trace 未触发 replication，该分支只有源码 / CPU 定向覆盖。

B2 的 worker 1 出现一次正常 KV-pool retraction，四个 TP rank 各记一行，合计一次事件；释放 KV 并重计算后全部请求完成。最终日志审计见 [runtime-events.json](results/runtime-235b/runtime-events.json)。

输出一致性、实际 token 数及客户端等待统计保留在 aggregate 与逐请求数据中。温度为零时，不同 batch 形状仍会产生数值与文本差异；本次没有评价 LooGLE 答案准确率。

## 本轮修正及验证

此前验证隔离了 MPQ 的分组与轮转逻辑，没有执行完整准入预算和时间戳 IPC。本轮补齐以下差异，五项各自提交；没有通过改变原算法调性能。

| 提交 | 修正 |
| --- | --- |
| [`4a04480751`](https://github.com/nannaer/sglang-sparse-prefix/commit/4a04480751) | 在线 TPOT 按原公开 HTTP 路径除以实际 `output_len`；单 token 输出不追加样本 |
| [`844ce09c8c`](https://github.com/nannaer/sglang-sparse-prefix/commit/844ce09c8c) | MPQ 普通容量不足时继续扫描同组后项，锁后容量不足仍停止本轮 |
| [`47858bc59e`](https://github.com/nannaer/sglang-sparse-prefix/commit/47858bc59e) | FIFO 使用分词前到达时间，跨直接 / DP 两跳 IPC 保存，retract 不重设 |
| [`dda4ec78e8`](https://github.com/nannaer/sglang-sparse-prefix/commit/dda4ec78e8) | 先处理续 chunk，有预算才选择 waiting 组；chunk-only 不消耗轮转 |
| [`c773f3e560`](https://github.com/nannaer/sglang-sparse-prefix/commit/c773f3e560) | 准入、取消、超时等离队路径释放 MPQ membership 和候选快照引用 |

源码验证结果：

- 全局差分：36 traces、9012 次选择及 9012 次完成尝试，完整状态与原源码一致。9006 次正常完成，6 次双方触发同一原断言；另有 migration / replication、50 个时间模型组合及 516 项部分可用性检查。
- 准入 / chunk：执行真实 `SchedulePolicy`、`PrefillAdder`、Scheduler raw 方法的 AST，**41 项通过**。旧行为负对照能抓到 chunk-only 错误推进 cursor。
- FIFO：真实 pickle 与跨进程时钟转换、反序分词完成、同批时间相同、retract 等 **8 项通过**。
- 原 HTTP→原 finish→当前 finish：输出 1 / 2 / 45 tokens **3 个场景通过**。
- 等待取消生命周期：另一套 **41 项通过**，含无后续调度时 weakref / GC 释放及两个旧行为负对照。

后四项是 CPU 源码语义检查，GPU、allocator 和通信等基础设施使用桩，未覆盖全部 LoRA、PD、SWA / Mamba 组合。235B 实验验证下面固定配置的端到端运行。原版候选 ID、虚拟驱逐重复计数和特定 split / expiry 断言缺陷仍保留；现代页分配、slot、单 chunk 和 retract 控制流是版本适配，不能声称与旧 runtime 逐 batch 相同。完整伪代码与边界见[实现文档](README.md)。验证脚本保留在仓库外，不提交单元测试。

## 固定配置与执行协议

- 位置：3 号机 1727，容器 `sglang_chenjiefei_ds4`；8×H200，两个独立 TP4 副本。
- 模型：**Qwen3-235B-A22B-Instruct-2507，完整 BF16、未量化**。复用已验证的 118 个 shard checkpoint；模型核验溯源、软件版本、完整命令见 [protocol.json](results/runtime-235b/protocol.json)。本轮没有重新校验全部权重 payload。
- 两端 worker、frontend、replay 均固定 `c773f3e560`。context 32768、memory fraction 0.90、max running 64、chunk 512、Triton attention、FlashInfer CUTLASS MoE、关闭 CUDA graph。
- 每副本实际 KV token 容量 **328506**；TP4 容量不乘 4。worker 0 / 1 的固定 seed 分别是 `659938497` / `556540270`。三个 worker session 的静态配置与实际容量均核验。
- 工作负载：原生成器 AST + 旧 schema LooGLE，24 文档、393 请求，prompt 13559–32722 tokens，输出上限 45，temperature 0。数据 / 生成器的 revision、随机流和校准输入沿用[旧实验的溯源](RESULTS.md#工作负载与校准)。

```text
workload SHA-256 = c7b08adf0d9a9f5763ff26c7e7d2508020e3c5cb2ecc4a4ae0c531cecb50dc54
arrivals SHA-256 = 4d828135a0c485bb1c4559d4df00133dc46ddf4b5f93b98c353038730e299c40
prefill scale   = 2.387698396256729
initial TPOT    = 0.06731290138453584 s/token
```

本轮只测 **1.4 RPS**，不把旧版 0.7 RPS 结果当作修正后的测量。1.4 是指数到达分布参数；固定有限轨迹最后到达为 256.469 秒，对应约 1.532 请求/秒。四轮重复使用逐字节相同的 workload / arrivals。正式轨迹 seed 为 10；warmup 使用独立 seed42 输入。replay 的默认 seed42 不改变已固定的 workload / arrivals。

执行次序为 **A1 Preble → B1 RR → B2 RR → A2 Preble（ABBA）**。从 MPQ 切到 LPM、再切回 MPQ，各自重启两个 worker；每次新 session 在两端执行相同六 profile warmup，不重新拟合时间模型。B1 / B2 复用同一 LPM worker 进程，B2 前只 flush，不重新 warmup 或 reseed；两次 Preble 分别使用新 MPQ session。每轮前两端空闲并 flush；Preble 每轮新建 frontend、空全局树。这里的冷 cache 指 KV 清空，编译 cache 保留，启动 / warmup 时间不计入 replay 时长。时间参数在上轮正式实验前已冻结，本轮沿用原值。校准中的统计 TPOT 与原公开服务的在线反馈分母有区别，后者已按上表修正。

ABBA 用于减轻执行时间顺序的影响，只有一次 ABBA、每策略两次重复，仍不能排除机器状态或非线性阶段效应。本轮既修改了运行语义，又改变了旧版按策略阶段执行的协议，因此前后结果差不能单独归因某个修复。

TTFT / E2E 从 arrival sleep 醒后计时，含 semaphore、HTTP、frontend、排队与计算，不含 sleep 唤醒迟延。客户端并发上限 256，RR 按 `index % 2` 直连两端，Preble 包含 frontend 开销。两方同时改变全局与本地调度，没有模块消融。实际输出按 context 预算核验；最长 prompt 可能将输出上限收紧至 44。

每策略合并两轮 **786 条原始请求**计算 mean 和线性插值 p95，没有平均两轮 p95。cache 是 `Σcached_tokens / Σprompt_tokens`，完成 RPS 是成功请求数除以该策略两轮 replay 时长之和，包括到达等待和尾部排空，不代表饱和吞吐。这些请求共享文档前缀和条件，不能当作 786 次独立实验。

## 证据与复现

- [aggregate.json](results/runtime-235b/aggregate.json)：逐轮、合并指标、输出一致性、输入与原文件 hash、配置检查。
- [requests.jsonl.gz](results/runtime-235b/requests.jsonl.gz)：1572 条逐请求指标及输出 hash，不包含 prompt / 输出全文。
- [protocol.json](results/runtime-235b/protocol.json)：固定版本、启动命令、环境、warmup、冷 cache 和进程清理证据。
- [calibration.json](results/runtime-235b/calibration.json)：沿用的冻结参数与本轮 warmup。
- [validation.json](results/runtime-235b/validation.json)：源码检查、负对照与覆盖限制。
- [runtime-events.json](results/runtime-235b/runtime-events.json)：六份 worker 日志的 hash、去重 retract 事件及原始行。

复现入口见[启动与轨迹准备](README.md#启动两份真实-235b-副本)。使用 `arrivals-r1.4.json` 与 `--rate 1.4`，按上述 ABBA 顺序执行，并固定新 session 的 seeds / 容量 / 校准参数。每轮单独输出目录，保留逐请求与完成标记。

原始记录保留在本地 `preble-results/reference-runtime-235b-1727-20261015`。旧版八轮 3144 请求的结果继续保留在[历史报告](RESULTS.md)，其 runtime 和执行协议不同，不作为本轮五项修正的 GPU 验证。
