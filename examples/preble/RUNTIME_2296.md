# 2296 235B 对照验证

运行版本 [`091d70e1e7`](https://github.com/nannaer/sglang-sparse-prefix/commit/091d70e1e73955a2cd4f26bc5e9e7c9acf519db5)，Preble 参考 `1a35eae20c`。**Preble + MPQ 与 RR + LPM 各一轮**，先 Preble 后 RR。

最新零输出完成反馈另由[源码差分](results/reaudit-20261015/zero-count-feedback.json)覆盖；本轮 Preble 最短输出 22 tokens，不覆盖该边界。

## 结果

两轮均 **393/393 成功**，各 393 个 TTFT。时延单位为秒。

| 指标 | Preble + MPQ | RR + LPM |
| --- | ---: | ---: |
| TTFT mean / p50 / p95 | 28.313 / 18.765 / 80.285 | 42.322 / 21.577 / 147.955 |
| 端到端 mean / p50 / p95 | 62.435 / 57.411 / 126.287 | 76.554 / 62.704 / 171.768 |
| 整轮用时 | 279.524 | 329.190 |
| 完成请求/秒 | 1.406 | 1.194 |
| 输入缓存比例 | 83.539% | 81.048% |
| 输出 token | 17,576 | 17,498 |
| worker0 / worker1 请求数 | 184 / 209 | 197 / 196 |

本次 Preble 平均 TTFT / p95 TTFT 降低 **33.1% / 45.7%**，平均 / p95 端到端时延降低 **18.4% / 26.5%**。Preble 最终 `active=[0,0]`，migration 95、replication 0；migration 只调整未来归属，replication 的覆盖来自源码差分。RR 严格按请求 index % 2 直连 worker，使用本地 LPM。

两轮日志均未见错误、OOM 或 retract，自有进程已清理，8 张 GPU 恢复空闲。逐请求重算时延、分位数、缓存比例及 RPS 与汇总一致；[比较数据](results/runtime-2296/comparison.json)保留计算结果与来源 hash。

## 配置与执行

- **2 号机 2296，容器 sglang_chenjiefei_ds4**；8×H200，两个独立 TP4。
- **Qwen3-235B-A22B-Instruct-2507，完整 BF16、未量化**，snapshot `56e16a623ffb2855ca901a65166a9170e99df127`，权重位于开发目录外；核验了全部 118 分片的 header / 尺寸与索引，未对 470GB payload 重新计算 hash。
- context 32768、memory fraction 0.90、max running 64、chunk 512；Triton attention、FlashInfer CUTLASS MoE、关闭 CUDA graph。两端 seed 为 659938497 / 556540270。
- 实际每副本 KV 容量 **328505 tokens**，frontend 使用相同值；旧 1727 为 328506，不为凑数修改内存策略。
- 原版生成器 AST + 旧 schema LooGLE：24 文档、393 请求，输入 13559–32722 tokens，最大输出 45，temperature 0；固定 1.4 RPS arrival 文件，最后计划到达 256.469 秒，并发上限 256。
- 每轮两端完整 ready 日志出现后，才执行每端相同六 profile warmup；沿用冻结 prefill scale `2.387698396256729`、initial TPOT `0.06731290138453584`，不重新拟合。正式回放前 flush 两端 KV，Preble frontend 从空树开始。两轮 worker 命令仅调度策略不同；runtime 缓存目录分别隔离，底层编译缓存保留。

TTFT / 端到端时延从 arrival sleep 醒后起钟，含 semaphore、HTTP、frontend 与计算，不含唤醒迟延。完成 RPS 的分母包含到达等待和尾部排空。输入缓存比例为总 cached_tokens / 总 prompt_tokens。

12 对预热样本的输入/输出 hash、token/cache 计数一致；六个冷 prefill TTFT 差异均小于 0.5%，18K 带 16K 前缀 profile 的 RR TTFT 最多快 7.8%。正式回放 273 条输出 hash 相同、120 条不同，其中四条长度不同、RR 共少 78 tokens（0.444%）；这四条的 RR 均正常命中 EOS151643。不同 batch 下不能保证贪心输出相同，未保存全文/token 序列，不能确定输出差异原因。

## 证据与限制

- Preble：[逐请求](results/runtime-2296/requests.jsonl.gz)、[汇总](results/runtime-2296/summary.json)、[协议](results/runtime-2296/protocol.json)、[预热](results/runtime-2296/warmup.json)、[核验](results/runtime-2296/audit.json)、[遥测](results/runtime-2296/telemetry.jsonl.gz)。
- RR：[逐请求](results/runtime-2296/rr-lpm/requests.jsonl.gz)、[汇总](results/runtime-2296/rr-lpm/summary.json)、[协议](results/runtime-2296/rr-lpm/protocol.json)、[预热](results/runtime-2296/rr-lpm/warmup.json)、[核验](results/runtime-2296/rr-lpm/audit.json)、[遥测](results/runtime-2296/rr-lpm/telemetry.jsonl.gz)。同一[模型核验](results/runtime-2296/model-provenance.json)，另有独立[协议审查](results/runtime-2296/protocol-review.json)。逐请求证据保留客户端时钟与输出 hash，不含 prompt / 输出全文。
- 额外 frontend select / finish 精确耗时未落盘，不据此推断路由开销；audit 明确保留此缺失。客户端时间、路由统计及 15 秒系统 / GPU 遥测完整保存。
- 这是各一次、固定先后顺序的组合方案比较，同时改变全局路由、本地调度及 frontend 路径；不隔离单项收益，也不证明稳定收益或[论文](https://arxiv.org/abs/2407.00023)数值复现。预热与遥测不能证明两轮全程硬件状态相同。旧两轮差异见[简短结论](VARIANCE_ANALYSIS.md)。

启动与回放命令见[运行方法](README.md)。原始记录位于 `preble-results/reaudit-runtime-235b-2296-20261015-ready-gated` 与 `preble-results/reaudit-runtime-235b-2296-20261015-rr-lpm-ready-gated`。
