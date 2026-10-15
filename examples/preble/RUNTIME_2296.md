# 2296 单轮 235B 验证

运行版本 [`091d70e1e7`](https://github.com/nannaer/sglang-sparse-prefix/commit/091d70e1e73955a2cd4f26bc5e9e7c9acf519db5)，Preble 参考 `1a35eae20c`。本次只运行 **一轮 Preble + MPQ**，没有 RR 对照或第二轮。

## 结果

**393/393 成功**，393 个 TTFT，最终 `active=[0,0]`。时延单位为秒。

| 指标 | Mean | p50 | p95 |
| --- | ---: | ---: | ---: |
| TTFT | 28.313 | 18.765 | 80.285 |
| 端到端时延 | 62.435 | 57.411 | 126.287 |

整轮 279.524 秒，完成 1.406 请求/秒，输出 17,576 tokens；输入缓存比例 83.539%。两端分别处理 184 / 209 请求，migration 95、replication 0。Migration 只调整未来归属；本轮未触发 replication，其覆盖来自源码差分。

日志未见错误、OOM 或 retract。已清理本轮自有进程，8 张 GPU 回到空闲，保留其他服务。逐请求重算 mean / p95 / cache 与汇总一致。

## 配置与执行

- **2 号机 2296，容器 sglang_chenjiefei_ds4**；8×H200，两个独立 TP4。
- **Qwen3-235B-A22B-Instruct-2507，完整 BF16、未量化**，snapshot `56e16a623ffb2855ca901a65166a9170e99df127`，权重位于开发目录外；核验了全部 118 分片的 header / 尺寸与索引，未对 470GB payload 重新计算 hash。
- context 32768、memory fraction 0.90、max running 64、chunk 512；Triton attention、FlashInfer CUTLASS MoE、关闭 CUDA graph。两端 seed 为 659938497 / 556540270。
- 实际每副本 KV 容量 **328505 tokens**，frontend 使用相同值；旧 1727 为 328506，不为凑数修改内存策略。
- 原版生成器 AST + 旧 schema LooGLE：24 文档、393 请求，输入 13559–32722 tokens，最大输出 45，temperature 0；固定 1.4 RPS arrival 文件，最后计划到达 256.469 秒，并发上限 256。
- 两端完整 ready 日志出现后，才执行相同六 profile warmup；沿用冻结 prefill scale `2.387698396256729`、initial TPOT `0.06731290138453584`，不重新拟合。正式回放前 flush 两端 KV，frontend 从空树开始。

TTFT / 端到端时延从 arrival sleep 醒后起钟，含 semaphore、HTTP、frontend 与计算，不含唤醒迟延。完成 RPS 的分母包含到达等待和尾部排空。输入缓存比例为总 cached_tokens / 总 prompt_tokens。

## 证据与限制

- [逐请求指标](results/runtime-2296/requests.jsonl.gz)、[汇总](results/runtime-2296/summary.json)：保留客户端计划 / 实际唤醒 / 发送 / 首 token / 完成时钟与输出 hash，不含 prompt / 输出全文。
- [执行协议与 hash](results/runtime-2296/protocol.json)、[模型核验](results/runtime-2296/model-provenance.json)、[运行核验](results/runtime-2296/audit.json)、[系统遥测](results/runtime-2296/telemetry.jsonl.gz)。
- 额外 frontend select / finish 精确耗时未落盘，不据此推断路由开销；audit 明确保留此缺失。客户端时间、路由统计及 15 秒系统 / GPU 遥测完整保存。
- 这次验证固定配置能够端到端运行；单轮且无新对照，不证明稳定性能收益，也不等同[论文](https://arxiv.org/abs/2407.00023)数值复现。旧两轮差异的已确认事实与根因限制见[简短结论](VARIANCE_ANALYSIS.md)。

启动与回放命令见[运行方法](README.md)。完整原始记录位于 `preble-results/reaudit-runtime-235b-2296-20261015-ready-gated`。
