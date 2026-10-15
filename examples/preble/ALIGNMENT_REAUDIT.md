# Preble 一致性验证

当前 runtime：[`9e289eb768`](https://github.com/nannaer/sglang-sparse-prefix/commit/9e289eb76877558e5eea5e2506e04910e372ea32)。参考：[WukLab/Preble `1a35eae20c`](https://github.com/WukLab/preble/tree/1a35eae20c416f7062ca81775aab72890323fd97)。SGLang base：`ef11e3904885c8050f167a4a6f805da03e9e9bae`。

## 对齐结论

已复核原公开服务默认路径：全局 E2、180 秒窗口负载、二维 prefill 模型、虚拟 LRU 驱逐代价、migration / replication、持久 MPQ、FIFO / quota / 准入顺序，以及普通 MPQ retract 顺序。HTTP 默认流式、反馈计时和成功预算记账也按原路径处理。

| 模块 | 保留的精确行为 |
| --- | --- |
| 候选选择 | 健康池保留原归属 set 迭代顺序，以及原 argmin 返回候选位置的缺陷。 |
| HTTP 反馈 | 路由后起钟；转发帧后记时；首个 JSON 即 TTFT，包括 completion_tokens=0；latency 截止最后 JSON，不计 DONE / close。省略 stream 默认 True。 |
| TPOT 与零输出 | 成功反馈按 `(latency-TTFT)/max(1,output_len)` 计算；输出非 1 且 TPOT 非零才采样。成功零输出也保留原 histogram 断言；零输出错误 / 取消走现代清理。 |
| 完成记账 | 原公开 HTTP 成功反馈预算为 0，旧 decode ledger 不扣实际预算；active / 路径引用仍释放。实际路由 / 重平衡读活跃引用，不读此 ledger。失败 / 取消释放实际预算。 |
| MPQ retract | 非 speculative 路径按 `(arrival_time, output_len)` 升序后 pop，即晚到先撤，同时间输出多者先撤，完全相同则按 batch 反序。 |

235B 被测版本为 `091d70e1e7`；当前唯一生产增量是零输出完成反馈，使用受控 HTTP 响应做源码差分，未作为 GPU 性能测量。

## 验证结果

在 **2296 / sglang_chenjiefei_ds4 / runtime Python 3.13.13** 中执行原源码与当前源码 / AST 差分；HTTP 使用实际 FastAPI / StreamingResponse，基础设施受控。

| 验证 | 结果 |
| --- | --- |
| 全局状态 | 36 traces、9012 select + 9012 finish 尝试；9006 正常完成，6 次双方同一原断言，状态一致。覆盖 2 / 3 / 4 worker、eviction 开关、split / expiry，浮点归一至小数点后 11 位。 |
| 全局定向 | migration / replication 各覆盖 2 / 3 / 4 worker；50 个时间模型组合、516 项部分可用映射通过。 |
| 候选顺序 | 960 项健康池、1200 项部分可用池通过；修前 230 项健康池差异被检出。 |
| Retract | 306 项与原版比较 victim 顺序；1200 项确认非 MPQ / speculative 路径不变；1 项现代最后请求 abort 保护；2 个修前负对照检出差异。 |
| HTTP 计时 | 24 个成功差分、24 个修前负对照、7 个清理场景、1 个默认流式背压对照通过。 |
| HTTP 预算 | 6 个成功差分、6 个修前负对照、2 个低层实际预算对照、6 个失败 / 取消 / 幂等场景通过。 |
| 零输出反馈 | 48 项正常反馈差分通过，覆盖输出 0 / 1 / 2 / 45、budget 0 / 45、零 / 正间隔及三种 stream 设置；4 项复现原版同一断言；24 项错误 / 取消 / 幂等控制保持。修前 6 项 TPOT、4 项断言负对照检出。 |
| 既有准入验证 | 真实准入 / chunk 41 项、FIFO / IPC 8 项、取消生命周期 41 项通过，见[先前证据](results/runtime-235b/validation.json)；覆盖方法本轮未改，retract 顺序另按上表验证。 |

[既有验证数据与 SHA256](results/reaudit-20261015/semantic-validation.json)、[零输出反馈验证](results/reaudit-20261015/zero-count-feedback.json)及[完整差分记录](results/reaudit-20261015/zero-count-feedback.json.gz)。脚本保留在仓库外，未提交单元测试。以上是 CPU 语义验证，真实大模型运行见 [2296 对照结果](RUNTIME_2296.md)。

## 必要适配与边界

当前 Req、页级 KV / slot 分配、对齐、single continuing chunk、SWA / Mamba 校验与物理 KV 释放沿用现代 runtime；retract 保留已生成 token。实际准入和执行时机不承诺与旧 runtime 逐 batch 相同。

原公开服务 highest_idx=None；模拟器可选 hint 未暴露。部分 worker 不可用过滤、显式非流式和取消清理是现代服务扩展。保留原候选位置缺陷、虚拟 LRU 重复父项及特定 split / expiry 断言；共同断言不算成功请求。因此一致性结论限定为上述公开路径与验证覆盖。
