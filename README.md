# SGLang Prefix Load Balancing

在当前 SGLang 上接入 Preble：按照 prompt 前缀、预计计算负载与驱逐重算代价，把请求分配给独立模型副本；副本内部使用 MPQ 调度等待请求。

项目：[`nannaer/sglang-sparse-prefix`](https://github.com/nannaer/sglang-sparse-prefix)。原设计来自 [Preble 论文](https://arxiv.org/abs/2407.00023) 与 [WukLab/Preble](https://github.com/WukLab/preble)，底层 runtime 来自 [SGLang](https://github.com/sgl-project/sglang)。

## 一次请求怎么运行

```text
客户端发送 prompt
    ↓
Frontend：tokenize → 更新全局前缀树 → 估计候选副本代价 → 选择副本
    ↓
Worker：进入持久 MPQ 分组 → 按组轮转与 FIFO 准入 → SGLang 生成
    ↓
完成反馈：释放引用，记录 TTFT，用输出长度与时延更新 TPOT
```

全局路由同时考虑三件事：前缀目前分配给谁、各副本还有多少工作、放入新请求后可能驱逐哪些前缀。可复用前缀较长时在既有归属中选；否则探索全部副本，选择估计代价最低的一端。

```text
route(prompt):
    leaf = 全局压缩前缀树.insert(prompt)
    candidates = 既有归属 if leaf 段短于它之前的 context else 全部副本
    cost[w] = 窗口 prefill 代价 + 在途 decode 代价 + 虚拟驱逐重算代价
    worker = 原版最小代价选择(candidates, cost)
    更新引用、虚拟缓存与窗口；满足条件时调整未来路由归属
    forward(worker, 相同的 prompt token IDs)
```

每个 worker 的真正 KV 留在自己的 HBM。全局虚拟缓存用于预测代价，不是共享 KV 存储；这里的前缀迁移 / 复制是调整未来请求归属，新副本通过后续请求计算并建立缓存。

## 是否与 Preble 一致

对齐目标是原版**公开服务默认路径**，参考 commit 固定为 `1a35eae20c416f7062ca81775aab72890323fd97`。SGLang 保持 base `ef11e3904885c8050f167a4a6f805da03e9e9bae`，没有降级或导入旧 runtime。

| 模块 | 当前实现 |
| --- | --- |
| 全局 E2 | 原版节点段 / context 判断、归属继承与候选选择 |
| 负载估计 | 原版 180 秒窗口、miss/count、prefill / decode 公式与二维时间模型 |
| 全局驱逐 | 原版虚拟 GPU radix cache、LRU victim 和重算代价；CLI 默认开启 |
| 重平衡 | 原版 warmup、migration 与 TTFT 热点 replication |
| 本地 MPQ | 持久十组、9→0 轮转、分词前到达时间 FIFO、`10 × (group+1)` 新请求配额 |

必要适配集中在当前 `Req` / radix / `PrefillAdder` 接口、chunk 准入、HTTP 生命周期以及硬件时间 / 容量参数。原模拟器的可选 hit-rate hint 未暴露；原版候选 ID、虚拟驱逐重复计数和特定 split 后断言等缺陷也保留并记录，因此不声称所有输入都能成功处理。

[实现文档](examples/preble/README.md) 给出完整伪代码、代价公式、逐文件源码映射、适配与原版缺陷的链接，以及两份 TP4 235B 副本的启动命令。

## 验证与复现

全局差分覆盖 **9012 次选择与 9012 次完成尝试**，状态与原源码一致：9006 次正常完成，6 次双方触发同一原断言。本轮又验证了真实准入 / chunk 源码 41 项、FIFO 与 pickle 8 项、取消生命周期另 41 项、原 HTTP TPOT 3 个场景，并用旧行为负对照检查 chunk 轮转与引用释放。它们是使用 CPU 基础设施桩的语义检查；旧 5000 次分组差分的覆盖盲区也已在[实现文档](examples/preble/README.md)中说明。验证脚本不作为单元测试提交。

真实推理使用 **Qwen3-235B-A22B-Instruct-2507、完整 BF16、8×H200、两个独立 TP4 副本**。本轮固定 runtime `c773f3e560`，用原版 LooGLE 生成器与旧 schema 数据产生 24 文档 / 393 请求的轨迹，在 1.4 RPS 档按 **Preble → RR → RR → Preble（ABBA）** 执行，每轮清空 cache、固定同一份到达文件。

四轮 **1572/1572 请求成功**。合并两轮的平均时延如下，单位为秒：

| 策略 | TTFT mean | E2E mean |
| --- | --- | --- |
| Preble + MPQ | 21.346 | 51.428 |
| RR + LPM | 34.717 | 65.251 |

本次轨迹上平均 TTFT / E2E 分别降低 **38.5% / 21.2%**。Preble 两轮波动仍明显；这是一档、两次重复的描述性比较。

逐轮数据、统计口径、源码验证与复现证据见[本轮实验报告](examples/preble/RUNTIME_RESULTS.md)。这是现代 SGLang 上的大模型验证，模型、硬件和随机流不同于论文，不等同论文数值复现。旧版 0.7 / 1.4 RPS 八轮数据保留在[历史报告](examples/preble/RESULTS.md)，其实现和执行协议不同，不作为当前五项修正的 GPU 验证。

公开入口：

- [准备原版生成器轨迹](benchmark/preble/prepare_reference_loogle.py)：固定源码 / 数据版本与 hash，输出 token IDs 和固定 arrival 文件。
- [校准部署时间模型](benchmark/preble/calibrate.py)：先 warmup，再冻结二维原模型的整体 scale 与初始 TPOT。
- [重放请求](benchmark/preble/bench_loogle.py)：同一份 workload / arrivals，用 Preble 或 round-robin 路由，保存逐请求与汇总指标。

安装沿用 [SGLang 安装说明](https://docs.sglang.io/get_started/install.html)，运行步骤见 [实现与运行方法](examples/preble/README.md)。本项目保留 SGLang 的 [Apache-2.0 许可证](LICENSE)。
