# SGLang Prefix Load Balancing

把 [Preble](https://github.com/WukLab/preble/tree/1a35eae20c416f7062ca81775aab72890323fd97) 的全局前缀路由与本地 MPQ 调度接入当前 SGLang。参考 commit `1a35eae20c`，当前 runtime `091d70e1e7`，SGLang base `ef11e39048`。

## 请求执行流程

```text
/generate → frontend 分词、更新全局前缀树、选择副本
          → 发送相同 token IDs → worker MPQ 排队 → SGLang prefill / decode
          → 转发流式输出 → 完成反馈、释放引用、更新 TTFT / TPOT
```

各 worker 的实际 KV 留在本地 HBM。Frontend 的虚拟缓存用于估算驱逐重算代价；migration / replication 调整未来路由归属，由后续请求建立缓存，不传输 KV，也没有接 Mooncake。

## 调度算法

压缩 radix 节点的 `segment` 是节点自身 token 数，`context_before` 是此前路径长度。`segment > context_before` 为大节点；请求组是从叶向上最近的大节点。

```text
route(prompt, output_budget):
    leaf, splits = 全局 radix.insert(prompt)
    按原规则继承 split 的归属、引用和窗口统计
    group = leaf 向上最近的大节点
    candidates = 从 leaf 向上最近非空归属 if leaf.segment < leaf.context_before else 全部副本
    w = 唯一候选；多个候选时按原版 load + eviction_cost 选择
    更新归属、虚拟缓存、在途引用和 180 秒窗口
    若虚拟占用超容量：按原虚拟 LRU 驱逐
    若累计请求 ≥ 50×副本数 且 leaf 到 group 的深度差 < 3：
        仅计窗口请求数 ≥ 2 的组；最大负载 ≥ 1.5×最小负载时尝试重平衡
        多个可迁组：从小代价组开始迁，直到平衡或迁移会反转两端负载
        单个热点组：目标尚无该组、两半窗都有样本、后半窗平均 TTFT ≥ 前半窗 2 倍时添加归属
    forward(w, prompt token IDs)
```

候选代价如下。`T(new_tokens, full_context)` 保留原二维 prefill 模型，按部署整体缩放；`g` 是该副本所属前缀组，`v` 是原虚拟 LRU 选出的 victim：

```text
prefill(g) = miss_rate(g) × 窗口请求数(g) × T(g.segment, g.full_context) / 归属副本数(g)
decode(g,w) = 在途引用数(g,w) × 组输出预算(g) × median(TPOT[w])
load(w) = Σg [prefill(g) + decode(g,w)]
eviction_cost(w) = Σ大节点v [miss_rate(v) × 窗口请求数(v) × T(v.segment, v.full_context)]
```

原版对参与代价比较的每个候选都模拟释放新 leaf 段所需空间，不等实际容量满了才加驱逐代价；缺少窗口统计时按原规则回退。输出长度采样在原公开路径中被注释，因此组输出估计回退到最近预算。

```text
worker_prefill():
    先续已有 chunk，扣除 token / slot / chunk 预算
    若预算耗尽：返回，不推进 MPQ cursor
    首次进入等待队列：按本地命中率固定分入 10 组，等待期间不换组
        group = max(0, ceil(10 × hit_rate) - 1)，取值 0…9
    从 cursor 按 9→0 循环选择首个非空组；组内按分词前到达时间 FIFO
    刷新所选组匹配；新请求配额 = min(组大小, 10×(组号+1))
    顺序尝试准入，实际加入才消耗配额：
        普通容量 / 输入不满足 → SKIP，继续同组后项
        锁后容量不足或本轮预算耗尽 → 停止
    请求准入或取消离队时清除 MPQ 引用；retract 后重新分组

worker_retract_if_decode_oom():
    非 speculative MPQ：按(原到达时间, 已输出长度)升序排序，从后撤回

finish(request):
    释放原副本路径引用，记录组 TTFT
    output_len > 1 且 latency > TTFT > 0 时：TPOT[w].append((latency - TTFT) / output_len)
```

## 与 Preble 的一致性

已按原**公开服务默认路径**复核上述算法、候选迭代顺序、HTTP 反馈与非 speculative MPQ retract。保留原 argmin 候选位置缺陷、虚拟 LRU 重复父项和特定 split / expiry 断言，以及成功 HTTP 反馈预算为 0 的旧 ledger 行为；实际路由代价使用活跃引用。

当前页级 KV / slot、single chunk、物理释放与已生成输出保留属于 runtime 适配，不能承诺与旧版逐 batch 相同。原模拟器可选 hint 未接入；不可用 worker 过滤与显式非流式是现代服务扩展。详细对应与边界见[一致性验证](examples/preble/ALIGNMENT_REAUDIT.md)。

## 验证结果

| 验证 | 结果 |
| --- | --- |
| 全局源码差分 | 9012 次选择 / 完成尝试状态一致；9006 正常完成，6 次双方同一原断言。 |
| 本地调度 | 准入 / chunk 41 项、FIFO / IPC 8 项、取消 41 项；retract 306 项对原版、1200 项既有路径、1 项现代保护通过。 |
| 服务与候选 | 健康池 960、部分池 1200、HTTP 计时与预算差分通过，含修前负对照。 |
| 真实推理 | 2296 指定容器，BF16 Qwen3-235B-A22B、8×H200、2×TP4；Preble + MPQ 与 RR + LPM 各一轮，均 393/393 成功。 |

同一 LooGLE 输入与到达轨迹，时延单位为秒：

| 指标 | Preble + MPQ | RR + LPM |
| --- | ---: | ---: |
| 平均 / p95 TTFT | 28.313 / 80.285 | 42.322 / 147.955 |
| 平均 / p95 端到端时延 | 62.435 / 126.287 | 76.554 / 171.768 |
| 整轮用时 | 279.524 | 329.190 |
| 输入缓存比例 | 83.539% | 81.048% |

本次 Preble 平均 TTFT 降低 **33.1%**，平均端到端时延降低 **18.4%**。六个冷 prefill 预热样本的 TTFT 差异均小于 0.5%；正式输出为 17,576 / 17,498 tokens，四条请求长度不同且正常停止。各方案一次、固定先后顺序，比较包含全局路由与本地调度，不证明稳定收益或论文数值复现。[235B 对照报告](examples/preble/RUNTIME_2296.md)提供配置、指标与证据。

源码语义验证与 GPU 性能证据分别记录。旧 A1/A2 的后台速度在路由分叉前已不同，旧 ABBA 不能作为控制住副本速度的比较，见[延迟差异结论](examples/preble/VARIANCE_ANALYSIS.md)。

[启动、轨迹准备与执行命令](examples/preble/README.md) · [源码验证数据](examples/preble/results/reaudit-20261015/semantic-validation.json) · [历史实验](examples/preble/RUNTIME_RESULTS.md)

安装沿用 [SGLang 文档](https://docs.sglang.io/get_started/install.html)，许可证为 [Apache-2.0](LICENSE)。
