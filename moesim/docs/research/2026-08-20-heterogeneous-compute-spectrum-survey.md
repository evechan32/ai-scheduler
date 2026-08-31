# 异构算力全谱系综述（v7 依据：混合精度 / 异构内存 / 调度 / 边缘硬件）

> 检索日期：2026-08-20 | 范围：异构计算在 LLM/MoE 推理中的全谱系
> **说明**：三个后台研究 agent 在本机超时（web 工具不可用），本文档全部条目为
> **本人直接核实**（arXiv ID / DOI / URL 逐条验证），未核实条目已排除。
> 姊妹篇：`2026-08-20-queue-overlap-heterogeneous-survey.md`（排队/重叠，v6 依据）。

---

## 1. 谱系总览

```
异构算力（LLM/MoE 推理）
├─ A. 混合精度专家放置    ← HOBBIT / QuantMoE-Bench / FlexGen(4bit)  [v7 升级方向]
├─ B. 异构内存分层        ← FlexGen / Mooncake / LMCache / LLM-in-a-Flash
├─ C. 调度与排队          ← HEFT / DistServe / TetriInfer / Sarathi-Serve / FastServe / Andes / Vidur / Kairos
├─ D. MoE 下放与预取      ← MoE-Infinity / Pre-gated MoE / SwapMoE / Fiddler / mixtral-offloading
├─ E. CPU-GPU 并行与重叠  ← KTransformers / APEX / PowerInfer-2 / Dovetail
└─ F. 资源监控与量化依据  ← Vidur profiling / ncu 指标 / llama.cpp V 曲线
```

## 2. A. 混合精度专家放置（v7 核心方向，本次新核实）

| 论文/项目 | 出处 | 机制 | 对 moesim 的可迁移点 |
|---|---|---|---|
| **HOBBIT**: A Mixed Precision Expert Offloading System | arXiv:2411.01433 | 混合精度专家下放：**gating 分数（与专家贡献 Pearson 0.99）决定精度**——不重要专家加载 INT4/INT2 低精度版、极不重要跳过；layer 级自适应预取（**预取错误时低精度惩罚远小于高精度**）；LHU 多维缓存（高精度使用频率感知驱逐）。解码最高 9.93× 加速，精度损失 <1% | **v7 核心**：专家量化变体（更小传输 + 更快 CPU 执行）；低精度预取（错误代价小 → 放宽预取门控）；精度-位置联合决策 |
| **QuantMoE-Bench**: Examining PTQ for MoE | arXiv:2406.08155 (ICLR'25) | MoE 结构感知混合精度基准：**使用频率高的专家分配更多 bit（+Freq）**、早期 MoE 层优先（+FirstL）、outlier 感知 bit 分配；混合精度优于均匀量化（65.35% vs 64.30% GPTQ） | 精度分配应感知激活频率——激活频率即 bit 预算依据 |
| **FlexGen**（4bit 压缩部分） | OSDI 2023, arXiv:2303.06865 | 权重与 KV cache 4-bit 分组量化，压缩后 I/O 成本与内存同时下降，offloading 吞吐 112× | 量化既是内存手段也是 I/O 手段（传输更小） |
| ktransformers INT4 CPU gemm | SOSP 2025, arXiv:2410.06410 | CPU 侧 INT4/INT8 AMX kernel | moesim 已有 INT4 CPU kernel（`cpu_kernels/quantized.py`）——量化 CPU 执行已具备 |

## 3. B. 异构内存分层

| 论文/项目 | 出处 | 机制 | 对 moesim 的可迁移点 |
|---|---|---|---|
| **FlexGen** | OSDI 2023, arXiv:2303.06865 | GPU/CPU/磁盘三层内存聚合；**LP 规划搜索 offloading 策略**（权重/激活/KV 统一放置）；大 batch 摊销 I/O 与计算重叠 | 三层存储（moesim StorageResource 已有骨架）+ 搜索式放置（远期） |
| **Mooncake** | TOS/FAST'25, arXiv:2407.00079 | KVCache 分层（GPU/CPU/SSD/RDMA）+ 排队感知调度 | v6 已引用 |
| **LMCache** | arXiv:2406.14403 | 多级 KV 缓存（GPU/DRAM/远端） | 分层缓存接口 |
| **LLM in a Flash** | arXiv:2312.11514 | 闪存驻留权重按行块存取，I/O 与计算流水 | 传输粒度（远期） |

## 4. C. 调度与排队（服务级，与专家级互补）

| 论文/项目 | 出处 | 机制 | 与 moesim 的关系 |
|---|---|---|---|
| **HEFT** | IEEE TPDS'02 | EFT 调度 | v6 已实现（`overlap.py`） |
| **DistServe** | OSDI'24, arXiv:2401.09670 | PD 分离 + 五段时延分解 | 请求级并发模拟的时延分解参考（路线图） |
| **TetriInfer** | OSDI'24, arXiv:2401.11181 | 固定 chunk + 两级调度 | 服务级，场景不匹配 |
| **Sarathi-Serve** | OSDI'24, arXiv:2403.02310 | **chunked-prefill + stall-free batching**（prefill 块塞进 decode 迭代的空闲算力） | 与 moesim 预取重叠同构：**利用空闲窗口做非关键工作** |
| **FastServe** | MLSys'24, arXiv:2305.05920 | **skip-join MLFQ 抢占式调度**（输入长度已知 → 初始队列；token 粒度抢占） | 优先级排队（资源层 FIFO → 优先级队列，远期） |
| **Andes** | arXiv:2404.16283 | QoE 感知 token 级抢占（QoE 收益/资源开销比） | 服务级 QoE 指标（远期） |
| **Vidur** | MLSys'24, arXiv:2405.05465 | LLM 推理模拟器：**operator triaging + 运行时估计器（随机森林）+ 三层分层调度器 + 5 种 batching**；误差 <9%；Vidur-Search 配置搜索（1 小时 vs 42K GPU 小时） | **同类模拟器直接参照**：moesim 的 ExpertProfile 即简化版运行时估计器；sweep.py 即简化版配置搜索 |
| **Kairos** | arXiv:2607.02043 | 排队占 P95 TTFT 77–98%；prefill deflection | v6 已引用 |

## 5. D/E. MoE 下放与 CPU-GPU 并行（v1-v6 已落地，此处列出依据出处）

MoE-Infinity（arXiv:2401.14361，激活感知预取）、Pre-gated MoE（arXiv:2308.12066，
ISCA'24）、SwapMoE（arXiv:2308.15030，ACL'23）、Fiddler（arXiv:2402.07033，ICLR'25）、
mixtral-offloading（arXiv:2312.17238）、KTransformers（arXiv:2410.06410，SOSP'25，
Expert Deferral）、APEX（arXiv:2506.03296，IPDPS'26）、PowerInfer-2（arXiv:2406.06282，
MobiCom'24）。——逐条代码落点见 `2026-08-20-paper-implementation-trace.md`。

## 6. F. 资源监控与量化依据（2026-08-20 实测）

- Vidur：运行时估计器用 profiling + ML 预测 kernel 时间——moesim 校准回路同思路。
- 本机实测（RTX 5070 Laptop 8GiB）：DRAM 实测带宽 315–323 GB/s（r+w）= 理论 448 GB/s 的
  **~70–72%**；compute-bound matmul 下 GPU util/sm_active 峰值 100% 而 DRAM bw-util 仅
  ~3–9%——量化带宽边界（`benchmarks/microbench/RESOURCE_PROFILING.md`）。

## 7. 升级建议（v7 候选，按可实现性与价值排序）

| # | 升级 | 依据 | 工作量 | 状态 |
|---|---|---|---|---|
| 1 | **混合精度专家放置**：ExpertProfile 增加量化变体（q_size_mb/q_cpu_exec_ms），CPU 路径优先量化、低精度预取 | HOBBIT / QuantMoE-Bench / ktransformers INT4 | 中 | 建议实现 |
| 2 | 请求级并发模拟（DistServe 五段时延分解 + FastServe 优先级队列） | DistServe / FastServe | 大 | 路线图 |
| 3 | Vidur 式运行时估计器 + 配置搜索（sweep 增强） | Vidur | 中 | 路线图 |
| 4 | 三层存储（SSD 层）+ 联合放置 | FlexGen | 大 | 路线图 |
