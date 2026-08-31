# 异构算力排队与重叠研究综述（v6 依据）

> 检索日期：2026-08-20 | 范围：**排队影响 / CPU 资源影响 / 通算并行 / 计算-传输重叠**
> 用途：作为 moesim v6（队列与重叠感知调度）的设计依据。arXiv ID 均已逐条核实。

---

## 1. 核心结论（给 v6 设计的一句话摘要）

1. **排队是时延的主要成分**：Kairos（arXiv:2607.02043）在 2P2D 生产集群上实测，P95 TTFT
   中 **prefill 执行仅占 2–23%，其余 77–98% 是排队等待 + KV 传输**——调度决策必须看见队列。
2. **CPU 算力不是免费资源**：llama.cpp 社区实测表明 CPU offload 只在显存溢出时获益，
   过度 offload 使吞吐单调下降；KTransformers 的 Expert Deferral 说明 CPU 参与需要
   **主动填充空闲窗口**，否则 CPU 利用率停在 75% 以下。
3. **重叠需要显式的预取流水线**：MoE-Infinity 用「激活感知预取优先级队列 + 每链路独立
   I/O 线程」把 PCIe 传输藏在 GPU 计算背后；PowerInfer-2 用 neuron-cluster 粒度
   I/O-计算流水线。**预取必须受带宽预算约束**——过度预取反而挤占 PCIe。
4. **通算并行 = 排队感知的任务分派**：HEFT（TPDS'02）给出「以最早完成时间为目标、
   把通信成本计入决策」的经典范式；APEX（IPDPS'26）证明 profiling-informed 的动态
   分派优于静态规则。

---

## 2. 顶会论文（已核实）

### 2.1 排队与调度理论

| 论文 | 出处 | 机制 | 对 moesim 的可迁移点 |
|------|------|------|----------------------|
| **Kairos**: Towards Load-Aware Prefill Deflection for Disaggregated LLM Serving | arXiv:2607.02043 | prefill 节点饱和时把请求 deflect 到 decode 节点做 chunked prefill；每请求估计 prefill 节点 TTFT（含队列等待）与 decode 节点 TBT 可行性 | **排队占比实测证据**（77–98%）；「预估等待 = 聚合排队作业时间」的估计方法（Mooncake 同款） |
| **DistServe**: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving | OSDI 2024, arXiv:2401.09670 | prefill/decode 分池；把请求生命周期拆成 **prefill 排队 → prefill 执行 → 传输 → decode 排队 → decode 执行** 五段做时延分解 | **时延分解的排队阶段划分**——v6 指标应区分排队等待与执行时间 |
| **TetriInfer**: Inference without Interference | OSDI 2024, arXiv:2401.11181 | 固定大小 chunk 使算力饱和；两级调度 + 长度预测避免 decode 热点 | 资源使用率预测驱动调度；热点避免 |
| **HEFT**: Performance-Effective and Low-Complexity Task Scheduling for Heterogeneous Computing | IEEE TPDS 13(3):260-274, 2002, DOI:10.1109/71.993206 | 异构处理器 list 调度：按向上秩选任务，**分配给最早完成时间（EFT）最小的处理器**，通信成本计入 | **v6 放置决策的核心公式**：EFT = 队列等待 + 传输/执行时间，取最小者 |
| **Mooncake**: A KVCache-centric Disaggregated Architecture for LLM Serving | TOS/FAST'25, arXiv:2407.00079 | Conductor 全局调度：**TTFT 预估 = 前缀命中减省的计算 + 排队等待（聚合队列内各请求 prefill 时间）**；超载预测性拒绝 | 排队感知路由 + 等待时间聚合估计；CPU/DRAM/SSD 分层缓存 |

### 2.2 CPU-GPU 并行与重叠

| 论文 | 出处 | 机制 | 对 moesim 的可迁移点 |
|------|------|------|----------------------|
| **KTransformers**: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models | SOSP 2025, arXiv:2410.06410 | AMX/AVX512 专用 kernel；**异步 CPU-GPU 任务调度**（无锁队列 + CUDA graph 封装 submit/sync）；**Expert Deferral**：把不紧迫的专家任务延后到 CPU，把 CPU 利用率从 <75% 提到 ~100%（最高 1.45× 吞吐） | **重叠机制的直接范本**：CPU 执行与 GPU 执行并行，任务经队列异步分派；延迟同步点 |
| **APEX**: Asynchronous Parallel CPU-GPU Execution for Online LLM Inference on Constrained GPUs | IPDPS 2026, arXiv:2506.03296 | profiling-informed 调度：预测 CPU/GPU 子任务执行时间，动态分派最大化重叠；**Asynchronous Overlap**：统一 batch 计算 linear 层后分支，CPU 注意力结果延迟到下一轮才同步 | 模型驱动（基于执行时间预测）的分派决策；延迟同步最大化 CPU 计算窗口 |
| **PowerInfer-2**: Fast LLM Inference on a Smartphone | MobiCom 2024, arXiv:2406.06282 | neuron-cluster 细粒度分解；**I/O-计算流水线**：集群粒度 pipelining，权重加载与矩阵计算重叠；NPU/CPU 按稀疏度动态分配比例 | 传输与计算重叠的粒度抽象；I/O 预算与计算负载的联动 |
| **MoE-Infinity**: Activation-Aware Expert Offloading for Efficient MoE Serving | arXiv:2401.14361 (v2: offloading 版) | 请求级激活追踪（EAM/EAMC）→ **激活感知预取**：优先级队列 + 每 PCIe 链路独立 I/O 线程，顺序取专家避免带宽争用；多层预取按「层邻近度」定优先级；SSD→DRAM 与 DRAM→GPU 双级流水 | **v6 prefetch 机制范本**：预取优先级、PCIe 串行化（避免并发争用）、预取必须与执行重叠；「预取失败比不预取更糟」的谨慎原则 |
| **Pre-gated MoE** | ISCA 2024, arXiv:2308.12066 | 推测路由（pre-gating）提前预测专家激活，为预取争取时间窗 | 预取时间窗 = 路由预测与执行之间的间隙 |
| **Fiddler** | ICLR 2025, arXiv:2402.07033 | CPU-GPU 编排 + AVX512_BF16 CPU kernel | CPU 算力入决策的早期代表（v1 已引用） |

### 2.3 专家驻留/缓存（与 v5 衔接）

| 论文 | 出处 | 机制 | 对 moesim 的可迁移点 |
|------|------|------|----------------------|
| **SwapMoE** | ACL 2023, arXiv:2308.15030 | 可调内存预算的专家交换 | 驻留预算约束（v5 residency 基础） |
| **HOBBIT** | arXiv:2411.01433 | 混合精度专家下放（GPU 高精度 / CPU 低精度） | 精度-位置联合决策（远期） |
| **SmartMoE / EdgeMoE / FasterMoE** | — | 专家并行、负载均衡、通信重叠 | 通算并行的通信重叠（远期） |

---

## 3. 高星开源项目（已核实）

| 项目 | 仓库 / Stars | 异构执行 | 重叠机制 | 排队/调度 | CPU 资源管理 | 对 moesim 的启示 |
|------|------|------|------|------|------|------|
| **ktransformers** | kvcache-ai/ktransformers (~11k) | CPU AMX/AVX512 kernel + GPU FlashInfer；热专家 GPU / 冷专家 CPU | 异步任务队列 + CUDA graph 封装同步；Expert Deferral 填充 CPU 空闲 | 无锁任务队列 | CPUinfer 线程池、NUMA 感知张量并行 | 任务队列抽象 + 延迟同步点；CPU 并行度由线程池控制 |
| **llama.cpp** | ggml-org/llama.cpp (~80k) | `--n-cpu-moe N` 把前 N 层 MoE 专家权重留在 CPU（PR #15077，tensor override） | ggml 图内多线程；CUDA graph 支持 n-cpu-moe（PR #18934） | batch 调度；`--threads`/`--threads-batch` 分离生成/批量线程 | 线程池 + CPU affinity mask | **CPU offload 的 V 曲线实证**：只卸到「刚好放得下」，再多就吞吐单调下降——CPU 卸载不是免费的 |
| **vLLM** | vllm-project/vllm (~60k) | CPU backend（torch CPU）+ 可选 CPU KV offload | **async scheduling**（PR #19970，NanoFlow 思路）：调度器提前一步运行，与 GPU forward 重叠；V1 EngineCore 独立进程重叠 CPU 侧处理 | V1 scheduler：token 预算 + lookahead slots + 抢占 | 多进程 executor；`max_concurrent_batches` | 调度与执行重叠的工程实践；输出占位符（先调度后出 token） |
| **SGLang** | sgl-project/sglang (~25k) | HiCache 多级 KV（device/host/remote）+ Mooncake TE | 调度与 forward 重叠；chunked prefill | RadixAttention 前缀缓存 | 与 Mooncake TE 集成 | 分层缓存的传输重叠 |
| **MoE-Infinity** | efficientmoe/moe-infinity | 专家权重重置 GPU/主机/SSD；激活感知缓存 | 激活感知预取 + 每 GPU I/O 线程；FP4/MXFP4 路径 | 连续 batching + 抢占/换出 | pinned memory + DMA | 预取优先级队列、PCIe 串行化、缓存-预取协同 |
| **PowerInfer / PowerInfer-2** | SJTU-IPADS/PowerInfer (~9k) | hot/cold 神经元划分；NPU/CPU 混合 | neuron-cluster I/O-计算流水线 | 自适应比例调整 | 大/小核利用 | 粒度与流水线；batch 变化时动态调分配比例 |
| **Mooncake** | kvcache-ai/Mooncake | KVCache 分层（GPU/CPU/SSD/RDMA） | Transfer Engine：多 NIC 带宽聚合 + 拓扑感知选路 + 故障切换；KV 流式传输与 prefill 重叠 | Conductor 全局调度（缓存感知 + 队列感知 + SLO 拒绝） | NUMA 感知路径选择 | 传输引擎抽象；排队感知调度器 |
| **TensorRT-LLM** | NVIDIA/TensorRT-LLM | 多后端；MoE 权重流式加载 | `overlap_expert_weight_load`：MoE 权重加载与计算重叠 | 批处理调度 | — | 权重加载重叠的算子级实现 |

> Stars 为检索时近似值（2026-08）；机制描述基于各项目文档/PR/论文原文核实。

---

## 4. 对 v6 的设计映射

| v6 需求 | 依据 | 设计落点 |
|---------|------|----------|
| **排队影响** | Kairos/DistServe/Mooncake/HEFT | 资源层暴露 `queue_depth/wait_time/utilization`；策略用 **EFT（排队等待 + 执行）** 决策；指标含队列深度与等待聚合 |
| **CPU 资源影响** | llama.cpp V 曲线 / KTransformers / APEX | CPU 队列深度 + 利用率反馈进状态；CPU 饱和时 inflate CPU 成本（`cpu_eff = cpu_exec × (1+contention)`），避免把专家压给已饱和的 CPU |
| **通算并行** | KTransformers / APEX / PowerInfer-2 | CPU/GPU 双计算资源并行执行（保留 v4 真并行）；放置决策由 EFT 在两条路径间选择 |
| **重叠性** | MoE-Infinity / vLLM async / PowerInfer-2 | 新增 `prefetch` action：预取传输与当前步计算重叠（不进关键路径）；**带宽预算门控**（PCIe 拥塞时不预取）防止过度预取 |
