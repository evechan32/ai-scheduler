# MoE 推理优化论文综述（不涉及重训练）

> 检索日期：2026-08-16 | 来源：arXiv API（Semantic Scholar 限流，arXiv 为主）
> 主题：**不涉及重训练的 MoE/LLM 推理并行与优化**

---

## 1. MoE 专家 Offloading（异构 CPU/GPU 调度）

| 论文 | 年份 | 核心思路 |
|------|------|---------|
| **DALI**: Workload-Aware Offloading for Efficient MoE Inference on Local PCs | 2025 | 负载感知 offloading，本地 PC MoE 推理 |
| **Accelerating MoE Inference by Hiding Offloading Latency with Speculative Decoding** | 2025 | 投机解码隐藏 offloading 延迟 |
| **AcceptMoE**: Commitment-Weighted Self-Sizing Verifier Expert Sets | 2025 | 承诺加权自定尺寸验证器集合，MoE 投机解码 |
| **VisMMOE**: Exploiting Visual-Expert Affinity for VL MoE Offloading | 2025 | 视觉专家亲和度，VL-MoE offloading |
| **HOBBIT**: Mixed Precision Expert Offloading for Fast MoE Inference | 2024 | 混合精度专家 offloading（v1 调研已收录） |
| **CoMoE**: Collaborative Optimization of Expert Aggregation and Offloading | 2025 | 专家聚合与 offloading 协同优化，边缘 MoE |
| **Taming Latency-Memory Trade-Off via Fine-Grained Expert Offloading** | 2025 | 细粒度专家 offloading，时延-内存权衡 |
| **MoE-SpeQ**: Speculative Quantized Decoding with Proactive Expert Prefetching | 2025 | 投机量化解码 + 主动专家预取 |
| **DAOP**: Data-Aware Offloading and Predictive Pre-Calculation | 2025 | 数据感知 offloading + 预测预计算 |
| **TriMoE**: Augmenting GPU with AMX-Enabled CPU and DIMM-NDP | 2025 | **GPU + AMX CPU + DIMM-NPD 三路异构**，高吞吐 MoE |

## 2. CPU-GPU 异构并行推理

| 论文 | 年份 | 核心思路 |
|------|------|---------|
| **Dovetail**: CPU/GPU Heterogeneous Speculative Decoding | 2025 | CPU/GPU 异构投机解码 |
| **APEX**: Asynchronous Parallel CPU-GPU Execution for Online LLM Inference | 2025 | **异步并行 CPU-GPU 执行**，受限 GPU 在线推理 |
| **MIST**: Co-Design for Heterogeneous, Multi-Stage LLM Inference | 2025 | 异构多阶段 LLM 推理协同设计 |
| **Characterizing and Optimizing LLM Inference on CPU-GPU Coupled Architectures** | 2025 | CPU-GPU 耦合架构推理表征与优化 |
| **Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference** | 2025 | 混合 CPU-GPU 自动张量调度，消费级设备 |
| **SiPipe**: Bridging CPU-GPU Utilization Gap for Pipeline-Parallel Inference | 2025 | 管道并行中 CPU-GPU 利用率差距 |
| **CaraServe**: CPU-Assisted and Rank-Aware LoRA Serving | 2025 | CPU 辅助 LoRA 推理 |
| **HeteroMosaic**: Heterogeneous Execution for Energy-Efficient Edge LLM | 2025 | 异构执行，能效边缘 LLM |

## 3. Training-Free 推理优化

| 论文 | 年份 | 核心思路 |
|------|------|---------|
| **SparseInfer**: Training-free Prediction of Activation Sparsity | 2025 | 激活稀疏度预测（训练无关） |
| **DeltaLLM**: Training-Free Temporal Sparsity for Edge LLM | 2025 | 时间稀疏性，边缘 LLM |
| **WiSparse**: Weight-Aware Mixed Activation Sparsity | 2025 | 权重感知混合激活稀疏 |
| **Prox**: Training-Free FFN Activation Sparsity via Approximate Salience | 2025 | FFN 激活稀疏近似通道显著性 |
| **Eagle**: Efficient Training-Free Router for Multi-LLM | 2025 | 多 LLM 训练无关路由 |

## 4. MoE 专家并行与调度

| 论文 | 年份 | 核心思路 |
|------|------|---------|
| **Fine-Grained Scheduling of Disaggregated Expert Parallelism** | 2025 | 细粒度调度，解聚专家并行 |
| **Shortcut-connected Expert Parallelism** | 2025 | 捷径连接专家并行加速 MoE |
| **Least-Loaded Expert Parallelism: Load Balancing** | 2025 | 最少负载专家并行，负载均衡 |
| **UBEP**: Re-architecting Expert Parallelism Communication Library | 2025 | 专家并行通信库重构（生产超节点） |
| **MoEntwine**: Wafer-scale Chips for Expert Parallel Inference | 2025 | 晶圆级芯片专家并行推理 |

## 5. MoE 后训练量化（PTQ，不改权重）

| 论文 | 年份 | 核心思路 |
|------|------|---------|
| **QuantMoE-Bench**: Examining PTQ for MoE | 2025 | MoE 后训练量化基准 |
| **EAQuant**: Expert-Aware Optimization for MoE PTQ | 2025 | 专家感知 MoE 量化优化 |
| **VEQ**: Modality-Adaptive Quantization for MoE VLMs | 2025 | 模态自适应量化，MoE 视觉语言模型 |
| **TileQ**: Low-Rank Quantization of MoE with 2D Tiling | 2025 | 低秩量化 + 2D 分块 |

## 6. 投机解码 / 注意力 / KV 优化（推理时，无需重训练）

| 论文 | 年份 | 核心思路 |
|------|------|---------|
| **Scout Before You Attend**: Sketch-and-Walk Sparse Attention | 2025 | 草图行走稀疏注意力，长上下文 |
| **Kascade**: Practical Sparse Attention for Long-Context LLM | 2025 | 长上下文稀疏注意力 |
| **Revisiting Judge Decoding via Training-Free Distributional Divergence** | 2025 | 训练无关分布散度解码 |

---

## 优化方向全景总结

**不涉及重训练的 MoE 推理优化可组合成五层管线**（对应我们的 moesim 框架）：

1. **调度层**（已完成 ✅）：异构专家调度——CPU/GPU 专家并发执行（moesim 真并行），
   对应 DALI/HOBBIT/CoMoE/TriMoE 的 offloading 调度思路
2. **量化层**（已完成 ✅）：专家级 PTQ——INT8/INT4 量化（moesim 已实现），
   对应 QuantMoE-Bench/EAQuant/TileQ 的专家感知量化
3. **稀疏层**（待扩展）：激活稀疏——FFN 稀疏执行（Prox/SparseInfer），
   可跳过不活跃专家计算
4. **投机层**（待扩展）：投机解码——小模型起草 + 大模型验证（Dovetail 已做 CPU/GPU 异构版），
   可隐藏 offloading 延迟（MoE-SpeQ）
5. **内存层**（部分完成）：KV 管理——KV 分层调度（moesim KVWeightedPolicy），
   对应稀疏注意力/长上下文优化

**与本项目（moesim）的映射**：
- 已实现：调度层（真并行 CPU/GPU）+ 量化层（INT4/INT8）+ KV 层（KVWeightedPolicy）
- 可借鉴：APEX 的异步并行执行模式、Dovetail 的 CPU/GPU 投机分工、Prox 的激活稀疏、
  TriMoE 的三路异构（GPU+CPU+NDP）
