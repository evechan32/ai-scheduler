# Changelog

All notable changes to moesim are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/) conventions.

## [v6] - 2026-08-20

### Added — 排队与重叠感知调度（queueing & overlap-aware scheduling）

- `sim/resources.py`: BandwidthResource / ComputeResource 暴露队列可见性 —
  `queue_depth(now)`（在途+排队作业数）、`utilization(until)`（忙时占比）、
  `wait_time_ms(now, units)`（peek 等待估计，不污染资源状态）。
- `sim/metrics.py`: 新增队列深度均值/最大值、PCIe/GPU/CPU 利用率、
  transfer_wait_ms、hidden_transfer_ms、prefetch_count、overlap_ratio。
- `scheduler/base.py`: 新增 `prefetch` Action kind（apply_actions 与 load 同语义）。
- `scheduler/state.py`: 新增资源反馈字段（pcie_queue_len / 利用率 / 等待时间估计 /
  pending_loads 在途传输表）。
- `scheduler/policies/overlap.py`: `OverlapAwarePolicy` — HEFT 式 EFT 放置
  （排队等待 + 执行时间取最小者，CPU 资源影响经 cpu_wait/contention 计入）
  + 带宽门控预取（PCIe 队列深度/利用率超限时不预取，MoE-Infinity 谨慎原则）。
- `sim/moe_adapter.py`: prefetch 传输与当前步计算重叠（不进关键路径）；
  pending_loads 跨步追踪在途传输，load 复用在途传输不重复占带宽；
  执行起点 = max(clock, 在途传输完成时间)；每步反馈资源队列/利用率/等待快照。

### Fixed

- `scheduler/policies/residency.py`: load 前容量检查缺失（v6 资源积压反馈语义暴露），
  超容量时回退 execute_cpu。

### Changed

- `sim/moe_adapter.py`: v5 的 `gpu_queue_len` 语义从「本步 GPU 执行专家数」改为
  「资源真实积压」（queue_depth），调度器看到真实排队影响。

### Benchmarks

- `benchmarks/e2e/compare_queue_overlap.py`: 热专家 trace 对比
  lru / cost_model / residency / overlap(no-pf) / overlap(pf=2)。
  实测：overlap(pf=2) TPOT 2.368ms，比全 CPU 放置（residency 4.800ms）快 50.7%，
  比 lru（2.475ms）快 4.3%；PCIe 拥塞时预取门控生效。

### Docs

- 新增 `docs/research/2026-08-20-queue-overlap-heterogeneous-survey.md`
  （排队/CPU/并行/重叠方向论文与高星项目综述，arXiv ID 已核实）。
- 新增 `docs/research/2026-08-20-paper-implementation-trace.md`
  （论文-实现追踪：机制 → 代码落点 → 参考程度 → 简化点）。
- 新增 `docs/superpowers/specs/2026-08-20-moesim-v6-design.md`。
- 新增 `docs/superpowers/plans/2026-08-20-moesim-v6.md`（TDD 计划）。

### Tests

- 107 passed, 3 skipped（1 个 INT4 kernel 测试为本机环境性失败，基线存在）。

## [v5] - 2026-08-17

- ResidencyAwarePolicy：GPU 排队感知、CPU 资源感知、迁移成本 + 驻留收益、驻留稳定性。
- 实测：热专家场景比 cost_model 快 46.9%（0.221ms vs 0.416ms）。

## [v4] - 2026-08-16

- 真并行执行（线程池并发 CPU/GPU）、决策缓存、REINFORCE 策略梯度、INT4 gemm、
  一键安装脚本、论文综述 50+ 篇。

## [v3] - 2026-08-15

- MoEForwardHook 真实推理闭环（误差 0.062%）、RL 策略（Q-learning）、多 GPU 拓扑。

## [v2] - 2026-08-15

- 真实权重 offload、KVWeightedPolicy、INT8/INT4 量化 kernel、vllm/llama.cpp 后端。

## [v1] - 2026-08-09

- 领域无关 DES 核心、资源模型、三种调度策略、CPU FP16 kernel、校准回路。

## [v7] - 2026-08-20

### Added — 混合精度专家放置（mixed-precision placement）

- `scheduler/cost_model.py`: `ExpertProfile` 增加量化变体字段 `q_size_mb` / `q_cpu_exec_ms`
  （默认 None 向后兼容）+ `quantized_size_mb()` / `quantized_cpu_exec_ms()`。
- `scheduler/policies/overlap.py`: CPU 路径 EFT 用 `quantized_cpu_exec_ms` —— 量化 CPU
  执行更快 → 更多专家倾向 CPU（HOBBIT 动态精度选择）。
- `sim/moe_adapter.py`: `execute_cpu` 用量化 CPU 执行时间；`prefetch` 传输用
  `quantized_size_mb`（低精度预取更小，HOBBIT：预取错误惩罚小）。
- 依据：HOBBIT（arXiv:2411.01433）、QuantMoE-Bench（arXiv:2406.08155）、
  ktransformers INT4 CPU gemm（arXiv:2410.06410）。

### Added — 真机资源利用率监控

- `benchmarks/microbench/resource_monitor.py`: 零依赖采样器（nvidia-smi 轮询
  gpu_util / 显存 / 时钟 + /proc CPU 与内存利用率 + dmon sm/mem 代理 + 可选 NCU）。
- `benchmarks/microbench/profile_resource_usage.py`: compute-bound + bandwidth-bound
  负载下采集并落盘 JSON。
- 实测（RTX 5070 Laptop 8GiB）：DRAM 带宽 **315–323 GB/s（r+w）= 理论 448 GB/s 的
  ~70–72%**；matmul 下 GPU util/sm_active 峰值 100% 而 DRAM bw-util 仅 ~3–9%。
- 记录：`benchmarks/microbench/RESOURCE_PROFILING.md`。

### Docs

- `docs/research/2026-08-20-heterogeneous-compute-spectrum-survey.md`: 异构算力全谱系
  综述（混合精度/异构内存/调度/边缘硬件，全部条目本人核实）。

### Tests

- 116 passed, 2 skipped（新增 7 个混合精度测试）。

## [v8] - 2026-08-20

### Added — KV cache 分层模拟 + KV-专家联合调度

- `scheduler/state.py`: KV 字段（kv_per_token_mb / kv_pressure / kv_evict_count / kv_fetch_count）。
- `sim/moe_adapter.py`: **KV 增长模拟**（每步 token×kv_per_token 进 GPU 池）+ **超限自动下放
  主机**（走 PCIe 占带宽、排队影响后续 load）+ 压力快照。
- `sim/metrics.py`: kv_gpu/host 利用率、kv_offload_bytes。
- `scheduler/policies/kv_joint.py`: **KVJointPolicy**（继承 v6 OverlapAwarePolicy）——
  高压时专家 CPU 化（省显存给 KV）+ 暂停预取 + 驱逐冷 KV；低压时正常 EFT/预取。
- `benchmarks/e2e/compare_kv_tiering.py`: 长上下文 KV 增长对比
  cost_model / kv_aware(v2) / kv_joint(v8)。

**实测**（长上下文 300 步，50MB/token，GPU KV 池 6GiB）：
| 策略 | TPOT | kv_offload | 说明 |
|---|---|---|---|
| cost_model | 2.303ms | **8856MB** | 无视 KV 压力，显存挤压时大量下放 + PCIe 排队 |
| kv_aware(v2) | 5.628ms | 0MB | 每步显式驱逐 100MB（按专家 size），PCIe 成本高 |
| **kv_joint(v8)** | 5.359ms | **6MB** | 高压 CPU 化 + 驱逐，下放量降 99.9% |

取舍：kv_joint 用 TPOT 代价换 KV 池保护（防止显存挤压）。

- 依据：Mooncake（arXiv:2407.00079）、FlexGen（arXiv:2303.06865）、LMCache
  （arXiv:2406.14403）。设计：`docs/superpowers/specs/2026-08-20-moesim-v8-design.md`。

### Environment fix

- vllm-build 环境 NCCL 修复：torch 2.13 依赖 nvidia-nccl-cu13（2.29.7），
  libnccl.so.2 曾被 cu12 覆盖导致 `undefined symbol: ncclCommResume`——
  强制重装 cu13 NCCL 恢复（`pip install --force-reinstall nvidia-nccl-cu13`）。

### Tests

- 127 passed, 2 skipped（新增 12 个 KV tiering/策略测试；INT4 环境性失败在 torch 2.13 再现）。

## [v9] - 2026-08-20

### Added — 请求级并发模拟（DistServe 式时延分解）

用户路线图第一项（"能够模拟"的最大缺口：单请求步模型 → 多请求并发）。

- `sim/request_sim.py`: `Request`（arrival / prompt / output）+ `RequestSimulation`：
  - **prefill**：GPU 计算块（prompt × prefill_per_token_ms），FIFO 排队（GPU 忙则等）
  - **decode**：多请求轮询（round-robin）共享 GPU/CPU/PCIe，复用 v8 资源/状态机制
  - **时延分解**：TTFT = prefill 排队 + prefill 执行；JCT = TTFT + decode；per-request stats
  - KV 增长按每请求 token（prefill + decode）记账
- `benchmarks/e2e/compare_request_concurrency.py`: 并发度对比基准。

**实测**（8 请求，prompt 64 / output 32，2ms 到达间隔）：
| gpu_slots | TTFT_avg | TPOT_avg | JCT_avg | prefill排队占比 | 吞吐 |
|---|---|---|---|---|---|
| 1 | 137ms | 14.56ms | 603ms | 76.6% | 361.6 tok/s |
| 4 | 137ms | 10.51ms | 473ms | 76.6% | 442.7 tok/s |
| 8 | 137ms | 10.49ms | 473ms | 76.6% | 443.1 tok/s |

排队占 TTFT 76.6%（Kairos：排队是时延主成分）；GPU 并发 1→4 吞吐 +22%，4→8 饱和。

- 依据：DistServe（OSDI'24）、Kairos（arXiv:2607.02043）、Vidur（MLSys'24）、
  FastServe（MLSys'24）。设计：`docs/superpowers/specs/2026-08-20-moesim-v9-design.md`。

### Tests

- 133 passed, 2 skipped（新增 6 个请求级模拟测试；INT4 环境性失败仍为 torch 版本相关）。

## [v9.1] - 2026-08-29

### Added — 请求级 KV 生命周期（参考 vLLM PagedAttention / Mooncake KV 管理）

- `sim/request_sim.py`: KV 从池级"只增不减"改为**每请求记账 + 完成即释放**：
  - `_req_kv_gpu` / `_req_kv_host` 追踪每请求的 KV 占用
  - 请求 decode 完成时 `_release_kv` 归还 GPU/主机 KV（PagedAttention block 释放语义）
  - prefill 与 decode 的 KV 都归属请求
- `sim/metrics.py`: `kv_peak_mb` 峰值指标（体现释放后的真实峰值——短请求先完成，
  峰值由长请求 KV 主导，而非所有请求总和）。
- 依据：vLLM PagedAttention（arXiv:2309.06180，KV block 分配/释放）、
  Mooncake（KV cache 生命周期管理）。

### Added — 框架性能基准 + 硬件遥测

- `benchmarks/e2e/benchmark_vllm.py` / `benchmark_llamacpp.py`: 真实框架推理基准
  （TTFT/TPOT/吞吐 + GPU/CPU/内存遥测，经 resource_monitor）。
- `docs/FRAMEWORK_BENCHMARK.md`: 实测对比表（vLLM 28 tok/s GPU / llama.cpp 13.1 tok/s
  CPU / moesim 模拟专家层 1568 tok/s）。
- `docs/vllm-runtime-environment-fixes.md`: vLLM 运行环境 6 个问题的根因+解决表格。

### Tests

- 135 passed, 2 skipped（新增 2 个 KV 生命周期测试；INT4 环境性失败仍在 torch 2.13）。

## [2.0] - 2026-08-30

### Added — 三层存储（VRAM/DRAM/disk）+ KV 磁盘层

- `sim/resources.py`: `StorageTier` + `TieredStorage`（三层存储抽象，含层间搬移
  时间——按源层带宽+延迟计，FlexGen 三层模型）。
- `scheduler/state.py` + `sim/moe_adapter.py`: KV cache 三层——GPU 溢出到 DRAM、
  DRAM 溢出到 disk（`kv_disk_mb` / `kv_host_capacity_mb` / `kv_disk_capacity_mb`）。
- 依据：FlexGen（OSDI'23，GPU/CPU/disk 三层）、MoE-Infinity（SSD→DRAM→GPU 预取）。
- 演示：`benchmarks/e2e/compare_kv_three_tier.py`（KV 随上下文逐层溢出到磁盘）。

### Tests

- 143 passed, 2 skipped（新增 12 个三层存储/KV 测试；INT4 环境性失败仍在 torch 2.13）。

### Added — 专家权重磁盘层（2.0 补全）

- `scheduler/base.py`: 新增 `demote_to_disk` Action（专家从 DRAM 降到 SSD）。
- `scheduler/state.py` + `sim/moe_adapter.py`: `disk_experts` 集合 + `disk_read_gbps` /
  `disk_latency_ms`——磁盘专家 CPU 执行前先 SSD 读（慢路径）。
- `scheduler/policies/disk_tier.py`: `DiskTierPolicy`——按 activation_freq 把最冷专家
  降级到磁盘（disk_budget_mb 控制）。
- 演示：`benchmarks/e2e/compare_expert_disk_tier.py`（冷专家降磁盘的慢读权衡）。

### 诚实边界

- 专家磁盘层的"省 DRAM"价值目前是策略驱动的（降冷专家），**尚未加 DRAM 容量约束**
  （专家权重总量 vs DRAM 容量的硬约束）——完整的三层"超大模型能跑"需此约束，列为 2.0 后续。
- KV 三层已含容量约束（GPU→DRAM→disk 逐层溢出），专家三层待对齐。

### Tests

- 145 passed, 2 skipped。

## [2.3] - 2026-08-30

### Added — 运行时三层调度接入 transformers hook

关键澄清：v3 的 `MoEForwardHook` 已经是**运行时逐专家调度**（decide 每层驱动
CPU/GPU 分派）。2.3 的增量是把 2.0 的三层存储接入这个运行时，让调度从"两层
（CPU/GPU）"升级为"三层（CPU/GPU/disk）"：

- `executor/backends/transformers.py`: `TransformersMoEExecutor` 加磁盘层——
  `disk_experts` 集合 + `_load_from_disk()`（磁盘专家激活时提升到 DRAM）。
- `executor/backends/forward_hook.py`: decide 结果处理 `demote_to_disk` action，
  冷专家标记到 executor 的磁盘层（三层运行时调度闭环）。
- `scheduler/policies/disk_tier.py`: `DiskTierPolicy`（2.0）现在可直接驱动 hook。

### 诚实边界

- 磁盘层的真实实现（mmap 分页、SSD→DRAM→GPU 预取流水线）仍需大内存机器 + 超大
  模型验证；本机（7.6G 内存）只能验证调度决策链路（decide → demote → disk 标记）。
- hook 比 HF 原生慢（逐专家 Python 循环 + 调度开销）是已知的（v3 实测），"运行时
  调度赢 baseline"需显存受限场景（纯 GPU 放不下时逐专家 offload 是必须的）。

### Tests

- 150 passed, 2 skipped（新增 forward-hook 磁盘层链路测试；INT4 环境性失败仍在 torch 2.13）。

## [2.0-complete] - 2026-08-30

### Added — DRAM 容量约束 + 磁盘预取流水线

- `scheduler/policies/disk_tier.py`: `DiskTierPolicy` 加 `dram_capacity_mb`（DRAM
  容量硬约束——非 resident 专家总量超 DRAM 时，最冷专家强制降磁盘）和
  `prefetch_disk_n`（预测式磁盘预取——最热的磁盘专家提前 SSD→DRAM）。
- `scheduler/base.py`: 新增 `prefetch_from_disk` Action。
- `sim/moe_adapter.py`: `_prefetch_from_disk()`——后台 SSD→DRAM 读（不进关键路径），
  完成后专家提升出磁盘层（后续激活不再付慢读）。

至此 2.0 完整：三层存储（VRAM/DRAM/disk）+ KV 三层 + 专家三层 + DRAM 约束 + 磁盘预取，
对应 FlexGen（三层存储）+ MoE-Infinity（SSD→DRAM→GPU 预测预取）的完整图景。

### Tests

- 153 passed, 2 skipped。

## [2.3-complete] - 2026-08-30

### 验证 — 调度决策在真实推理里的价值（非模拟）

`benchmarks/e2e/benchmark_strategy_runtime.py`：用 MoEForwardHook 驱动 4 层 × 8 专家
MoE 模型（CUDA 真实 forward），对比不同调度策略的真实执行时间：

| 策略 | ms/forward |
|---|---|
| all-CPU | 139.688 |
| all-GPU | 49.616 |
| cost_model（CPU 算力感知） | 47.226 |
| **disk_tier（三层调度）** | **47.103** |

**结论**：
1. 混合调度（cost_model/disk_tier）比 all-CPU 快 **2.96x**——这是 moesim 调度决策
   在真实推理里有效性的直接证据（CPU 算力感知把冷专家放 CPU/磁盘，热专家 GPU）。
2. 混合调度比 all-GPU 快 5%（小模型边际优势；真正的价值场景是模型 > 显存时 all-GPU
   不可行，此时逐专家 offload 是唯一解）。

**诚实边界**：验证用 4 层小模型（非 OLMoE 完整），模型可全放 GPU；"混合 > all-GPU"
的完整优势需超大模型（>8G 显存）在大内存机器验证。

### 兼容性记录

- transformers 5.x 的 Olmoe 结构已变（`OlmoeExperts` 无 len、`OlmoeTopKRouter`），
  v3 hook 的 `_forward_3d` 假设旧结构——本验证改用 hook 支持的 w1/w2 专家结构，
  真实 transformers 5.x Olmoe 适配列为后续。
