# moesim — MoE 异构调度框架 + 通用离散事件模拟器（设计文档）

- 日期：2026-08-09
- 状态：已批准（2026-08-09 头脑风暴确认）
- 作者：Sisyphus / 用户协作

## 1. 背景与动机

在显存受限环境（消费级 GPU + 主机内存）下运行 MoE 大模型时，存在两类下放需求：

1. **专家（expert）下放**：MoE 模型的专家权重与计算可在 GPU 与 CPU（主机内存）之间搬移，激活稀疏性使得每步只需部分专家驻留 GPU。
2. **KV cache 分层**：KV cache 可在 GPU / 主机内存 / 存储之间分层搬移，支撑长上下文。

现有系统各自解决一部分：MoE-Infinity（专家 trace 预取）、Fiddler（CPU 计算划分）、SwapMoE（专家缓存替换）、llama.cpp `--n-cpu-moe`（工程实现）、SGLang HiCache / LMCache（KV 分层）。**没有一个系统把"专家放置 + 计算划分 + KV 分层"统一到一个可插拔的调度框架中，且以 CPU 算力感知为核心决策依据。**

本项目 `moesim` 的目标：
- 交付一个**异构计算感知的 MoE 下放调度器**（CPU 算力计入决策，按专家算力需求动态划分计算位置）；
- 交付一个**领域无关的通用离散事件模拟器**（`sim/`），作为策略开发、验证、复现的基础设施——MoE 调度是它的第一个领域应用，而非唯一用户；
- 以**开源项目**标准交付（文档、API 设计、可复现基准、无硬件可跑的 demo）。

## 2. 验证环境与约束

- **开发/验证硬件**：12G 显存 GPU + 16G 主机内存（严格约束）；规划上兼容 24G + 200G 服务器。
- **主力验证模型**：Qwen3-30B-A3B（Q4 ~18GB，12G GPU + ~6G 内存分载，激活 3B）；对照 Qwen3-14B-A3B（Q4 ~9GB，全 GPU 基线）。
- **PCIe 带宽是硬瓶颈**：下放方案的吞吐上限由 PCIe 决定，模拟器必须显式建模。
- 12G/16G 机器上无法运行 DeepSeek-V3/R1（671B 级），此类模型仅作为模拟器中的抽象画像验证。

## 3. 设计决策摘要（头脑风暴结论）

| 维度 | 决策 |
|---|---|
| 调度对象 | 专家放置/预取/缓存替换 **+** KV cache 分层，统一决策 |
| 核心创新 | 异构计算感知调度（CPU 算力计入决策） |
| 载体 | 独立调度层（不绑定推理框架） |
| 执行交互 | 通用 `ExpertExecutor` 接口，后端可插拔（transformers 先，vllm/llama.cpp 后续） |
| CPU 算子 | 自研 PyTorch C++ 扩展，先 FP16 后量化 |
| 验证方式 | 调度策略先在确定性模拟器上开发/验证，再真机终验 |
| 产出 | 开源项目 |
| 项目名 | `moesim` |

## 4. 架构总览

```
┌─────────────────────────────────────────────────────────┐
│                    user app / benchmark                  │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                Scheduler Core（调度核心）                  │
│   decide(state, clock) → actions   纯函数，可插拔策略      │
│   policies/: cost_model · activation_freq · lru          │
│   cost_model: ExpertProfile（CPU/GPU/PCIe 参数表）        │
└─────────────┬──────────────────────────┬────────────────┘
              │ 同一 decide() 双端复用      │
┌─────────────▼───────────┐   ┌──────────▼───────────────┐
│  sim/ 通用离散事件模拟器   │   │  executor/ 执行层          │
│  （与 MoE 无关的可复用层） │   │  ExpertExecutor 抽象       │
│  - Event / Clock / DES   │   │  - GPUExecutor           │
│  - Resource 模型族        │   │  - CPUExecutor（自研算子） │
│  - 带宽/队列/竞争建模      │   │  - 后端适配: transformers  │
└─────────────┬───────────┘   │    vllm / llama.cpp (v2)   │
              │               └──────────┬───────────────┘
              └────── 校准闭环（微基准 → 模拟参数）──────────┘
```

## 5. 模块分解

### 5.1 `sim/` — 通用离散事件模拟器（可复用核心）

**设计原则：与 MoE 领域解耦。** `sim/` 不 import 任何 MoE 概念；MoE 通过"注册资源 + 实现 decide 回调"接入。

- `sim/core.py`：
  - `Event`：带时间戳的不可变动作，`(time, priority, payload)`。
  - `EventQueue`：基于 `heapq` 的优先级队列，按时间弹出。
  - `Clock`：单调虚拟时间。
  - `Simulation`：`run()` 循环——弹出最早事件 → `dispatch(event)` → 回调产生新事件。回调由用户注册（调度器/资源均以回调接入）。
- `sim/resources.py`：资源模型族（领域无关）：
  - `BandwidthResource`：带宽/延迟模型，`transfer_time(mb) = mb/bw + latency`；并发请求排队（FIFO），竞争自然叠加。
  - `ComputeResource`：串行/并行计算单元，`process_time(work)`；支持线程池语义（并发度上限）。
  - `StorageResource`：分层存储（L1/L2/L3），带逐出策略钩子。
- `sim/calibrate.py`：从微基准数据生成资源参数（带宽、延迟、专家执行时间表）。
- `sim/metrics.py`：指标聚合——延迟、吞吐、利用率、排队长度直方图。

**可复用性保证**：资源模型与事件循环不依赖领域语义；任何"资源受限 + 异步搬移 + 调度决策"问题（KV 分层、分布式缓存、数据管线）都能套用。

### 5.2 `scheduler/` — MoE 调度核心

- `scheduler/base.py`：`Scheduler` 抽象——`decide(state, clock) → list[Action]`。
  - 纯函数契约：输入 `state`（可序列化），输出 `actions`（可序列化）。**模拟器与真实引擎走同一代码路径**；同时为未来 RL 训练保留干净的 state→action 接口。
  - `Action` 类型：`LoadExpert(experts)`、`UnloadExpert(experts)`、`ExecuteGPU(expert, batch)`、`ExecuteCPU(expert, batch)`、`EvictKV(page)`、`FetchKV(page)`。
- `scheduler/state.py`：`ScheduleState`——专家驻留表、请求队列、KV 占用、资源余量（可序列化 dataclass）。
- `scheduler/cost_model.py`：`ExpertProfile`（每个专家的 GPU 执行时间、CPU 执行时间、权重大小 MB、激活频率统计）；由校准数据填充。
- `scheduler/policies/`（v1 三个可对比策略）：
  - `cost_model.py`：**异构感知主策略**——以专家执行成本 + PCIe 搬运成本 + CPU/GPU 计算能力对比做放置决策（Fiddler 思路 + PowerInfer hot/cold + MoE-Infinity trace 预取，统一到本框架）。
  - `activation_freq.py`：按历史激活频率预取/淘汰（LFU 式）。
  - `lru.py`：最近最少使用（基线）。
- KV 分层：v1 在 `ScheduleState` 中建模 KV 占用与 evict/fetch 动作，策略层可感知；**完整的 KV 分层调度策略 v2 交付**（见 §8）。

### 5.3 `executor/` — 执行层

- `executor/base.py`：`ExpertExecutor` 抽象——`load(experts)`、`unload(experts)`、`execute_gpu(expert, hidden_states)`、`execute_cpu(expert, hidden_states)`。
- `executor/cpu_kernels/`：自研 PyTorch C++ 扩展，FP16 专家 FFN（先 FP16，量化 v2）。
- `executor/backends/transformers.py`：替换 HF MoE 层 forward 的适配器（MoE-Infinity 同款路径），v1 交付。
- `executor/backends/vllm.py`、`llama_cpp.py`：v2。
- `executor/kv_manager.py`：GPU/主机内存 KV 分层管理的真实实现（v2 完整；v1 提供接口骨架）。

### 5.4 `benchmarks/` — 校准与对比

- `microbench/`：测量真实 CPU/GPU/PCIe 参数（专家执行时间、PCIe 带宽、内存带宽），输出 JSON 喂给 `sim/calibrate.py`。
- `e2e/`：端到端对比脚本——`moesim` vs llama.cpp `--n-cpu-moe` vs MoE-Infinity（Qwen3-30B-A3B）。

## 6. 数据流（MoE 领域接入示例）

1. 请求进入推理引擎 → 模型 forward 到达 MoE 层 → router 选出 top-k 专家。
2. 调度器 `decide(state, clock)` 输出动作：GPU 驻留专家直接 `ExecuteGPU`；CPU 侧专家 `ExecuteCPU`；缺失专家触发 `LoadExpert`（PCIe 搬移）；必要时 `UnloadExpert` / `EvictKV` / `FetchKV`。
3. 执行层按动作执行；模拟器模式下同一批动作被解析成模拟事件（加载完成/执行完成）。
4. 每步结束更新 `ScheduleState`，进入下一步。

## 7. 测试策略

- **单元测试**（无硬件）：每个策略在确定性模拟器上的行为——可断言精确数值（如缓存命中率、TPOT）。
- **校准测试**：模拟 TPOT vs 实测 TPOT 误差 < 20% 为通过；锁在 CI 上（真机标记为可选）。
- **端到端**（12G/16G 真机）：Qwen3-30B-A3B 跑通 `moesim`，与 llama.cpp/MoE-Infinity 对比。
- 模拟器确定性保证：同一输入两次运行结果逐位相同（无随机、无时间依赖）。

## 8. 交付边界

**v1（MVP）**：
- `sim/` 完整（core + resources + calibrate + metrics）
- `scheduler/`：3 个策略 + cost_model + state
- `executor/`：CPU FP16 算子 + transformers 适配 + ExpertExecutor 抽象
- `benchmarks/`：微基准 + 端到端对比（llama.cpp、MoE-Infinity）
- 文档：README、API 文档、设计文档

**v2（扩展点，不在 v1 范围）**：
- 量化 CPU 算子（INT4/INT8）
- KV cache 分层调度策略（与专家联合调度的完整实现）
- vllm / llama.cpp 后端适配
- RL / 模仿学习策略（基于 `sim/` 做训练环境）
- 多 GPU / 集群拓扑模拟

## 9. 参考工作

- MoE-Infinity（arXiv:2401.14361）：请求级 trace 专家预取/缓存
- Fast Inference of MoE with Offloading / mixtral-offloading（arXiv:2312.17238）：LRU 专家缓存 + 推测预取
- Fiddler（arXiv:2402.07033，ICLR 2025）：CPU-GPU 编排，AVX512_BF16 CPU 内核
- SwapMoE（arXiv:2308.15030，ACL 2023）：可调内存预算的专家交换
- HOBBIT（arXiv:2411.01433）：混合精度专家下放
- PowerInfer（arXiv:2312.12456，SOSP）：hot/cold 神经元划分
- Pre-gated MoE（arXiv:2308.12066，ISCA）：推测路由预取
- SGLang HiCache、LMCache：KV 分层
- llama.cpp `--n-cpu-moe`：工程基线
- ktransformers：CPUinfer 异构推理（参考 CPU 执行性能基线）

## 10. 项目布局

```
moesim/
├── sim/                # 通用离散事件模拟器（领域无关）
│   ├── core.py         # Event / EventQueue / Clock / Simulation
│   ├── resources.py    # Bandwidth / Compute / Storage 资源模型
│   ├── calibrate.py    # 微基准 → 参数
│   └── metrics.py      # 指标聚合
├── scheduler/          # MoE 调度核心
│   ├── base.py         # Scheduler 抽象 + Action 类型
│   ├── state.py        # ScheduleState
│   ├── cost_model.py   # ExpertProfile
│   └── policies/       # cost_model / activation_freq / lru
├── executor/           # 执行层
│   ├── base.py         # ExpertExecutor 抽象
│   ├── cpu_kernels/    # 自研 PyTorch C++ 扩展（FP16）
│   ├── kv_manager.py   # KV 分层管理接口骨架
│   └── backends/       # transformers（v1）/ vllm, llama_cpp（v2）
├── benchmarks/
│   ├── microbench/     # CPU/GPU/PCIe 参数测量
│   └── e2e/            # 端到端对比
├── tests/              # 单测（无硬件）
├── docs/
│   └── superpowers/specs/
└── README.md
```
