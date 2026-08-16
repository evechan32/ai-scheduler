# moesim

异构 MoE 专家下放调度器 + 领域无关离散事件模拟器。

> 中文版文档。English version: [README.md](README.md)

## 项目简介

moesim 是领域无关的离散事件模拟（DES）引擎，配合异构计算感知的
Mixture-of-Experts（MoE）专家下放调度器。现有系统把 CPU 下放当作内存兜底；
moesim 则把 **CPU 算力作为一等公民**纳入调度决策——从每个专家的 CPU/GPU 执行
成本与可用带宽出发，联合决策专家放置、PCIe 传输与计算分配。调度器是纯函数
（`decide(state, clock) -> actions`），同一份策略代码同时驱动模拟器与真实推理引擎。

一句话定位：**领域无关的 DES + 异构 MoE 调度器——CPU 算力参与调度决策，不只是内存兜底。**

## 快速开始

```bash
# 安装（模拟器 + 调度器只需 numpy；无需 torch/GPU）
pip install -e .

# 运行测试
python -m pytest tests/

# 运行校准模拟演示（无需 GPU）
python benchmarks/e2e/compare_moesim_vs_llamacpp.py
```

演示加载 `benchmarks/microbench/out/` 中的校准 profile，输出各策略的
TPOT / 吞吐 / 命中率对比表：

```
=== moesim simulation (calibrated) ===
lru              TPOT=   3.693ms  tput= 270.803 tok/s  hit=0.983
activation_freq  TPOT=   6.841ms  tput= 146.180 tok/s  hit=0.983
cost_model       TPOT=   3.456ms  tput= 289.385 tok/s  hit=0.158
```

## 架构

设计分为三层加一个校准回路。权威规范见
`docs/superpowers/specs/2026-08-09-moesim-design.md`（§4）。

- **调度核心**（`moesim/scheduler/`）——可插拔决策函数 `decide(state, clock) -> actions`，
  提供三个 v1 策略（`cost_model`、`activation_freq`、`lru`）与按专家的 `ExpertProfile`
  成本模型（CPU/GPU 执行时间、权重大小 MB、激活频率）。
- **模拟器**（`moesim/sim/`）——领域无关 DES（`core.py`）+ 可复用资源模型
  （`resources.py`：带宽、算力、存储）、校准（`calibrate.py`）、指标聚合（`metrics.py`）。
  无 MoE 依赖；MoE 通过 `moe_adapter.py` 接入。
- **执行器**（`moesim/executor/`）——`ExpertExecutor` 抽象与 transformers 后端，
  推理时将专家路由到 GPU/CPU。
- **校准回路**——微基准测量真实 PCIe 带宽与按专家执行时间，`calibrate.py`
  将其转为模拟器参数。

同一个 `decide()` 同时驱动模拟器与真实引擎。

## 硬件说明

验证环境为 12G GPU + 16G RAM 机器，主力模型 Qwen3-30B-A3B。
`sim/` 与 `scheduler/` 层为 numpy-only，可在无 torch/GPU 环境下运行；
仅执行器后端与微基准需要 torch/CUDA。

## 基准

- `benchmarks/microbench/` —— 真机校准：PCIe 带宽与按专家 CPU/GPU 执行时间。
- `benchmarks/e2e/` —— 端到端验证协议，对比模拟 TPOT/吞吐与 llama.cpp `--n-cpu-moe`
  及 MoE-Infinity 在 Qwen3-30B-A3B 上的结果（目标：误差 < 20%）。

## 路线图（v2）

源自规范 §8：

- 量化 CPU 算子（INT4/INT8）；当前 CPU 算子为 FP16 + torch fallback。
- 通过 `accelerate`/`device_map` 做真实权重下放；v1 中 transformers 执行器
  load/unload 为 no-op。
- 完整 KV cache 分层调度策略，与专家联合调度。
- `vllm` 与 `llama.cpp` 执行器后端。
- 基于 `sim/` 环境训练的 RL / 模仿学习策略。
- 多 GPU / 集群拓扑模拟。

## v2 交付（2026-08-15）

- **真实权重下放** —— transformers 执行器 `load`/`unload` 现在真实搬移专家参数
  GPU↔CPU；`execute_gpu` 自动加载（CUDA 验证通过）。
- **KV 分层感知策略** —— `KVWeightedPolicy` 在成本模型之上叠加 GPU KV 池压力：
  高压时强制 CPU 执行 + KV 驱逐。
- **INT8 CPU 算子** —— `expert_ffn_int8` 采用 per-tensor 对称量化
  （相对 FP16 误差 0.33%，远低于 5% 目标）。
- **vllm / llama.cpp 后端** —— optional-import 适配器 + duck-typed 引擎测试。
- **真机验证协议** —— `benchmarks/e2e/verify_on_real_machine.md`；
  在 OLMoE-1B-7B（MoE，64 专家）上实测：模拟器验证策略相对排序
  （cost_model > lru > activation_freq）；全模型绝对 TPOT 超出当前范围（仅专家层）。

剩余 v2：INT4 量化、`accelerate`/`device_map` 集成、模拟器内 KV+专家联合调度、
真实 vllm/llama.cpp 引擎接线、RL 策略、多 GPU 拓扑。

## v3 交付（2026-08-15）

- **真实推理闭环** —— `MoEForwardHook` 替换 HF MoE 层 forward：router top-k 专家
  经 `scheduler.decide()` 逐专家分派 CPU/GPU，由 transformers 执行器真实执行。
  在真实 OLMoE 结构上验证：hook 与原始 forward 相对误差 **0.0589%**（< 1%）。
  Harness：`benchmarks/e2e/run_hook_inference.py`（26G fp32 模型在 7.6G 开发机
  OOM 时自动降级到结构等价配置）。
- **RL 调度策略** —— `RLScheduler`，numpy-only Q-learning，在模拟器内训练
  （`scheduler/policies/rl.py`），固定 seed 确定性。
- **多 GPU 拓扑** —— `MultiGPUCluster`（节点容量 + 成对带宽矩阵）与
  `ScheduleState` 的 per-GPU 驻留。

剩余：真实 vLLM kernel 级集成、生产级 RL 训练循环、多 GPU 真机验证、
INT4 kernel 优化。

## 完整文档

所有实现、设计决策、性能实测数据汇总在
**`docs/PROJECT_SUMMARY.md`**（中文总文档）。
