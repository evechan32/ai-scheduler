# moesim v3 — 真实引擎联调 + RL 策略 + 多 GPU 拓扑 Design

> 2026-08-15。前置：v1（41 测试）+ v2（63 测试）完成，PR #1 已合并（511c7da）。

## 1. 目标

把调度器接入真实模型推理，形成"调度决策 → 真实异构执行"闭环；随后以 `sim/` 为环境训练 RL 调度策略；最后扩展多 GPU 拓扑模拟。三方向按 联调 → RL → 多GPU 顺序推进。

## 2. 关键决策（头脑风暴确认）

- **载体选择**：替换 HF transformers 的 `OlmoeSparseMoeBlock.forward`（Python 级、逐专家可控）——vLLM/SGLang 是 Triton/CUDA fused kernel 黑盒无法插入逐专家分派；llama.cpp 是编译期固定 `--n-cpu-moe`。HF 是唯一能验证"自研调度器驱动异构执行"的载体（KTransformers 同款路径）。
- **调度时机**：每层 MoE 前向实时调度——router 选出 top-k 专家 → `scheduler.decide(state, clock)` → 按决策分派。
- **模型**：OLMoE-1B-7B-0125（MoE：7B 总参/1.3B 激活/64 专家/8 每 token），safetensors（26G）+ GGUF 已下载到 `/home/qyw/models/`。

## 3. 架构

```
┌──────────────────────────────────────────────────────────┐
│  TransformersMoEExecutor (v2: load/unload 真实搬移)       │
│    └─ MoEMoEForwardHook: 替换 OlmoeSparseMoeBlock.forward │
│         router 选 top-k → scheduler.decide() → 分派       │
│         已驻留→execute_gpu；决策CPU→execute_cpu(INT8/FP16) │
│         缺失→load 后执行；结果合并回 HF 主流程             │
└──────────────────────────────────────────────────────────┘
```

## 4. 模块分解

### 4.1 `executor/backends/forward_hook.py` — MoE 前向钩子（v3 核心）

- `MoEForwardHook(executor, scheduler, profiles, pcie)`：
  - `install(model)`: 替换每个 `OlmoeSparseMoeBlock` 的 `forward`，保存原 forward。
  - `forward_impl(self, hidden_states, expert_mask=None, ...)`: 用原 router 选专家 → 构造 `requested` → `scheduler.decide()` → 执行。
  - `uninstall(model)`: 恢复原 forward。
- 与 `TransformersMoEExecutor` 协作：executor 提供 `execute_gpu/execute_cpu/load/unload`。
- 正确性验证：hook 输出 vs 原始 forward 输出数值对比（tolerance）。

### 4.2 `scheduler/policies/rl.py` — RL 调度策略（v3 第二方向）

- Gym 风格：`sim/` 作为环境，`decide` 作为动作空间（load/unload/execute_cpu/execute_gpu）。
- 先做简单 Q-learning/模仿学习（numpy-only），目标：在模拟中接近或超过 cost_model 策略。
- numpy-only 约束保持（RL 用 numpy 实现，不引入 torch/gym 依赖）。

### 4.3 `sim/resources.py` + `scheduler/state.py` — 多 GPU 拓扑（v3 第三方向）

- `resources.py`: 新增多 GPU 节点建模（每 GPU 独立容量/带宽）。
- `state.py`: `ScheduleState` 记录 per-GPU 驻留（`resident` 从 set 扩展为 per-gpu 结构，兼容旧字段）。
- 策略感知 GPU 间带宽差异。

## 5. 验证

- 单测（无硬件）：hook 逻辑在 mini-MoE 上正确分派；RL 策略在模拟中收敛；多 GPU 状态正确。
- 联调（真机）：OLMoE-1B-7B 前向通过 hook 执行，输出 vs 原始 forward 数值一致（相对误差 <1%）。
- 全量：v1+v2 63 测试保持通过；新增测试全绿。
- 约束：sim/scheduler numpy-only（torch 仅限 executor）；TDD；确定性。

## 6. 边界

- v3 交付：forward hook + RL 基础策略 + 多 GPU 模拟扩展。
- 明确不在 v3：INT4 内核优化（v2 已交付 FP16+INT8+INT4 基础版）、真实 vLLM kernel 级接入、生产级 RL 训练循环。

## 7. 风险

- OLMoE 前向数值对比：hook 重排专家执行顺序不影响数学结果（专家独立 FFN，求和顺序差异在浮点容差内）。
- RL 收敛时间：用模拟器（毫秒级）而非真机训练，控制回合数。
- 多 GPU：本机单卡，仅模拟层验证，真机多卡验证 deferred。
