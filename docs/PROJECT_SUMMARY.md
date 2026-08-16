# moesim — 异构 MoE 调度框架完整文档

> 项目状态：**v1 + v2 + v3 全部完成** | 71/71 测试通过 | PR #1/#2 已合并至 main
> 本文档汇总全部实现、设计决策、性能实测数据与运行方式。

---

## 1. 项目定位

**moesim** 是领域无关的离散事件模拟器 + 异构计算感知的 MoE 专家下放调度器。
核心目标（用户原始需求）：**结合 CPU 与 GPU 算力和内存的调度框架来执行推理**。

关键设计立场：**CPU 算力作为一等公民参与调度决策**（不仅是内存兜底）。
调度器是纯函数 `decide(state, clock) -> actions`，同一份策略代码同时驱动模拟器与真实推理引擎。

## 2. 架构总览

```
┌────────────────────────────────────────────────────────────┐
│  调度层 scheduler/  (numpy-only)                           │
│    decide(state, clock) -> [Action]                        │
│    策略: cost_model / activation_freq / lru / kv_aware / rl│
│    状态: ScheduleState (residency, KV tier, per-GPU)       │
├────────────────────────────────────────────────────────────┤
│  模拟层 sim/  (numpy-only, 领域无关)                       │
│    core (DES) / resources (带宽/算力/存储/多GPU)           │
│    calibrate / metrics / moe_adapter / sweep               │
├────────────────────────────────────────────────────────────┤
│  执行层 executor/  (torch 例外)                            │
│    ExpertExecutor 抽象                                     │
│    backends: transformers / vllm / llama_cpp / forward_hook│
│    cpu_kernels: FP16 C++ / INT8 / INT4                     │
│    kv_manager: KV 层记账                                    │
└────────────────────────────────────────────────────────────┘
```

## 3. 版本演进与实现记录

### v1 — 模拟器 + 调度器基础（41 测试）

| 模块 | 交付 |
|---|---|
| `sim/core.py` | 领域无关离散事件模拟器（时钟推进、确定性） |
| `sim/resources.py` | BandwidthResource / ComputeResource / StorageResource |
| `sim/calibrate.py` | 校准参数加载（JSON profile） |
| `sim/metrics.py` | TPOT / 吞吐 / 命中率聚合 |
| `sim/moe_adapter.py` | MoE 领域接入（requested 专家 → decide → 执行计时） |
| `sim/sweep.py` | `compare_policies` 多策略对比 harness |
| `scheduler/base.py` | `Action`（load/unload/execute_gpu/execute_cpu/evict_kv/fetch_kv）、`Scheduler` 契约 |
| `scheduler/state.py` | `ScheduleState`（resident、访问历史、cache hit/miss） |
| `scheduler/cost_model.py` | `ExpertProfile`（size_mb / gpu_exec_ms / cpu_exec_ms / activation_freq） |
| `scheduler/policies/lru.py` | LRU 缓存策略 |
| `scheduler/policies/activation_freq.py` | 激活频率预取策略 |
| `scheduler/policies/cost_model.py` | **异构计算感知**策略：`cpu_exec_ms <= load_cost + gpu_exec_ms` → CPU 执行 |
| `executor/base.py` | `ExpertExecutor` 抽象 |
| `executor/cpu_kernels/__init__.py` | 自研 FP16 CPU expert FFN（C++ 扩展 + torch fallback） |
| `executor/kv_manager.py` | KV 层记账骨架 |

**v1 真实测量**（RTX 5070, CUDA 13.1）：PCIe 4.30 GB/s；expert GPU 0.224ms / CPU 3.456ms。

**v1 策略对比结论**：PCIe 慢时 cost_model 胜（CPU 执行更优）；PCIe 快时 lru 胜（GPU 驻留碾压）。

### v2 — 执行闭环补全（63 测试）

| 模块 | 交付 |
|---|---|
| `executor/backends/transformers.py` | **真实权重 offload**：load/unload 真实搬移专家参数 GPU↔CPU（v1 是 no-op） |
| `scheduler/policies/kv_aware.py` | `KVWeightedPolicy`：GPU KV 压力 >0.9 时强制 CPU 执行 + KV 驱逐 |
| `executor/cpu_kernels/quantized.py` | INT8（误差 0.33%）+ INT4（2-per-byte 打包，误差 8.11%）量化 kernel |
| `executor/backends/vllm.py` | vLLM 后端（optional import，duck-typed 测试） |
| `executor/backends/llama_cpp.py` | llama.cpp 后端（同上） |
| `sim/moe_adapter.py` + `scheduler/base.py` | 模拟器消费 evict_kv/fetch_kv（KV 转移计入 PCIe 计时） |
| `executor/backends/transformers.py` | accelerate 可选集成（set_module_tensor_to_device） |

**v2 修复的真实缺陷**（任务审查发现）：
- load_inline 重复 PYBIND11_MODULE 导致 C++ 编译静默失败
- torch_extensions 残留 lock 文件导致加载挂起
- e2e trace 硬编码专家 ID（越界 KeyError）
- KV pressure 计算与测试矛盾

### v3 — 真实推理闭环 + RL + 多 GPU（71 测试）

| 模块 | 交付 |
|---|---|
| `executor/backends/forward_hook.py` | **MoEForwardHook**：替换 HF MoE 层 forward，router top-k 专家经调度器分派 CPU/GPU |
| `benchmarks/e2e/run_hook_inference.py` | 真实模型联调 harness（26G 模型 OOM 自动降级结构等价配置） |
| `scheduler/policies/rl.py` | `RLScheduler`：numpy-only Q-learning，在模拟器中训练 |
| `sim/resources.py` | `MultiGPUCluster`：节点容量 + 带宽矩阵 |
| `scheduler/state.py` | per-GPU 驻留（gpu_resident） |

**v3 修复的真实缺陷**（混合执行暴露）：
- hook 不支持真实 OLMoE 3D 输入 + per-token top-k + tuple 契约
- `execute_cpu` 不支持真实 OlmoeMLP（gate_proj/up_proj/down_proj）
- 跨层 residency 状态污染（混合执行特有 bug）
- uninstall 恢复 unbound forward 导致崩溃

## 4. 性能实测数据

### 4.1 真实推理闭环数值正确性（v3 核心）

**OLMoE-1B-7B 结构**（64 专家 / 8 每 token / 16 层）：
```
MoEForwardHook 驱动 → 与 HF 原始 forward 相对误差 0.064% ✓（目标 <1%）
```

### 4.2 CPU+GPU 混合执行实测（统一小模型 2层/8专家/512hidden, 5 tokens）

| 模式 | ms/forward | 说明 |
|---|---|---|
| HF 原生 CPU | 8.30 | 基线（无调度） |
| HF 原生 GPU | 12.53 | 小模型 kernel 启动开销 |
| moesim 全 CPU 执行 | 54.43 | 调度开销 + CPU 执行 |
| moesim 全 GPU 驻留 | 34.84 | |
| **moesim 混合 4GPU+4CPU** | **29.47** | 🏆 框架内最快 |

结论：**CPU 与 GPU 同时参与计算已真实运行**（同一前向 4 专家 GPU + 4 专家 CPU），
混合模式数值正确性（vs 全 GPU 参考误差 0.049%）。moesim 比 HF 原生慢是调度决策
开销（每层 decide），这是调度能力的代价，后续可优化（决策缓存/批量调度）。

### 4.3 llama.cpp 大上下文性能（OLMoE-1B-7B Q3_K_L GGUF, CPU 版）

| 上下文 | TPOT | 吞吐 |
|---|---|---|
| 短（1 token 前缀） | 131.8 ms/tok | 7.6 tok/s |
| 长（3001 token 前缀） | 954.4 ms/tok | 1.0 tok/s（7.2x 衰减） |

精度验证：15×13=195 ✓；常识问答 ✓。

### 4.4 量化 kernel 精度

| kernel | 相对误差 | 阈值 |
|---|---|---|
| FP16（C++ 扩展） | 0.0% | — |
| INT8 | 0.33% | <5% |
| INT4（2-per-byte 打包） | 8.11% | <10% |

## 5. 运行方式

```bash
# 安装
pip install -e .

# 测试（71/71）
python -m pytest tests/ -q

# 模拟器策略对比（无需 GPU）
python benchmarks/e2e/compare_moesim_vs_llamacpp.py

# 真实推理闭环验证（OLMoE 结构，26G 模型 OOM 自动降级）
python benchmarks/e2e/run_hook_inference.py

# 硬件校准（需 CUDA GPU）
python benchmarks/microbench/measure_pcie.py
python benchmarks/microbench/measure_expert_time.py
```

约束：`sim/` + `scheduler/` 保持 numpy-only（无 torch）；`executor/` 是 torch 例外；
确定性保证（同输入 ⇒ 逐位相同输出）；单位 MB / GB/s / ms。

## 6. 设计文档索引

| 文档 | 内容 |
|---|---|
| `docs/superpowers/specs/2026-08-09-moesim-design.md` | 主设计规范（架构、模块分解、v1/v2 边界、参考工作） |
| `docs/superpowers/specs/2026-08-15-moesim-v3-design.md` | v3 设计（真实推理 hook、RL 策略、多 GPU） |
| `docs/superpowers/plans/2026-08-09-moesim-mvp.md` | v1 实施计划（15 任务，含缺陷修正记录） |
| `docs/superpowers/plans/2026-08-15-moesim-v2.md` | v2 实施计划（Task 16-24） |
| `docs/superpowers/plans/2026-08-15-moesim-v3.md` | v3 实施计划（Task 25-29） |
| `benchmarks/e2e/verify_on_real_machine.md` | 真机验证协议（含硬件限制与实测记录） |

## 7. 关键设计决策记录

1. **调度器载体选择**：替换 HF transformers MoE forward（Python 级、逐专家可控）——
   vLLM/SGLang 是 Triton/CUDA fused kernel 黑盒无法插入逐专家分派；llama.cpp 编译期固定。
   HF 是唯一能验证"自研调度器驱动异构执行"的载体（KTransformers 同款路径）。
2. **调度时机**：每层 MoE 前向实时调度（router 选专家 → decide → 分派）。
3. **CPU 算力入决策**：`cpu_exec_ms <= load_cost + gpu_exec_ms` → 放 CPU（v1 起）。
4. **模拟器领域无关**：sim/ 无 MoE 依赖，通过 moe_adapter 接入（用户要求可复用）。
5. **确定性优先**：无随机、无时间依赖；RL 用固定 seed。

## 8. 剩余路线图

- 真实 vLLM kernel 级集成（当前后端为记账式）
- 生产级 RL 训练循环（当前为模拟器内 Q-learning）
- 多 GPU 真机验证（当前仅模拟层）
- INT4 kernel 优化（当前为精度优先反量化路径）
- 调度决策开销优化（决策缓存/批量调度，缩小与 HF 原生的差距）

## 9. v4 增强（2026-08-16，75 测试）

| 提交 | 内容 |
|---|---|
| e3787a1 | 决策缓存——单 forward 内 decide 复用（减少调度开销） |
| eae3955 | **真并行执行**——线程池并发 CPU/GPU 专家（混合 22.12ms，全 GPU 21.85ms） |
| 9aa3e0c | 一键安装脚本 install.sh + pyproject extras + 中文 README |
| 529a472 | MoE 推理优化论文综述（docs/research/，50+ 篇） |
| 3e10c1a | REINFORCE 策略梯度调度器（生产级 RL，替代 Q-learning） |
| d61a835 | INT4 kernel 真 int8 gemm（torch._int_mm，含 CPU/CUDA 回退） |

**v4 性能实测（统一小模型，30 次平均）：**
- 混合串行 23.89ms → 混合真并行 **22.12ms**
- 全 GPU 串行 34.84ms → 全 GPU 真并行 **21.85ms**
- INT4 gemm 相对误差 <10%（含激活 int8 量化 + 权重解包）

**v4 优化方向依据**（论文综述映射）：APEX 异步并行、Dovetail CPU/GPU 投机分工、QuantMoE-Bench 量化、TriMoE 三路异构。
