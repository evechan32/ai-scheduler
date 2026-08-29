# moesim 框架接入设计规范（异构调度 + KV offload 接入开源框架）

- 日期：2026-08-29
- 前置：模拟器阶段（v1-v9）已完成；本文档规划"真实框架接入"阶段
- 现状真相：见 `docs/FRAMEWORK_BENCHMARK.md` §3（调度器尚未接入任何框架）

## 1. 目标

让 moesim 的**异构算力调度**与 **KV cache offload 调度**真正在开源框架（vLLM /
llama.cpp）里生效，并**用同一模型、同一输入证明相对 baseline 有提升**。

## 2. 接入点研究结论（关键发现）

### 2.1 vLLM：现成接口，无需改 vLLM 代码（最短路径）

vLLM 0.27 已内置三类 offload 能力，粒度与 moesim 调度决策精确对齐：

| vLLM 接口 | 位置 | 粒度 | 对应 moesim |
|---|---|---|---|
| `uva.cpu_offload_params: set[str]` | `vllm/config/offload.py:34` | **参数名段子串匹配**，逐参数粒度 | 专家放置（decide → 冷专家参数段集合） |
| `prefetch.offload_params` + `offload_group_size` + `offload_prefetch_step` | `offload.py:54-76` | 层组 + 异步 H2D 预取 | prefetch 重叠（v6） |
| `SimpleCPUOffloadScheduler` | `v1/simple_kv_offload/manager.py:67` | KV block 级 LRU / lazy / async load | KV 分层调度（v8） |

**匹配逻辑确认**（`vllm/model_executor/offloader/uva.py:92`）：

```python
should_offload = any(f".{param}." in f".{name}." for param in self.cpu_offload_params)
```

点号包裹的**段子串匹配**——`param="layers.0.experts.3"` 精确匹配参数名
`model.layers.0.mlp.experts.3.gate_proj.weight`，且不含 `.3.` 之外（不会误伤 expert 3 的
其他 tensor 如 `weight_scale`）。因此 **moesim 的逐专家逐层放置可 100% 精确映射**。

### 2.2 llama.cpp：接入更困难

`--n-cpu-moe`（专家粒度）底层走 `override_tensor` 机制，但 llama-cpp-python 0.3.35
**未暴露**（仅 `n_gpu_layers` 层粒度）。接入需改 C++ 或绑定，列为后置阶段。

### 2.3 本质限制

vLLM offload 是**加载时静态**配置，moesim 的运行时动态 `decide()` 不能直接映射。
但 moesim 模拟器可以算出**最优静态放置计划**（冷专家 offload、热专家留 GPU），映射成
vLLM 配置——这已优于 vLLM 默认的"非选择性 offload 直到 `cpu_offload_gb` 满"。

## 3. 接入方案（分阶段）

### 阶段 1（可立即落地）：vLLM 静态专家放置映射

1. moesim 模拟器用 OLMoE 校准参数（GPU 0.076ms / CPU 0.639ms / PCIe 4.3GB/s），在
   8G 显存约束下计算最优静态专家放置（按 activation_freq：热专家留 GPU，冷专家 CPU）。
2. 映射成 `cpu_offload_params = {"layers.{i}.experts.{j}" for 冷专家}`。
3. vLLM 加载 OLMoE（bf16 13G，8G 显存装不下 → 靠 offload 放下）。
4. **对比（同一模型、同一输入）**：
   - vLLM 默认（无 offload → OOM，或全 offload → 慢）
   - vLLM + moesim 放置（冷专家 offload，热专家 GPU）
5. 验证指标：TPOT / 吞吐 / GPU 显存占用 / CPU 利用率（复用 resource_monitor）。

### 阶段 2：vLLM KV offload 参数映射

moesim v8 的 KV 分层调度（压力阈值、驱逐策略）映射到 `SimpleCPUOffloadScheduler`
的 lazy/eager 模式 + offload 池大小。对比 vLLM 默认 KV offload vs moesim 参数。

### 阶段 3（难，后置）：动态调度真接入

改造 vLLM `FusedMoE` 层或 llama.cpp 调度点，暴露逐专家运行时钩子，让 moesim
`decide()` 驱动运行时放置（当前 offload 是静态的）。工作量最大，列为远期。

## 4. 验证方案（诚实标准）

- **同一模型**（OLMoE-1B-7B）、**同一输入**、**同一硬件**（8G 显存 + 16G 内存）。
- 基线必须是**同等显存约束下**的公平对比（不是"纯 GPU 放不下"这种偷换）。
- 明确标注：哪些是静态放置映射（阶段 1），哪些是动态调度（阶段 3，未做）。
- 若阶段 1 结果"moesim 放置 ≈ vLLM 默认"，如实记录，不夸大。

## 5. 不承诺的事项

- 不修改 vLLM 代码（阶段 1/2 只用现有配置接口，避开 vLLM 贡献政策）。
- 不宣称"动态调度提升"（那是阶段 3 的目标，当前无实现）。
