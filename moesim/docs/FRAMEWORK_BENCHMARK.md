# 框架性能基准 + 硬件遥测对比

> 记录时间：2026-08-29 | 环境：WSL + RTX 5070 Laptop 8GiB（SM 12.0）
> 脚本：`benchmarks/e2e/benchmark_vllm.py`、`benchmark_llamacpp.py`、
> `compare_request_concurrency.py`（硬件遥测见 `benchmarks/microbench/resource_monitor.py`）

## 1. 同一模型、同一输入的真实对比（llama.cpp CUDA，OLMoE-1B-7B Q3_K_L）

> 2026-08-29 补测：用户指正此前对比混淆了模型/硬件。此为**同一 GGUF、同一 prompt**、
> 纯 CPU / 层粒度异构 / 纯 GPU 三档真实数据（llama.cpp 0.3.35 CUDA 版，SM120 编译）。

| 配置（n_gpu_layers） | 加载 | TPOT | 吞吐 | GPU util(mean/max) | GPU 显存 | CPU util |
|---|---|---|---|---|---|---|
| **纯 CPU**（0） | 5.0s | 40.88ms | 24.5 tok/s | 0.1% / 2% | 312 MiB | 36.5% |
| 异构（8 层 GPU） | 1.0s | 24.21ms | 41.3 tok/s | 17.9% / 97% | 1866 MiB | 45.7% |
| 异构（16 层 GPU） | 1.1s | 7.17ms | 139.6 tok/s | 46.1% / 74% | 5060 MiB | 32.7% |
| **纯 GPU**（99） | 1.1s | 5.58ms | 179.1 tok/s | 67.9% / 86% | 6960 MiB | 8.1% |

**结论（诚实版）**：
1. 纯 GPU 比纯 CPU 快 **7.3x**，代价是 6960 MiB 显存（8G 的 85%）——模型再大（fp16
   13G、Mixtral 8x22B 等）就放不下，纯 GPU 不可行。
2. 异构是**显存-性能权衡曲线**：16 层 GPU 用 73% 显存达到 78% 纯 GPU 性能（139.6 vs
   179.1）；8 层 GPU 用 27% 显存达到 23% 性能。
3. 这是 llama.cpp 的**层粒度**异构（整层 GPU 或 CPU）。moesim 是**专家粒度**（每个
   专家独立 CPU/GPU），在同样显存预算下能更精细地放置热专家——这是 moesim 相对
   llama.cpp `-ngl` 的差异点，见 §2 模拟对比。

数据源：`benchmarks/e2e/out/llamacpp_cpu_gpu_hybrid.json`；
脚本：`benchmarks/e2e/compare_cpu_gpu_hybrid_llamacpp.py`。

| 框架 | 模型 | 计算硬件 | 加载 | TPOT | 吞吐 | GPU util | CPU util | 内存占用 |
|---|---|---|---|---|---|---|---|---|
| **vLLM 0.27.2rc1** | Qwen3.5-2B（dense 2B 多模态） | GPU (SM12) | 76.1s | 35.7ms | **28.0 tok/s** | 29.6%（峰值 100%） | 10.2% | GPU 4.4–7.4 GiB，sys 77% |
| **llama.cpp 0.3.34** | OLMoE-1B-7B（MoE 64 专家，1.3B active） | CPU（多线程） | 3.7s | 76.2ms | **13.1 tok/s** | 0%（CPU 版） | 49.2%（峰值 100%） | sys 42% |
| **moesim（模拟）** | OLMoE-1B-7B 专家层（64 专家，8/token） | —（模拟） | — | 0.638ms | 1568 tok/s | — | — | 模拟值 |

## 2. 关键结论

1. **GPU vs CPU 异构**：vLLM（GPU）跑 dense 2B = 28 tok/s；llama.cpp（CPU）跑 MoE
   （1.3B active）= 13.1 tok/s。模型不同、硬件不同，**绝对吞吐不可直接比较**，但有
   参考意义的资源画像：GPU decode 是 memory-bound（GPU util 仅 29.6%），CPU 推理
   多线程吃满（CPU util 峰值 100%）。
2. **moesim 模拟的是专家层 only**（8 专家 × 0.076ms ≈ 0.64ms），不含 attention /
   embedding / 其余 15 层，**绝对 TPOT 与完整模型不可比**（与
   `verify_on_real_machine.md` 的诚实标注一致）。moesim 价值在于**相对策略排序**与
   异构调度决策的确定性验证，而非绝对时延预测。
3. **vLLM decode 慢的真相**：Qwen3.5-2B 的 linear-attention（`fused_recurrent_gated_delta_rule`）
   等 kernel 在 SM12 上 JIT 编译 spike + 未优化，首次推理 95s（含 JIT），warmup 后 28 tok/s。

## 3. 硬件遥测（resource_monitor 采集）

- GPU 利用率：`nvidia-smi utilization.gpu`（SM 活跃近似）；`nvidia-smi dmon -s u` 的
  `sm` 列（sm_active 代理）、`mem` 列（DRAM 带宽利用率代理）。
- DRAM 实测带宽：**315–323 GB/s（r+w）= 理论 448 GB/s 的 ~70–72%**（D2D copy 实测）。
- CPU / 系统内存：`/proc/stat` / `/proc/meminfo` 零依赖解析。
- NCU（精确 sm__throughput / dram__throughput）在本 WSL 容器不可用（`LibraryNotLoaded`），
  用 dmon 驱动指标代理，已标注。

## 4. vLLM 运行环境修复

6 个问题的完整根因与解决见 `docs/vllm-runtime-environment-fixes.md`（表格）。

## 5. 复现命令

```bash
# vLLM（vllm-build 环境，需先按 vllm-runtime-environment-fixes.md 设环境变量）
python benchmarks/e2e/benchmark_vllm.py --model /home/qyw/models/Qwen3.5-2B --max-tokens 64

# llama.cpp（py311 环境）
python benchmarks/e2e/benchmark_llamacpp.py --max-tokens 64

# 请求级并发对比（v9）
python benchmarks/e2e/compare_request_concurrency.py
```


## 2. moesim 在对比中的位置：专家粒度 vs llama.cpp 层粒度

llama.cpp 的异构是**层粒度**（`-ngl`：整层 GPU 或 CPU），moesim 是**专家粒度**
（每个专家独立 CPU/GPU 放置）。这是两者的本质差异：

- **层粒度**：前 L 层的全部专家 GPU，后 16−L 层全部 CPU——无法区分层内冷热专家。
- **专家粒度（moesim）**：按激活频率全局排序，热专家（跨层）进 GPU，冷专家 CPU。

**模拟对比**（`benchmarks/e2e/compare_heterogeneous_effect.py`，OLMoE 校准参数
GPU 0.076ms / CPU 0.639ms / PCIe 4.3GB/s，显存只够 16/64 专家）：

| 放置 | TPOT | 吞吐 | 说明 |
|---|---|---|---|
| 纯 CPU（全专家 CPU） | 2.556ms | 391 tok/s | 可行但慢 |
| 纯 GPU 上界（512MB） | 0.314ms | 3183 tok/s | 需全部权重进 GPU（放不下） |
| naive LRU（128MB） | 4.805ms | 208 tok/s | miss 就 PCIe load，被带宽拖死 |
| **cost_model 专家粒度（128MB）** | **0.530ms** | **1886 tok/s** | 热专家 GPU + 冷专家 CPU |

**诚实边界**：
1. moesim 模拟的是**专家层 only**（MoE FFN），llama.cpp 是**完整模型**（含 attention /
   embedding），绝对 TPOT 不可直接对比（`verify_on_real_machine.md` 早已标注）。
2. moesim 的 4.8x（vs 纯 CPU）本质是"用了 GPU"；真正体现**调度价值**的是
   **LRU 4.805ms → cost_model 0.530ms = 9.1x**——两者都用 GPU，差别仅在
   "冷专家走 CPU（0.639ms）而非 PCIe load（1.86ms）"这一 CPU 算力感知决策。
3. 模拟参数中 GPU 0.076 / CPU 0.639 / PCIe 4.3 是**实测**（v1/v3 微基准）；
   concurrency（GPU 8 路 / CPU 2 路）是**假设**，非实测。

**结论**：真实框架三档（llama.cpp）给出可信的显存-性能曲线；moesim 的贡献是
**专家粒度调度**（比层粒度更细的放置自由度）+ **确定性模拟**（策略开发/复现环境），
而非宣称绝对时延超越某框架。


## 3. 真实框架接入现状与路径（2026-08-29 调研）

### 3.1 真相：moesim 调度器目前未接入任何生产框架

`executor/backends/vllm.py` 和 `llama_cpp.py` 都是**记账式 wrapper**——`load/unload`
只改 `residency` dict，`execute_gpu` 转发 `engine.generate()` / `llama.eval()`，
**不调用 moesim 的 `decide()`，也不控制框架内部的专家放置**。因此：

- "llama.cpp + moesim 调度 vs llama.cpp baseline 是否有提升" —— **当前无法测，无数据**
- "vllm vs vllm + moesim 异构调度" —— **同样无法测**

唯一真接入是 transformers 的 `MoEForwardHook`，v3 实测**比 HF 原生慢**（29.47ms vs
12.53ms GPU，因每层 `decide()` 调度开销）。所以 moesim 目前的真实价值是**确定性
模拟器（策略开发/复现）+ 异构执行正确性验证**，而非"调度比框架 baseline 快"。

### 3.2 接入路径调研结论（关键发现）

**vLLM 有现成的、粒度可用的 offload 接口，是 moesim 接入的最短路径，且无需改
vLLM 代码**（`vllm/config/offload.py`）：

| vLLM 接口 | 粒度 | 对应 moesim 能力 |
|---|---|---|
| `uva.cpu_offload_params`（参数名段集合） | 按参数名段（如 `experts` / `experts.3`）指定 offload 哪些权重 | **专家放置的直接映射点**（decide 输出 → 参数段集合） |
| `prefetch.offload_params` + `offload_group_size` + `offload_prefetch_step` | 层组 + 异步 H2D 预取 | **prefetch 重叠**（v6 的传输-计算重叠） |
| `SimpleCPUOffloadScheduler`（`v1/simple_kv_offload/`） | KV block 级 LRU / lazy / async load | **KV 分层调度**（v8 的 GPU/主机 KV 分层 + 压力驱逐） |

**本质限制**：vLLM offload 是**加载时静态**配置（非运行时逐专家动态）。moesim 的
动态 `decide()` 不能直接映射，但 moesim 模拟器可以算出**最优静态放置计划**，映射成
vLLM 配置参数——这仍优于 vLLM 默认的"非选择性 offload 直到 `cpu_offload_gb` 满"。

**llama.cpp 接入更困难**：`--n-cpu-moe`（专家粒度 offload）底层走 `override_tensor`
机制，但 llama-cpp-python 0.3.35 未暴露（仅 `n_gpu_layers`）。需改 C++ 或绑定。

### 3.3 建议的第一个落地接入

1. moesim 模拟器用 OLMoE 校准参数（GPU 0.076 / CPU 0.639 / PCIe 4.3GB/s），在 8G
   显存约束下算出**最优静态专家放置计划**（哪些专家留 GPU、哪些 offload CPU）。
2. 映射成 vLLM `cpu_offload_params`（专家参数名段）。
3. 对比：vLLM 默认 offload（非选择性）vs vLLM + moesim 放置（按激活频率选冷专家）。
4. 验证：同样显存预算下，moesim 放置的 TPOT/吞吐是否优于 vLLM 默认。

## 4. 2.1 本机验证（vLLM 选择性 offload 链路打通，2026-08-30）

**目标**：验证 moesim 的 offload 计划能通过 vLLM 真实生效（机制验证，小模型）。

**关键修正**：vLLM 的 `LLM()` API 暴露的 offload 参数是 **`offload_params` + `offload_group_size`
+ `offload_num_in_group`**（prefetch offload），而非 `cpu_offload_params`（那是 config 层内部
字段，LLM API 未暴露）。之前方案文档/脚本用错参数名，导致 offload 静默不生效。

**实测**（Qwen3.5-2B，8GiB 显存，`offload_params={"mlp"}` + group_size=24 + num_in_group=12）：

| 配置 | 显存 | TPOT | GPU util |
|---|---|---|---|
| full GPU 基线 | 7426 MiB | 48.23ms | 29.4% |
| offload 后 12 层 mlp | **6776 MiB** | **270.19ms** | 36.8% |

**结论**：
1. ✅ 选择性 offload 真实生效（显存 ↓650MiB，offload 层走慢路径 → TPOT 5.6x）
2. ✅ moesim → vLLM 链路打通（offload 计划 → `offload_params` → 生效）
3. ⚠️ vLLM prefetch offload 的层选择是**固定模式**（group N 层、每组 offload 后 M 层），
   不是"任意选层"；`offload_params` 选的是**参数段**（mlp / experts / attention），
   即"offload 哪些类型的参数"，粒度是"参数段 × 固定层组模式"。
4. OLMoE 14G 完整对比仍需 ≥32G 内存机器（本机 7.6G OOM）。

脚本：`benchmarks/e2e/benchmark_vllm_offload_selective.py`；
配置生成器：`scripts/moesim_vllm_config.py`。
