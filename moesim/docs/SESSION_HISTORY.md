# moesim 开发会话历史记录

> 记录时间：2026-08-17
> 项目：moesim — 异构计算感知的 MoE 专家下放调度框架
> 当前分支：feat/moesim-v5 | 测试：82/82 通过

---

## 1. 会话目标与起源

用户最初的诉求：**能否在 12G 显存 + 16G 内存的机器上运行 Qwen3-30B-A3B（MoE 模型）**。
经深入调研与头脑风暴，最终演进为构建 **moesim**——一个完整的异构计算感知调度框架，
核心立场是 **CPU 算力作为一等公民参与调度决策**（不仅是内存兜底），调度器为纯函数
`decide(state, clock) -> actions`，同一策略代码同时驱动模拟器与真实推理引擎。

## 2. 五个版本迭代记录

### v1 — 模拟器 + 调度器基础（41 测试）
- 领域无关 DES 核心（`sim/core.py`）：时钟推进、确定性
- 资源模型（`sim/resources.py`）：BandwidthResource / ComputeResource / StorageResource
- 校准（`sim/calibrate.py`）+ 指标聚合（`sim/metrics.py`）+ 多策略对比（`sim/sweep.py`）
- 调度器契约（`scheduler/base.py`）：Action 六种类型 + Scheduler 抽象
- 三种策略：lru / activation_freq / **cost_model**（`cpu_exec_ms <= load_cost + gpu_exec_ms` → CPU）
- 自研 FP16 CPU expert FFN kernel（C++ 扩展 + torch fallback）
- **真实测量**：PCIe 4.30 GB/s；expert GPU 0.224ms / CPU 3.456ms

### v2 — 执行闭环补全（63 测试）
- **真实权重 offload**：transformers executor load/unload 真实搬移专家参数 GPU↔CPU
- **KV 感知策略**：KVWeightedPolicy（GPU KV 压力 >0.9 → 强制 CPU + KV 驱逐）
- **量化 kernel**：INT8（误差 0.33%）+ INT4（2-per-byte 打包，8.11%）
- vllm / llama.cpp 后端（optional import）
- 模拟器消费 evict_kv/fetch_kv（KV 转移计入 PCIe 计时）
- accelerate 可选集成

### v3 — 真实推理闭环 + RL + 多 GPU（71 测试）
- **MoEForwardHook**：替换 HF MoE 层 forward，router top-k 专家经调度器分派 CPU/GPU
- **实测**：hook vs 原始 forward 相对误差 0.062%（<1% 目标）
- RL 调度策略（numpy-only Q-learning）
- MultiGPUCluster（节点容量 + 带宽矩阵）+ per-GPU 驻留

### v4 — 性能增强（75 测试）
- **真并行执行**：线程池并发 CPU/GPU 专家（混合 23.89→22.12ms，全 GPU 34.84→21.85ms）
- 决策缓存（单 forward 内 decide 复用）
- **REINFORCE 策略梯度调度器**（生产级 RL）
- INT4 gemm（torch._int_mm 真整数路径）
- 一键安装 install.sh + 中文 README + 论文综述 50+ 篇

### v5 — 调度器增强（82 测试）【当前版本】
针对用户指出的 4 个真实缺陷：
1. **GPU 排队感知**：`gpu_contention = queue_len / concurrency`，严重排队时强制 CPU
2. **CPU 资源感知**：`cpu_contention` 计入决策
3. **专家迁移成本 + 驻留收益**：`record_load` 累计 PCIe 成本，收益超 load_cost 则长期驻留
4. **驻留稳定性**：驻留专家默认留 GPU（不轻易驱逐）

**ResidencyAwarePolicy 实测**（热专家 80% + 冷 20% trace）：
| 策略 | TPOT | 吞吐 |
|---|---|---|
| cost_model | 0.416ms | 2404 tok/s |
| **residency** | **0.221ms** | **4525 tok/s** |

**ResidencyAwarePolicy 比 cost_model 快 46.9%**（热专家稳定驻留，避免反复迁移）。

## 3. 真实推理性能实测

### 模型量化进 GPU（关键突破）
- OLMoE-1B-7B safetensors 是 fp32 26G，无法直接进 8G 显存
- **bitsandbytes NF4 4-bit 量化**：`load_in_4bit=True + device_map='auto'`
- 权重压缩至 ~4G，**成功进 GPU**（显存 6510 MiB，占 8G 的 80%）
- 真实模型 GPU 推理：362.8ms/forward

### 框架对比
| 框架 | 方式 | 性能 |
|---|---|---|
| llama.cpp | Q3_K_L GGUF CPU | 28 tok/s（短上下文） |
| transformers | NF4 4-bit GPU | 362.8ms/forward |
| moesim hook | 混合执行（小模型） | 22.12ms/forward |
| vLLM | ❌ SM12 需源码编译 | 进行中 |

## 4. vLLM 源码编译（进行中）

**背景**：vLLM 预编译 wheel 不含 RTX 5070（SM 12.0 消费级 Blackwell）kernel，
需从源码编译（vLLM PR #38412 确认）。

**方案**：
- 独立 venv `/tmp/opencode/vllm-venv`（Python 3.13 + torch 2.13 + vLLM 0.27.2rc1）
- vLLM 源码 `/home/qyw/projects/vllm`（main 分支）
- `TORCH_CUDA_ARCH_LIST="12.0"` + `MAX_JOBS=2`（防 7.6G 内存 OOM）

**依赖绕过**（8 个 GitHub 依赖，网络受限全用 SRC_DIR 本地源）：
| 依赖 | 本地源 |
|---|---|
| CUTLASS | .deps/cutlass-src |
| Triton | /home/qyw/vllm-deps/triton（+手工 CMakeLists 绕过 MLIR 检查） |
| FlashAttention | .deps/flash-attention-src |
| DeepGEMM | .deps/deepgemm-src |
| Qutlass | /home/qyw/vllm-deps/qutlass |
| FlashMLA | /home/qyw/vllm-deps/FlashMLA |
| MSA(fmha_sm100) | /home/qyw/vllm-deps/MSA |
| FlashKDA | /home/qyw/vllm-deps/FlashKDA |
| tml-fa4 | /home/qyw/vllm-deps/tml-fa4 |

**当前状态**：已绕过全部 GitHub 克隆，编译进入 C++/CUDA 阶段。
**最近错误**：`CUDA compiler and CUDA toolkit headers are incompatible`（nvcc 13.3 与
venv 内 cccl 头文件版本不匹配，`cuda_view.cu.o` 编译失败）——待解决：统一 CUDA
toolkit 版本（nvcc 与 cccl 需同源）。

## 5. GitHub 交付

| PR | 内容 | 状态 |
|---|---|---|
| PR #1 | v1+v2（511c7da） | ✅ 已合并 |
| PR #2 | v3（3705176） | ✅ 已合并 |
| PR #3 | v4 | ✅ 已创建 |
| v5 | 待推送 | 未推送（feat/moesim-v5 分支） |

仓库：https://github.com/evechan32/ai-scheduler（evechan32/ai-scheduler）

## 6. 关键文档索引

| 文档 | 内容 |
|---|---|
| `docs/PROJECT_SUMMARY.md` | 项目总文档（中文，含全部版本与实测） |
| `docs/research/moe-inference-optimization-survey.md` | 论文综述 50+ 篇（不重训练优化） |
| `docs/superpowers/specs/2026-08-09-moesim-design.md` | 主设计规范 |
| `docs/superpowers/plans/` | v1-v5 实施计划 |
| `benchmarks/e2e/verify_on_real_machine.md` | 真机验证协议 |
| `README.md` / `README.zh-CN.md` | 中英文 README |

## 7. 环境要点

- 主环境：`/home/qyw/miniconda3/envs/py311`（torch 2.9，82 测试未受影响）
- vLLM venv：`/tmp/opencode/vllm-venv`（torch 2.13 + vLLM 0.27.2rc1）
- 模型：`/home/qyw/models/olmoe-1b-7b`（safetensors 26G）+ `olmoe-1b-7b-gguf`（Q3_K_L）
- 编译脚本：`/tmp/opencode/vllm_build.sh`；日志：`/tmp/opencode/vllm_build.log`
- GitHub token：fine-grained（gh 已登录），可创建 PR 不可合并（缺 Contents write）

## 8. 下一步计划

1. 解决 vLLM CUDA 头文件不兼容（统一 nvcc/cccl 版本）→ 完成编译
2. 编译成功 → 测 vLLM 推理 OLMoE → 记录框架对比到文档
3. v5 同步推送到 ai-scheduler → PR #4
4. 剩余路线图：真实 vLLM kernel 级集成、生产级 RL、多 GPU 真机、INT4 优化

## 9. 过程中修复的关键缺陷（记录）

- load_inline 重复 PYBIND11_MODULE 导致 C++ 编译静默失败
- torch_extensions 残留 lock 文件导致加载挂起
- hook 不支持真实 OLMoE 3D 输入 + per-token top-k + tuple 契约
- execute_cpu 不支持真实 OlmoeMLP（gate_proj/up_proj/down_proj）
- 跨层 residency 状态污染（混合执行特有 bug）
- uninstall 恢复 unbound forward 导致崩溃
- KV pressure 计算与测试矛盾
- v5 中 queue_len 反馈与驻留策略的振荡问题
