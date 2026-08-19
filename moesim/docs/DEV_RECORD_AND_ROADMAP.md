# moesim 开发记录与待实现清单

> 更新：2026-08-17 | 测试：82/82 通过 | 分支：feat/moesim-v5（源码）、docs/full-record（本仓库）
> 本文档汇总全部开发历史、当前状态、待实现项，作为项目权威记录。

---

## 第一部分：已完成的开发（v1-v5）

### 项目定位
**moesim** — 异构计算感知的 MoE 专家下放调度框架。核心立场：CPU 算力作为一等公民
参与调度决策（不仅是内存兜底）。调度器为纯函数 `decide(state, clock) -> actions`，
同一策略代码同时驱动模拟器与真实推理引擎。

### v1 — 模拟器 + 调度器基础（41 测试）✅
- 领域无关 DES 核心（`sim/core.py`）：时钟推进、确定性保证
- 资源模型（`sim/resources.py`）：Bandwidth/Compute/Storage
- 校准（`sim/calibrate.py`）+ 指标（`sim/metrics.py`）+ 多策略对比（`sim/sweep.py`）
- 调度契约（`scheduler/base.py`）：6 种 Action + Scheduler 抽象
- 策略：lru / activation_freq / **cost_model**（CPU 算力入决策）
- 自研 FP16 CPU expert FFN kernel（C++ 扩展 + torch fallback）
- 实测：PCIe 4.30 GB/s；expert GPU 0.224ms / CPU 3.456ms

### v2 — 执行闭环补全（63 测试）✅
- 真实权重 offload（GPU↔CPU 参数搬移）
- KVWeightedPolicy（KV 压力感知）
- INT8（0.33% 误差）+ INT4（8.11%）量化 kernel
- vllm / llama.cpp 后端 + accelerate 集成
- 模拟器消费 evict_kv/fetch_kv

### v3 — 真实推理闭环（71 测试）✅
- **MoEForwardHook**：替换 HF MoE forward，router 专家经调度器分派 CPU/GPU
- 实测：与 HF 原始 forward 误差 0.062%（<1%）
- RL 调度策略（numpy-only Q-learning）
- MultiGPUCluster + per-GPU 驻留

### v4 — 性能增强（75 测试）✅
- **真并行执行**：线程池并发 CPU/GPU（混合 22.12ms）
- 决策缓存（单 forward 内 decide 复用）
- REINFORCE 策略梯度调度器
- INT4 gemm（torch._int_mm）
- 一键安装 install.sh + 中文 README

### v5 — 调度器增强（82 测试）✅【当前版本】
针对 4 个真实缺陷：GPU 排队感知、CPU 资源感知、迁移成本 + 驻留收益、驻留稳定性
- **ResidencyAwarePolicy** 实测：热专家场景比 cost_model 快 46.9%
  （0.221ms vs 0.416ms，吞吐 4525 vs 2404 tok/s）

### 真实推理性能实测
- **NF4 4-bit 量化进 GPU**：OLMoE-1B-7B（26G fp32 → ~4G），显存 6510MiB，362.8ms/forward
- llama.cpp（Q3_K_L CPU）：28 tok/s（短上下文）
- moesim hook 混合执行（小模型）：22.12ms/forward

---

## 第二部分：框架编译状态

### vLLM 源码编译（进行中）
- **目的**：RTX 5070（SM 12.0）无预编译 wheel，需源码编译
- **已解决**：8 个 GitHub 依赖绕过（SRC_DIR 本地源）、CUDA nvcc/runtime 版本错位
- **当前阻塞**：/tmp venv 丢失 → 需用 conda 重建干净环境（用户已同意，conda create 超时未确认）
- **完整指南**：`docs/vllm-build-guide.md`
- **venv 路径建议**：`/home/qyw/vllm-venv`（持久路径，勿放 /tmp）

### SGLang（调研完成，未开始）
- 要求：torch 2.11 + transformers 5.12.1 + flashinfer cu13（与主环境不兼容）
- 需独立 conda 环境；flashinfer cu13 的 SM12 兼容性待确认
- 注意：SGLang 与 vLLM 需**两个独立环境**（torch 版本不同）

---

## 第三部分：待实现清单

### 高优先级
| # | 事项 | 状态 |
|---|---|---|
| 1 | 用 conda 重建 vLLM 编译环境并完成编译 | 🔄 进行中 |
| 2 | vLLM 编译成功后测 OLMoE 推理 → 记录框架对比 | ⏳ |
| 3 | v5 代码同步推送 ai-scheduler → PR #4 | ⏳ |
| 4 | SGLang 环境搭建 + SM12 flashinfer 兼容性验证 | ⏳ |

### 中优先级（moesim 增强）
| # | 事项 | 说明 |
|---|---|---|
| 5 | 真实 vLLM kernel 级集成 | 当前后端为记账式 |
| 6 | 生产级 RL 训练循环 | 当前为模拟器内 Q-learning |
| 7 | 多 GPU 真机验证 | 当前仅模拟层 |
| 8 | INT4 kernel 性能优化 | 当前为精度优先反量化路径 |
| 9 | 调度决策开销优化 | 缩小与 HF 原生差距（决策缓存已部分解决） |

### 低优先级 / 远期
| # | 事项 | 说明 |
|---|---|---|
| 10 | KV+专家联合调度的生产实现 | 模拟器已支持，真机未验证 |
| 11 | 激活稀疏优化 | 论文综述中 Prox/SparseInfer 方向 |
| 12 | 投机解码集成 | Dovetail CPU/GPU 异构投机方向 |
| 13 | TriMoE 式三路异构（GPU+CPU+NDP） | 论文综述方向 |

---

## 第四部分：关键文档索引

| 文档 | 路径 |
|---|---|
| 项目总文档 | `docs/PROJECT_SUMMARY.md` |
| 会话历史 | `docs/SESSION_HISTORY.md` |
| vLLM 编译指南 | `docs/vllm-build-guide.md` |
| 论文综述（50+ 篇） | `docs/research/moe-inference-optimization-survey.md` |
| 主设计规范 | `docs/superpowers/specs/2026-08-09-moesim-design.md` |
| v1-v5 实施计划 | `docs/superpowers/plans/` |
| 真机验证协议 | `benchmarks/e2e/verify_on_real_machine.md` |

## 第五部分：GitHub 交付

| PR | 内容 | 状态 |
|---|---|---|
| PR #1 | v1+v2 | ✅ 已合并 |
| PR #2 | v3 | ✅ 已合并 |
| PR #3 | v4 | ✅ 已创建 |
| PR #4 | v5 + 文档 | ⏳ 待创建 |

仓库：https://github.com/evechan32/ai-scheduler
