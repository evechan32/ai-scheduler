# moesim 项目愿景与路线图（两步走）

> 2026-08-30 | 本文档固化项目的战略方向与分阶段目标。

## 两步走

### 第一步：模拟策略可行性 ✅ 已完成（v1–v9）

领域无关离散事件模拟器，验证异构 MoE 调度策略的可行性：
- CPU 算力感知放置（cost_model）、EFT + 预取重叠（overlap）、驻留价值（residency）
- KV 分层调度（KVJointPolicy）、请求级并发 + 时延分解、混合精度放置
- 确定性保证 + 单测 + 基准 + 论文依据（MoE-Infinity / HEFT / Mooncake / FlexGen / HOBBIT）

### 第二步：异构算力 + 异构存储的统一调度器，适配框架 🎯 进行中

**目标**：一个调度器同时利用**异构算力**（CPU + GPU）和**异构存储**（显存 + 主机内存
+ 磁盘），并真正适配开源框架（vLLM / llama.cpp），做运行时调度。

```
                     ┌─────────────────────────────────────────┐
  异构算力           │  CPU (通用核)   +   GPU (张量核)           │  ← 专家/算子的计算分派
                     └─────────────────────────────────────────┘
                     ┌─────────────────────────────────────────┐
  异构存储           │  显存 (VRAM)  →  主机内存 (DRAM)  →  磁盘 (SSD) │  ← 权重/KV 三层放置
                     │  热            →  温               →  冷          │
                     └─────────────────────────────────────────┘
  统一调度器         │  decide(算力占用, 存储水位, 激活频率) → 联合放置动作 │
  适配框架           │  vLLM (cpu_offload_params / KV offload) / llama.cpp (--n-cpu-moe) │
```

**核心调度对象**（三者联合，而非分开）：
1. **专家权重**：热专家显存 / 冷专家内存 / 最冷磁盘（FlexGen 三层）
2. **KV cache**：GPU KV 池 / 主机 KV 池 / 磁盘 KV（Mooncake / LMCache 分层）
3. **计算分派**：每个专家 CPU 还是 GPU 执行（Fiddler / KTransformers）

## 第二步的参考体系

| 维度 | 顶会论文 | 高星项目 |
|---|---|---|
| 三层存储 + offloading 策略搜索 | FlexGen（OSDI'23） | DeepSpeed-Inference |
| 专家预取流水线（SSD→DRAM→GPU） | MoE-Infinity | MoE-Infinity |
| KV 分层（GPU/CPU/SSD/RDMA） | Mooncake（FAST'25） | LMCache / SGLang HiCache |
| CPU 算力入决策 | Fiddler（ICLR'25） | ktransformers（SOSP'25） |
| 异步重叠 / 预取 | APEX（IPDPS'26）/ Pre-gated MoE | llama.cpp --n-cpu-moe |

## 第二步的分阶段路线图

| 阶段 | 内容 | 状态 |
|---|---|---|
| 2.0 | 模拟器补全异构存储：加**磁盘层**（FlexGen 三层），专家权重 + KV 三级放置 | ⏳ 未做 |
| 2.1 | **静态放置适配 vLLM**：moesim 算冷层 MoE + KV 分层 → `cpu_offload_params` + KV offload（方法 1+2，不改模型） | ✅ 配置生成器已做，待大内存机验证 |
| 2.2 | **静态放置适配 llama.cpp**：`--n-cpu-moe` / `-ngl` 配置对齐 | ⏳ |
| 2.3 | **运行时动态调度接入**：改框架暴露逐专家运行时钩子，`decide()` 每步驱动放置（阶段 3） | ⏳ 未做 |
| 2.4 | 三层存储 + 联合调度在模拟器 + 框架双端验证 | ⏳ 未做 |

## 关键认知（诚实）

- 模拟器（第一步）的价值是**策略可行性验证 + 相对排序**，不是绝对性能预测。
- 适配框架（第二步）的本质限制：现有框架的 offload 都是**加载时静态**或**层粒度**；
  真正逐专家运行时调度需要改造框架执行循环（阶段 2.3，大工程）。
- 异构存储的"磁盘层"目前任何主流框架都未完整支持（MoE-Infinity 是 SSD→DRAM→GPU
  多级，但非通用），这也是 moesim 的差异化机会。
