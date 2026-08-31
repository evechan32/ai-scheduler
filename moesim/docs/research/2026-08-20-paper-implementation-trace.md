# 论文-实现追踪（异构算力机制 → moesim 代码落点）

> 更新：2026-08-20 | 本文档回答一个问题：**当前开发参考了哪些顶会论文/高星项目，
> 每个机制在代码里具体落在哪里，参考到什么程度。**
> 姊妹篇：`docs/research/2026-08-20-queue-overlap-heterogeneous-survey.md`（调研证据）。

## 说明

moesim 对论文的参考是**机制级设计参考**，不是代码移植——把论文的核心思想提炼为
模拟器/调度器里的简化模型。下表逐条给出"论文机制 → 代码落点 → 参考程度 → 简化点"，
可在代码中逐行核对。

参考程度定义：
- **公式级**：决策公式直接取自论文（可逐行对应）。
- **机制级**：核心机制落地，但省略论文的配套工程。
- **思想级**：借鉴设计思想，实现形态不同。
- **反例证据**：用工程实证校准我们的设计立场（不直接实现）。

---

## 一、v6（2026-08-20，本次开发）

| 论文/项目 | 机制 | 代码落点 | 程度 | 简化/未做 |
|---|---|---|---|---|
| **HEFT**（IEEE TPDS'02） | 最早完成时间（EFT）调度：`EFT = 就绪等待 + 通信/执行`，选 EFT 最小的处理器 | `scheduler/policies/overlap.py`：`_cpu_eft()` / `_gpu_eft()`；`decide()` 中 `cpu_eft < gpu_eft → execute_cpu` | **公式级** | HEFT 的 DAG 任务图、向上秩排序、插入式调度未做（moesim 是逐专家贪心 EFT） |
| **MoE-Infinity**（arXiv:2401.14361） | 激活感知预取：预取与执行重叠、PCIe 串行化避免争用、"预取失败比不预取更糟" | `overlap.py::_prefetch()`：按 activation_freq 取 top-N 非驻留专家 emit `prefetch`；带宽门控 `pcie_queue_len <= max_pcie_queue` 且 `pcie_utilization <= util_prefetch_threshold`；`sim/moe_adapter.py::_book_transfer()`：预取不进关键路径（`critical=False`） | **机制级** | EAM/EAMC 序列级激活追踪、层邻近度优先级、SSD→DRAM→GPU 多级流水未做 |
| **KTransformers**（SOSP'25） | 异步 CPU-GPU 任务调度 + Expert Deferral（CPU 填充 GPU 空闲、CPU 利用率 <75%→~100%） | `overlap.py`：CPU/GPU 双路径并行执行，EFT 决定分派；`sim/moe_adapter.py::_step()`：GPU 与 CPU 计算资源并行 `schedule`，步完成 = max(两者) | 思想级 | 无锁任务队列、CUDA graph 封装、AMX kernel 是真实推理系统工程，moesim 只模拟并行语义 |
| **Kairos / Mooncake**（arXiv:2607.02043 / 2407.00079） | 排队感知调度：等待时间 = 聚合排队作业；队列是时延主成分 | `sim/resources.py`：`queue_depth()` / `utilization()` / `wait_time_ms()`（peek）；`sim/moe_adapter.py::_snapshot_feedback()`：每步把 pcie/gpu/cpu 队列深度、利用率、等待估计写入 `ScheduleState`；`overlap.py` 消费 `state.*_wait_ms` | **机制级** | 生产级调度器的 SLO 拒绝、请求级时延分解（DistServe 五段）未做；moesim 为专家级下放场景 |
| **APEX**（IPDPS'26） | profiling-informed 分派：预测 CPU/GPU 子任务执行时间，动态选择重叠执行 | `overlap.py`：以 `profile.cpu_exec_ms`/`gpu_exec_ms` + 队列等待做 EFT 预测分派 | 思想级 | APEX 的延迟同步、统一 batch 分支未做（moesim 无真实 kernel） |
| **PowerInfer-2**（MobiCom'24） | I/O-计算流水线（传输与计算重叠） | `moe_adapter.py`：prefetch 传输与当前步计算重叠；`pending_loads` 跨步追踪在途传输，执行起点 = max(clock, 在途完成) | 机制级 | neuron-cluster 粒度分解未做（moesim 粒度 = 专家） |
| **llama.cpp** `--n-cpu-moe`（工程） | CPU 卸载 V 曲线：只卸到"刚好放得下"，过度卸载吞吐单调下降 | `overlap.py`：CPU 争用计入决策（`cpu_exec × (1 + cpu_queue_len/concurrency)`），CPU 饱和时回退 GPU | 反例证据 | 校准我们的"CPU 不是免费资源"立场，未实现其 tensor override |

---

## 二、v1-v5（历史实现）

| 论文/项目 | 机制 | 代码落点 | 程度 |
|---|---|---|---|
| **Fiddler**（ICLR'25） | CPU-GPU 编排：CPU 算力入决策 | `scheduler/policies/cost_model.py`：`cpu_exec_ms <= load_cost + gpu_exec_ms → execute_cpu`（v1 起） | 公式级 |
| **Pre-gated MoE**（ISCA'24） | 推测路由提前预取 | `cost_model.py::prefetch_n`、`activation_freq.py`（激活频率预取） | 思想级 |
| **SwapMoE**（ACL'23） | 可调内存预算专家交换 | `cost_model.py` 容量约束 + LRU 驱逐；`lru.py` | 机制级 |
| **PowerInfer**（SOSP'24） | hot/cold 神经元划分 | `activation_freq.py`（频率划分 hot/cold 预取） | 思想级 |
| **MoE-Infinity / KTransformers**（工程路径） | 替换 HF MoE forward 插入逐专家分派 | `executor/backends/forward_hook.py`（`MoEForwardHook`，v3） | 机制级（真实代码路径参考） |
| **HOBBIT**（arXiv:2411.01433） | 混合精度下放 | `executor/cpu_kernels/quantized.py`（INT8/INT4） | 思想级 |
| **KTransformers / ktransformers 项目** | 混合执行数值验证路径 | v3 实测：hook vs 原始 forward 误差 0.062% | 工程参照 |
| **mixtral-offloading**（arXiv:2312.17238） | LRU 专家缓存 + 推测预取 | `lru.py` + `cost_model.py` prefetch | 思想级 |

---

## 三、明确未参考（及原因）

| 论文 | 机制 | 未参考原因 |
|---|---|---|
| **DistServe**（OSDI'24） | prefill/decode 分池、五段时延分解 | 集群级服务架构；moesim 场景是单机专家级下放。其**排队阶段分解思路**已列入路线图（请求级并发模拟） |
| **TetriInfer**（OSDI'24） | 固定 chunk、长度预测调度 | 同上，集群级；专家级下放无此问题 |
| **Splitwise / MemServe** | PD 分离、KV 记忆管理 | 场景不匹配（无跨机 KV 传输） |
| **DeepSpeed MoE / FasterMoE** | 专家并行通信重叠 | 多卡训练/推理场景，moesim 单卡+CPU 下放 |

---

## 四、验证方式

- 单测：`tests/scheduler/test_overlap_policy.py`（EFT/门控/驻留）、
  `tests/sim/test_prefetch_overlap.py`（重叠/在途复用/确定性）、
  `tests/sim/test_resources_v6.py`（队列可见性）。
- 基准：`benchmarks/e2e/compare_queue_overlap.py`（热专家 trace，
  overlap(pf=2) TPOT 2.368ms，比全 CPU 放置快 50.7%）。
- 确定性：同输入两次运行逐位相同（`test_deterministic_across_runs`）。
