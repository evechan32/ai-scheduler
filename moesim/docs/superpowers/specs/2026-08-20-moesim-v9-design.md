# moesim v9 — 请求级并发模拟 + 时延分解（设计规范）

- 日期：2026-08-20
- 前置：v6-v8（资源队列、KV 分层、混合精度）
- 依据：DistServe（OSDI'24，五段时延分解）、Mooncake（排队感知调度）、
  Kairos（排队占 TTFT 77-98%）、Vidur（请求级模拟器参照）、FastServe（MLFQ 排队）
  ——综述 §C 已核实

## 1. 背景与动机

用户持续强调"排队影响"与"能够模拟"。现有 `MoESimulation` 是**单请求步模型**：
一个 decode 序列串行 feed，无请求概念、无并发、无 prefill、无请求级排队。
真实服务的排队（Kairos：P95 TTFT 的 77-98% 是排队）完全不可见。

v9 增加**请求级并发模拟**：多请求共享 GPU/CPU/PCIe，prefill 与 decode 分离，
FIFO 排队，DistServe 式五段时延分解（prefill 排队 / prefill 执行 / 传输 /
decode 排队 / decode 执行）。

## 2. 设计决策

| 维度 | 决策 |
|------|------|
| 请求模型 | `Request`：arrival_ms / prompt_tokens / output_tokens / req_id |
| prefill | 建模为 GPU 上的计算块（prompt_tokens × prefill_per_token_ms），经 `gpu.schedule` 排队 |
| decode | 复用 v8 步内核（专家激活 + KV 增长），但**多请求共享资源并发推进** |
| 并发调度 | 轮询（round-robin）：每轮每个活跃请求推进一个 decode 步，全局时钟 = max(完成) |
| 排队 | prefill 队列 FIFO：请求到达时 GPU 忙 → 排队（记录排队时间）；decode 阶段连续 batching（无阻塞） |
| 时延分解 | TTFT = prefill 排队 + prefill 执行；JCT = 完成 - 到达；逐阶段累计 |
| 兼容性 | 独立新模块 `sim/request_sim.py`，不动现有 `MoESimulation` |
| 确定性 | 请求 trace 确定；资源竞争确定；无随机 |

## 3. 接口定义

### 3.1 `sim/request_sim.py`

```python
@dataclass(frozen=True)
class Request:
    req_id: int
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int

@dataclass
class RequestStats:
    ttft_ms: float          # prefill 排队 + prefill 执行
    prefill_queuing_ms: float
    prefill_exec_ms: float
    tpot_avg_ms: float      # 平均每 decode token
    jct_ms: float           # 完成 - 到达
    kv_offload_mb: float

class RequestSimulation:
    def __init__(self, scheduler, profiles, gpu_capacity_mb, pcie, gpu, cpu,
                 kv_per_token_mb=0.0, kv_gpu_capacity_mb=0.0,
                 prefill_per_token_ms=0.5, expert_trace=None):
        ...
    def run(self, requests: list[Request]) -> RequestMetrics
    # 每请求返回 RequestStats；聚合：平均/尾 TTFT、TPOT、JCT、吞吐、排队占比
```

- 解码步专家集：`expert_trace`（可调用 req_id → [expert 列表]）或默认
  `profiles 前 k 个`循环（确定性）。
- prefill：`gpu.schedule(max(now, arrival), prompt_tokens × prefill_per_token_ms)`，
  排队时间 = schedule 起点 - arrival。
- decode 轮询：活跃请求集合（已过 prefill、未完成输出），每轮各 feed 一步，
  复用 v8 的 `_step` 内核（专家执行 + KV 增长 + prefetch 重叠）。
- 指标：TTFT/TPOT/JCT 的 mean/p95；prefill 排队占比；系统吞吐（tok/s）。

### 3.2 复用

- `ComputeResource`（并发槽）、`BandwidthResource`（PCIe 排队）、
  `ScheduleState`（KV/驻留）、`Metrics`、v8 的 KV 记账与 prefetch 重叠逻辑。
- `_step` 内核提取为可复用函数（每请求推进一步，共享全局资源）。

## 4. 测试策略

- prefill 排队：GPU 忙时到达的请求 TTFT 包含排队（数值断言）。
- 并发 decode：两个请求共享 GPU 并发槽 → TPOT 劣化（资源竞争可见）。
- 时延分解：TTFT = 排队 + 执行（断言各段非负且相加）。
- 确定性：同 trace 两次运行逐位相同。
- 回归：既有 127 测试不动。

## 5. 验证场景

`benchmarks/e2e/compare_request_concurrency.py`：固定请求集，对比
并发度 1 vs 4 vs 8（CPU/GPU 并发槽不同）→ TTFT/TPOT/JCT 表，展示排队影响
（Kairos 论点：排队是时延主成分）。

## 6. 不在 v9 范围

- 抢占/优先级排队（FastServe MLFQ）
- 磁盘层（FlexGen 三层存储）
- 真实 prefill 专家级模拟（当前为计算块近似）
