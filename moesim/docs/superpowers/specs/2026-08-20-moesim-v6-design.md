# moesim v6 — 排队与重叠感知调度（设计规范）

- 日期：2026-08-20
- 状态：设计批准（本会话头脑风暴结论）
- 前置：v5（ResidencyAwarePolicy + 队列长度反馈）已合入本分支
- 依据：`docs/research/2026-08-20-queue-overlap-heterogeneous-survey.md`

## 1. 背景与动机（用户需求）

用户要求继续完善 moesim，明确四个方向：

1. **考虑排队影响** —— PCIe 传输队列、GPU/CPU 计算队列的深度与等待必须成为调度决策
   的输入；队列对 TPOT/吞吐的影响必须可测量。
2. **考虑 CPU 资源影响** —— CPU 不是无成本资源：CPU 队列深度与利用率要反馈进决策，
   CPU 饱和时不能继续把专家压给 CPU（llama.cpp V 曲线：过度卸载吞吐单调下降）。
3. **实现通算并行** —— CPU 与 GPU 同时执行不同专家（v4 真并行已落地执行层；v6 让
   模拟器与策略**排队感知地**选择并行分派）。
4. **实现一定的重叠性** —— 预取（prefetch）传输与当前步计算重叠，PCIe 传输不进
   关键路径（MoE-Infinity 激活感知预取、vLLM async scheduling 同思路）。

## 2. 现状缺陷（v5 遗留）

| # | 缺陷 | 位置 |
|---|------|------|
| 1 | 资源层不暴露队列：`BandwidthResource` 只有隐式 `_busy_until` 串行化；`ComputeResource` 只有槽位 `schedule`，无队列深度/利用率/等待时间查询 | `sim/resources.py` |
| 2 | 调度器拿不到 PCIe 队列与资源利用率，只能拿到 v5 的整数 `gpu_queue_len`/`cpu_queue_len`；**没有等待时间估计**（EFT 决策所需） | `scheduler/state.py` |
| 3 | 无重叠机制：`load` 只能在步内发生，预取下一批专家必须等当前步结束 | `sim/moe_adapter.py` |
| 4 | 指标不含队列/利用率/重叠统计 | `sim/metrics.py` |

## 3. 设计决策摘要

| 维度 | 决策 |
|------|------|
| 队列模型 | 资源层记录每次 reservation 的 (start, completion)，对外暴露 `queue_depth(now)` / `utilization(until)` / `wait_time_ms(now, units)`（**peek，不改变状态**） |
| 决策模型 | **EFT（最早完成时间，HEFT 范式）**：`cpu_EFT = cpu_wait + cpu_exec×(1+contention)` vs `gpu_EFT = pcie_wait + transfer + gpu_exec×(1+contention)`，取小者；排队与 CPU 影响都计入 |
| 重叠模型 | 新增 `prefetch` Action：预取传输**立即开始**、与当前步计算重叠、完成时间不进步关键路径；通过 `pending_loads` 跨步追踪在途传输 |
| 预取门控 | **带宽预算**：PCIe 队列深度超过阈值或利用率过高时禁止预取（MoE-Infinity 谨慎原则：过度预取挤占带宽反而更差） |
| 反馈回路 | 每步执行前，adapter 把资源队列深度/利用率/等待时间快照写入 `ScheduleState`，`decide()` 消费 |
| 兼容性 | 全部新增字段与方法默认值向后兼容；既有 82 测试语义不变（新增字段不改既有路径） |
| 确定性 | 无随机；两次运行逐位相同 |

## 4. 架构改动

```
sim/resources.py          sim/metrics.py
  BandwidthResource         + queue depth / utilization / wait
  + queue_depth()           + hidden transfer / overlap ratio
  + utilization()           + pcie/gpu/cpu utilization
  + wait_time_ms() peek
         │ 反馈快照
         ▼
scheduler/state.py  ──►  ScheduleState
  + pcie_queue_len / *_utilization / *_wait_ms
  + pending_loads (在途传输完成时间表)
         │ decide(state, clock)
         ▼
scheduler/policies/overlap.py  (OverlapAwarePolicy)
  EFT 放置（排队+CPU 影响） + prefetch（重叠，带宽门控）
         │ actions（load/unload/execute_gpu/execute_cpu/prefetch）
         ▼
sim/moe_adapter.py
  prefetch → 后台 PCIe 传输（重叠）→ pending_loads
  load     → 若已在途则复用 pending 完成时间（不重复占带宽）
  执行     → start = max(clock, pending_loads.get(eid))
```

## 5. 接口定义

### 5.1 `sim/resources.py`

- `BandwidthResource.reserve(now, size_mb) -> float`：语义不变；内部记录
  `(start, completion)` 到 `_reservations`，并累计 `_wait_ms += max(0, start - now)`。
- `BandwidthResource.queue_depth(now) -> int`：未完成 reservation 数（在途 + 排队）。
- `BandwidthResource.utilization(until) -> float`：忙时 / 窗口（串行通道 = busy_time/window）。
- `BandwidthResource.wait_time_ms(now, size_mb) -> float`：peek——
  `max(now, _busy_until) + transfer_time_ms(size) - now`（不改状态）。
- `ComputeResource.schedule(now, units) -> float`：语义不变；记录 `(slot, start, completion)`。
- `ComputeResource.queue_depth(now) -> int`：未完成 reservation 数。
- `ComputeResource.utilization(until) -> float`：Σ槽忙时 / (窗口×concurrency)。
- `ComputeResource.wait_time_ms(now, units) -> float`：peek——
  最早空闲槽位上的完成时间减 now（不改状态）。

### 5.2 `sim/metrics.py`（新增字段，默认 0）

- `pcie_queue_depth_avg/max`、`gpu_queue_depth_avg/max`、`cpu_queue_depth_avg/max`
- `pcie_utilization`、`gpu_utilization`、`cpu_utilization`
- `transfer_wait_ms`（PCIe 排队等待总量）、`hidden_transfer_ms`（被计算隐藏的传输量）、
  `prefetch_count`、`overlap_ratio = hidden / total_transfer`
- 新方法 `record_step(..., queue_stats, utilizations, ...)`（或逐字段累加方法）

### 5.3 `scheduler/state.py`

- 新增：`pcie_queue_len: int = 0`、`pcie_utilization: float = 0.0`、
  `gpu_utilization: float = 0.0`、`cpu_utilization: float = 0.0`、
  `gpu_wait_ms: float = 0.0`、`cpu_wait_ms: float = 0.0`、`pcie_wait_ms: float = 0.0`、
  `pending_loads: dict[str, float] = field(default_factory=dict)`（expert → 传输完成时间）
- v5 字段（gpu_queue_len / cpu_queue_len / residency_benefit / migration_cost_ms）保留。

### 5.4 `scheduler/base.py`

- `_VALID_KINDS` 增加 `"prefetch"`。
- `apply_actions`：`prefetch` 与 `load` 同语义（检查容量、加入 resident、更新 used_gpu_mb）。
- `Action(kind="prefetch", expert_ids=...)`。

### 5.5 `scheduler/policies/overlap.py` — `OverlapAwarePolicy`

```
__init__(pcie, gpu_concurrency=1, cpu_concurrency=4,
         prefetch_n=2, max_pcie_queue=2, util_prefetch_threshold=0.7)

decide(state, clock):
  对每个 requested 专家 e：
    若 e resident：
      GPU 严重排队(>1.5×) 且 CPU 更便宜 → execute_cpu；否则 execute_gpu（稳定性，v5）
    否则（非驻留）：
      cpu_EFT = cpu_wait_ms + cpu_exec × (1 + cpu_queue_len/gpu...cpu_concurrency)
      gpu_EFT = pcie_wait_ms + transfer_time + gpu_exec × (1 + gpu_queue_len/gpu_concurrency)
      resident 收益足够（v5 residency_benefit）→ load
      cpu_EFT < gpu_EFT → execute_cpu，否则 load
  预取（重叠）：
    若 pcie_queue_len <= max_pcie_queue 且 pcie_utilization <= 阈值：
      取 top-prefetch_n 个 非驻留、非 requested、activation_freq 最高的专家，
      容量允许则 emit prefetch（可含 resident 收益高的候选）
```

### 5.6 `sim/moe_adapter.py`

`_step()` 流程（每步）：
1. 剪枝 `pending_loads`（完成时间 <= clock 的条目移除）。
2. **反馈快照**：从资源查 queue_depth / utilization / wait_time_ms，写入 state
   （gpu_queue_len、cpu_queue_len、pcie_queue_len、*_utilization、*_wait_ms）。
3. `decide(state, clock)` → actions；`apply_actions(state, actions)`。
4. **load**：对每个 load 专家——若 `eid in pending_loads` 则复用其完成时间（不重复
   占 PCIe）；否则 `pcie.reserve(clock, size)`，记录 pending_loads + load_times。
5. **prefetch**：`pcie.reserve(clock, size)` → completion；`pending_loads[eid] = completion`；
   **不计入步关键路径**；累计 prefetch_count / hidden_transfer。
6. **执行**：GPU 执行 start = `max(clock, pending_loads.get(eid, clock))`（在途传输完成
   后才能执行）；CPU 执行 start = clock（权重已在主机内存）。
7. **KV**：evict/fetch 仍走 PCIe 计时而计入关键路径（不变）。
8. 步完成 = max(GPU/CPU 执行完成 + KV 完成)；**prefetch 完成不参与**。
9. 指标：记录队列深度均值/最大值、利用率、transfer_wait（load+prefetch+KV 的
   `max(0, start-clock)` 累计）、hidden_transfer（prefetch 的传输时间）。
10. v5 的 `record_load(eid, cost)` 保留（load + prefetch 都记驻留收益）。

## 6. 测试策略

- 资源层：queue_depth / utilization / wait_time_ms（peek 不污染状态）单测。
- 策略层：EFT 选择（CPU 等待小 → CPU；PCIe 拥塞 → CPU；CPU 饱和 → GPU）；
  预取门控（PCIe 拥塞不预取；空闲预取热专家）；驻留稳定性。
- 适配器层：**prefetch 重叠**（预取传输隐藏于计算，步时间 < 串行化时间）；
  预取后请求专家无 PCIe 停顿；在途 load 不重复占带宽；确定性（两次运行逐位相同）。
- 回归：全部既有测试通过（重点 test_moe_adapter / test_v5_feedback / test_resources）。

## 7. 验证场景（benchmark）

`benchmarks/e2e/compare_queue_overlap.py`：热专家 trace（80% 请求集中在少数专家），
对比 cost_model / residency / overlap(无预取) / overlap(预取) 的 TPOT/吞吐/命中率/
overlap_ratio —— 期望 overlap 策略在热场景以预取重叠胜出，且 PCIe 拥塞时门控生效。

## 8. 不在 v6 范围（远期）

- 请求级并发（多请求共享资源的真正排队，DistServe 式时延分解）
- 生产级 RL 训练循环；vLLM kernel 级集成；INT4 kernel 优化
- 多 GPU 真机验证；KV+专家联合调度真机化
