# moesim v8 — KV cache 分层模拟 + KV-专家联合调度（设计规范）

- 日期：2026-08-20
- 前置：v6（EFT + prefetch 重叠）、v7（混合精度放置）
- 依据：Mooncake（arXiv:2407.00079）、FlexGen（arXiv:2303.06865）、LMCache
  （arXiv:2406.14403）、HiCache（SGLang）——综述 §B 已核实

## 1. 背景与动机（用户需求）

用户要求两个能力：
1. **能够模拟** —— KV cache 目前只是记账（evict/fetch 按专家 size 走 PCIe 计时），
   没有 KV 随 decode 增长、没有 GPU KV 池容量约束、没有超限下放。
2. **异构算力调度 + KV cache 下放** —— 现有 KVWeightedPolicy（v2）只做"压力>0.9 强制
   CPU"，未与 v6 EFT/prefetch 结合；KV 下放（GPU 池满 → 主机）没有模拟。

v8 把 KV cache 建成一等公民：**KV 分层模拟（GPU/主机）+ 压力反馈 + 联合调度策略**。

## 2. 现状缺陷（v2-v6 遗留）

| # | 缺陷 | 位置 |
|---|------|------|
| 1 | 无 KV 增长模拟：decode 产生的 KV 字节不进入模拟 | `sim/moe_adapter.py` |
| 2 | 无 GPU KV 池容量水位管理：KV 只靠策略显式 evict | `scheduler/state.py` |
| 3 | 压力反馈是"专家执行数"级的（v2 KVWeightedPolicy 只读 kv_gpu_mb/capacity），未进 v6 EFT | `scheduler/policies/kv_aware.py` |
| 4 | 无 KV 利用率/迁移量指标 | `sim/metrics.py` |

## 3. 设计决策

| 维度 | 决策 |
|------|------|
| KV 增长 | adapter 每步 `kv_gpu_mb += token_count × kv_per_token_mb`（模型参数） |
| 超限下放 | GPU KV 池满 → 超出部分**自动下放到主机**（水位管理，确定性） |
| 压力反馈 | `state.kv_pressure = kv_gpu_mb / capacity` 每步快照（供策略消费） |
| 联合调度 | 新策略 `KVJointPolicy`：继承 v6 `OverlapAwarePolicy` EFT + KV 高压分支（专家倾向 CPU、暂停 prefetch、主动 evict） |
| 显式动作 | `evict_kv`/`fetch_kv` 保持旧语义（按专家 size 记账，测试兼容）；自动下放独立于显式动作 |
| 兼容性 | 新字段默认值向后兼容；既有 116 测试不破坏 |

## 4. 接口定义

### 4.1 `scheduler/state.py`（新增字段）

- `kv_per_token_mb: float = 0.0`
- `kv_pressure: float = 0.0`（kv_gpu_mb / kv_gpu_capacity_mb，adapter 快照）
- `kv_evict_count: int = 0`、`kv_fetch_count: int = 0`（显式动作计数）

### 4.2 `sim/moe_adapter.py`

- `MoESimulation.__init__(..., kv_per_token_mb: float = 0.0)`。
- `_step()` 中 KV 水位管理（在反馈快照之后、decide 之前）：
  1. `new_kv = token_count × kv_per_token_mb`；`kv_gpu_mb += new_kv`
  2. 超容量：`excess = kv_gpu_mb - kv_gpu_capacity_mb`（>0 时）
     `kv_host_mb += excess`；`kv_gpu_mb = capacity`；记录 `kv_offload_bytes` 指标
  3. 快照 `state.kv_pressure`
- 显式 evict_kv/fetch_kv 动作仍由 apply_actions + PCIe 计时处理（不变）。

### 4.3 `sim/metrics.py`（新增）

- `kv_gpu_utilization`（均值/max：kv_gpu_mb/capacity 每步采样）
- `kv_host_utilization`（kv_host_mb/host 容量，host 容量 = 参数）
- `kv_offload_bytes`（累计下放字节）、`kv_evict_count`、`kv_fetch_count`

### 4.4 `scheduler/policies/kv_joint.py` — `KVJointPolicy`

```
KVJointPolicy(OverlapAwarePolicy):
  __init__(..., kv_pressure_threshold=0.8)
  decide(state, clock):
    pressure = state.kv_pressure
    if pressure > threshold:
      # KV 高压：省 GPU 显存给 KV
      - 非驻留专家一律 execute_cpu（不再 load/prefetch 进 GPU）
      - 暂停 prefetch
      - 若 GPU KV 仍超（pressure>=1.0）→ evict_kv 最冷非请求 KV（旧语义，按专家 size）
    else:
      - 正常 v6 EFT + prefetch（重叠、混合精度）
```

## 5. 测试策略

- 模拟层：KV 增长消耗 GPU 池；超限自动下放主机（数值断言）；压力反馈值。
- 策略层：高压 → CPU + 无 prefetch + evict；低压 → 正常 EFT/prefetch。
- 指标层：kv 利用率/下放字节累计。
- 回归：既有 116 测试（重点 test_kv_simulation / test_apply_actions_kv）。

## 6. 验证场景

`benchmarks/e2e/compare_kv_tiering.py`：长上下文 trace（KV 持续增长），对比
cost_model / KVWeightedPolicy / KVJointPolicy 的 TPOT + KV 下放量 + KV 利用率——
期望 KVJointPolicy 在 KV 高压时避免显存挤压（CPU 化专家 + 下放 KV）。

## 7. 不在 v8 范围（路线图）

- 请求级并发模拟（DistServe 五段时延分解）
- 每请求 KV 记账（当前为池级水位）
- 磁盘层（FlexGen 三层存储）
