# Changelog

All notable changes to moesim are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/) conventions.

## [v6] - 2026-08-20

### Added — 排队与重叠感知调度（queueing & overlap-aware scheduling）

- `sim/resources.py`: BandwidthResource / ComputeResource 暴露队列可见性 —
  `queue_depth(now)`（在途+排队作业数）、`utilization(until)`（忙时占比）、
  `wait_time_ms(now, units)`（peek 等待估计，不污染资源状态）。
- `sim/metrics.py`: 新增队列深度均值/最大值、PCIe/GPU/CPU 利用率、
  transfer_wait_ms、hidden_transfer_ms、prefetch_count、overlap_ratio。
- `scheduler/base.py`: 新增 `prefetch` Action kind（apply_actions 与 load 同语义）。
- `scheduler/state.py`: 新增资源反馈字段（pcie_queue_len / 利用率 / 等待时间估计 /
  pending_loads 在途传输表）。
- `scheduler/policies/overlap.py`: `OverlapAwarePolicy` — HEFT 式 EFT 放置
  （排队等待 + 执行时间取最小者，CPU 资源影响经 cpu_wait/contention 计入）
  + 带宽门控预取（PCIe 队列深度/利用率超限时不预取，MoE-Infinity 谨慎原则）。
- `sim/moe_adapter.py`: prefetch 传输与当前步计算重叠（不进关键路径）；
  pending_loads 跨步追踪在途传输，load 复用在途传输不重复占带宽；
  执行起点 = max(clock, 在途传输完成时间)；每步反馈资源队列/利用率/等待快照。

### Fixed

- `scheduler/policies/residency.py`: load 前容量检查缺失（v6 资源积压反馈语义暴露），
  超容量时回退 execute_cpu。

### Changed

- `sim/moe_adapter.py`: v5 的 `gpu_queue_len` 语义从「本步 GPU 执行专家数」改为
  「资源真实积压」（queue_depth），调度器看到真实排队影响。

### Benchmarks

- `benchmarks/e2e/compare_queue_overlap.py`: 热专家 trace 对比
  lru / cost_model / residency / overlap(no-pf) / overlap(pf=2)。
  实测：overlap(pf=2) TPOT 2.368ms，比全 CPU 放置（residency 4.800ms）快 50.7%，
  比 lru（2.475ms）快 4.3%；PCIe 拥塞时预取门控生效。

### Docs

- 新增 `docs/research/2026-08-20-queue-overlap-heterogeneous-survey.md`
  （排队/CPU/并行/重叠方向论文与高星项目综述，arXiv ID 已核实）。
- 新增 `docs/research/2026-08-20-paper-implementation-trace.md`
  （论文-实现追踪：机制 → 代码落点 → 参考程度 → 简化点）。
- 新增 `docs/superpowers/specs/2026-08-20-moesim-v6-design.md`。
- 新增 `docs/superpowers/plans/2026-08-20-moesim-v6.md`（TDD 计划）。

### Tests

- 107 passed, 3 skipped（1 个 INT4 kernel 测试为本机环境性失败，基线存在）。

## [v5] - 2026-08-17

- ResidencyAwarePolicy：GPU 排队感知、CPU 资源感知、迁移成本 + 驻留收益、驻留稳定性。
- 实测：热专家场景比 cost_model 快 46.9%（0.221ms vs 0.416ms）。

## [v4] - 2026-08-16

- 真并行执行（线程池并发 CPU/GPU）、决策缓存、REINFORCE 策略梯度、INT4 gemm、
  一键安装脚本、论文综述 50+ 篇。

## [v3] - 2026-08-15

- MoEForwardHook 真实推理闭环（误差 0.062%）、RL 策略（Q-learning）、多 GPU 拓扑。

## [v2] - 2026-08-15

- 真实权重 offload、KVWeightedPolicy、INT8/INT4 量化 kernel、vllm/llama.cpp 后端。

## [v1] - 2026-08-09

- 领域无关 DES 核心、资源模型、三种调度策略、CPU FP16 kernel、校准回路。
