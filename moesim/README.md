# moesim

Heterogeneous MoE offloading scheduler + domain-agnostic discrete event simulator.

## What it is

moesim is a domain-agnostic discrete-event simulation (DES) engine paired with a
heterogeneous-computing-aware scheduler for Mixture-of-Experts (MoE) expert
offloading. Existing systems treat CPU offloading as a memory fallback; moesim
instead models CPU compute capacity as a first-class input to scheduling
decisions, choosing expert placement, PCIe transfers, and compute partitioning
jointly from per-expert CPU/GPU execution cost and available bandwidth. The
scheduler is a pure function (`decide(state, clock) -> actions`), so the same
policy code drives both the simulator and a real inference engine.

Pitch: domain-agnostic DES + heterogeneous MoE scheduler, where CPU compute
capacity participates in scheduling decisions.

## Quickstart

```bash
# install (simulator + scheduler need only numpy; no torch/GPU required)
pip install -e .

# run the test suite
python -m pytest tests/

# run the calibrated simulation demo (no GPU required)
python benchmarks/e2e/compare_moesim_vs_llamacpp.py
```

The demo loads the calibrated profiles in `benchmarks/microbench/out/` and
prints a per-policy TPOT / throughput / hit-rate table:

```
=== moesim simulation (calibrated) ===
lru              TPOT=   3.693ms  tput= 270.803 tok/s  hit=0.983
activation_freq  TPOT=   6.841ms  tput= 146.180 tok/s  hit=0.983
cost_model       TPOT=   3.456ms  tput= 289.385 tok/s  hit=0.158
```

## Architecture

The design splits into three layers plus a calibration loop. The authoritative
spec is `docs/superpowers/specs/2026-08-09-moesim-design.md` (§4).

- **Scheduler core** (`moesim/scheduler/`) — the pluggable decision function
  `decide(state, clock) -> actions`, with three v1 policies (`cost_model`,
  `activation_freq`, `lru`) and a per-expert `ExpertProfile` cost model
  (CPU/GPU execution time, weight size in MB, activation frequency).
- **Simulator** (`moesim/sim/`) — a domain-agnostic DES (`core.py`) with reusable
  resource models (`resources.py`: bandwidth, compute, storage), calibration
  (`calibrate.py`), and metric aggregation (`metrics.py`). It has no MoE
  dependencies; MoE is wired in through `moe_adapter.py`.
- **Executor** (`moesim/executor/`) — the `ExpertExecutor` abstraction and a
  transformers backend for routing experts to GPU/CPU at inference time.
- **Calibration loop** — microbenchmarks measure real PCIe bandwidth and
  per-expert execution time, which `calibrate.py` turns into simulator
  parameters.

The same `decide()` runs against both the simulator and the real engine.

## Hardware

Validated on a 12G GPU + 16G RAM machine with Qwen3-30B-A3B as the primary
model. The `sim/` and `scheduler/` layers are numpy-only and run without
torch or a GPU; only the executor backends and microbenchmarks require
torch/CUDA.

## Benchmarks

- `benchmarks/microbench/` — real-machine calibration: PCIe bandwidth and
  per-expert CPU/GPU execution time.
- `benchmarks/e2e/` — end-to-end verification protocol comparing simulated
  TPOT/throughput against llama.cpp `--n-cpu-moe` and MoE-Infinity on
  Qwen3-30B-A3B (target: < 20% error).

## Roadmap (v2)

From spec §8:

- Quantized CPU kernels (INT4/INT8); current CPU kernel is FP16 with a torch
  fallback.
- Real weight offloading via `accelerate` / `device_map`; the transformers
  executor load/unload is a no-op in v1.
- Full KV-cache tier scheduling policy, jointly scheduled with experts.
- `vllm` and `llama.cpp` executor backends.
- RL / imitation-learning policies trained on `sim/` as the environment.
- Multi-GPU / cluster topology simulation.
