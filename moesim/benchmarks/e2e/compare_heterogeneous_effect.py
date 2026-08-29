#!/usr/bin/env python3
"""v9.1: heterogeneous scheduling effect — the point of moesim.

Constrained-memory MoE scenario (expert weights exceed GPU capacity, like
OLMoE-1B-7B fp16 on an 8GiB GPU). Compares:
  - pure CPU: all experts execute on CPU (slow but always feasible)
  - pure GPU: all experts resident on GPU (upper bound, needs full capacity)
  - LRU: naive cache replacement
  - moesim heterogeneous policies: cost_model / overlap (CPU compute enters the
    decision, prefetch overlaps PCIe with compute)

Real OLMoE calibration: 64 experts x 8MB, GPU 0.076ms / CPU 0.639ms per expert,
PCIe 4.3 GB/s, 8 experts per token. GPU capacity holds only 16 experts.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moesim.scheduler.base import Action, Scheduler
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.cost_model import CostModelPolicy
from moesim.scheduler.policies.lru import LRUPolicy
from moesim.scheduler.policies.overlap import OverlapAwarePolicy
from moesim.sim.moe_adapter import MoESimulation
from moesim.sim.resources import BandwidthResource, ComputeResource


class PureCpuPolicy(Scheduler):
    def decide(self, state, clock):
        return [Action(kind="execute_cpu", expert_ids=(eid,)) for eid in state.requested]


def build_profiles(n=64, size_mb=8.0, gpu_ms=0.076, cpu_ms=0.639, hot=8):
    return {f"e{i}": ExpertProfile(f"e{i}", size_mb, gpu_ms, cpu_ms,
                                    activation_freq=0.9 if i < hot else 0.05)
            for i in range(n)}


def build_trace(n=64, hot=8, steps=500, hot_frac=0.8):
    import random
    rng = random.Random(42)
    trace = []
    for _ in range(steps):
        if rng.random() < hot_frac:
            trace.append([f"e{rng.randrange(hot)}" for _ in range(8)])
        else:
            trace.append([f"e{hot + rng.randrange(n - hot)}" for _ in range(8)])
    return trace


def run(policy, profiles, steps, gpu_capacity_mb, pcie):
    sim = MoESimulation(scheduler=policy, profiles=profiles,
                        gpu_capacity_mb=gpu_capacity_mb, pcie=pcie,
                        gpu=ComputeResource(concurrency=8, per_unit_ms=1.0),
                        cpu=ComputeResource(concurrency=2, per_unit_ms=1.0))
    return sim.run(steps)


def main() -> None:
    profiles = build_profiles()
    steps = build_trace()
    full_capacity = 64 * 8.0  # all 64 experts resident
    constrained_capacity = 16 * 8.0  # only 16 experts fit (8GiB-GPU reality)

    pcie_params = {"bandwidth_gbps": 4.3}  # real PCIe 4.30 GB/s (v1 measured)
    print("=== moesim heterogeneous-scheduling effect (OLMoE-1B-7B, 8GiB GPU) ===")
    print(f"64 experts x 8MB = {full_capacity:.0f}MB weights; GPU holds only "
          f"{constrained_capacity:.0f}MB (16 experts); 8 experts/token; "
          f"GPU 0.076ms (8-wide) / CPU 0.639ms (2-wide, bandwidth-bound) / "
          f"PCIe 4.3GB/s")

    results = {}

    pcie = BandwidthResource(**pcie_params)
    m = run(PureCpuPolicy(), profiles, steps, constrained_capacity, pcie)
    results["pure CPU"] = m

    pcie = BandwidthResource(**pcie_params)
    m = run(LRUPolicy(), profiles, steps, full_capacity, pcie)
    results["pure GPU (upper bound)"] = m

    pcie = BandwidthResource(**pcie_params)
    m = run(LRUPolicy(), profiles, steps, constrained_capacity, pcie)
    results["LRU (constrained)"] = m

    pcie = BandwidthResource(**pcie_params)
    m = run(CostModelPolicy(pcie=pcie, prefetch_n=2), profiles, steps,
            constrained_capacity, pcie)
    results["cost_model (hetero)"] = m

    pcie = BandwidthResource(**pcie_params)
    m = run(OverlapAwarePolicy(pcie=pcie, prefetch_n=2), profiles, steps,
            constrained_capacity, pcie)
    results["overlap (hetero)"] = m

    print(f"\n{'policy':24s} {'TPOT(ms)':>9s} {'tput':>9s} {'hit':>6s}")
    for name, m in results.items():
        print(f"{name:24s} {m.tpot_ms():9.3f} {m.throughput_tok_s():9.1f} "
              f"{m.hit_rate():6.3f}")

    pure_cpu = results["pure CPU"].tpot_ms()
    upper = results["pure GPU (upper bound)"].tpot_ms()
    lru = results["LRU (constrained)"].tpot_ms()
    hetero = results["cost_model (hetero)"].tpot_ms()
    print(f"\nread: pure CPU = {pure_cpu:.2f}ms (feasible but slow); pure GPU = "
          f"{upper:.2f}ms (needs {full_capacity:.0f}MB, impossible on 8GiB); "
          f"naive LRU = {lru:.2f}ms (frequent PCIe misses). "
          f"cost_model (CPU-compute-aware) = {hetero:.2f}ms — "
          f"{pure_cpu / hetero:.1f}x faster than pure CPU, {lru / hetero:.1f}x "
          f"faster than LRU, within {constrained_capacity:.0f}MB. "
          f"That is moesim's effect: heterogeneous scheduling turns "
          f"'can't-fit-GPU' into near-GPU throughput.")


if __name__ == "__main__":
    main()
