#!/usr/bin/env python3
"""v6: queueing & overlap comparison — EFT placement + prefetch overlap vs baselines.

Hot-expert trace (80% requests hit 2 hot experts). Compares cost_model,
residency, and OverlapAwarePolicy with and without prefetch, reporting TPOT,
throughput, hit rate, PCIe utilization, and overlap ratio.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moesim.scheduler.cost_model import profiles_from_dicts
from moesim.scheduler.policies.cost_model import CostModelPolicy
from moesim.scheduler.policies.lru import LRUPolicy
from moesim.scheduler.policies.overlap import OverlapAwarePolicy
from moesim.scheduler.policies.residency import ResidencyAwarePolicy
from moesim.sim.moe_adapter import MoESimulation
from moesim.sim.resources import BandwidthResource, ComputeResource


def build_trace(n_experts=8, hot=2, num_steps=200, hot_frac=0.8):
    trace = []
    for i in range(num_steps):
        if i % 5 == 4:
            trace.append([f"e{i % n_experts}" for i in range(hot, n_experts)])
        else:
            trace.append([f"e{i % hot}"])
    return trace


def profiles(n_experts=8, size_mb=100.0, gpu_exec_ms=1.0, cpu_exec_ms=4.0):
    rows = [
        {"expert_id": f"e{i}", "size_mb": size_mb, "gpu_exec_ms": gpu_exec_ms,
         "cpu_exec_ms": cpu_exec_ms, "activation_freq": 0.9 if i < 2 else 0.1}
        for i in range(n_experts)
    ]
    return profiles_from_dicts(rows)


def run_policy(policy, prof, steps, pcie_params, capacity_mb):
    pcie = BandwidthResource(**pcie_params)
    sim = MoESimulation(scheduler=policy, profiles=prof, gpu_capacity_mb=capacity_mb,
                        pcie=pcie, gpu=ComputeResource(concurrency=1, per_unit_ms=1.0),
                        cpu=ComputeResource(concurrency=4, per_unit_ms=1.0))
    return sim.run(steps)


def main() -> None:
    prof = profiles()
    steps = build_trace()
    pcie_params = {"bandwidth_gbps": 8.0}
    capacity_mb = 12 * 1024.0

    pcie = BandwidthResource(**pcie_params)
    policies = {
        "lru": LRUPolicy(),
        "cost_model": CostModelPolicy(pcie=pcie, prefetch_n=2),
        "residency": ResidencyAwarePolicy(pcie=pcie),
        "overlap(no-pf)": OverlapAwarePolicy(pcie=pcie, prefetch_n=0),
        "overlap(pf=2)": OverlapAwarePolicy(pcie=pcie, prefetch_n=2),
    }

    transfer_ms = pcie.transfer_time_ms(100.0)
    print("=== v6 queueing & overlap comparison (hot-expert trace) ===")
    print(f"trace: 200 decode steps, hot e0/e1 on 4/5 steps; 100MB experts, "
          f"PCIe {pcie_params['bandwidth_gbps']}GB/s ({transfer_ms:.1f}ms/load); "
          f"CPU 4ms vs GPU 1ms per expert")
    print(f"{'policy':16s} {'TPOT(ms)':>9s} {'tput':>9s} {'hit':>6s} "
          f"{'pcie_util':>9s} {'overlap':>7s} {'prefetch':>8s}")
    for name, policy in policies.items():
        m = run_policy(policy, prof, steps, pcie_params, capacity_mb)
        print(f"{name:16s} {m.tpot_ms():9.3f} {m.throughput_tok_s():9.1f} "
              f"{m.hit_rate():6.3f} {m.pcie_utilization:9.3f} "
              f"{m.overlap_ratio():7.3f} {m.prefetch_count:8d}")
    print("\nread: residency/overlap(no-pf) keep everything on CPU (12.5ms load > 4ms "
          "CPU); prefetch makes hot experts resident in the background so their "
          "execution moves to GPU — overlap(pf=2) is fastest.")

    congested = dict(pcie_params)
    congested["bandwidth_gbps"] = 2.0
    print("\n=== PCIe congested (2 GB/s) — prefetch gate limits background traffic ===")
    for name in ("overlap(no-pf)", "overlap(pf=2)"):
        policy = policies[name]
        m = run_policy(policy, prof, steps, congested, capacity_mb)
        print(f"{name:16s} TPOT={m.tpot_ms():8.3f}ms  hit={m.hit_rate():.3f}  "
              f"pcie_util={m.pcie_utilization:.3f}  overlap={m.overlap_ratio():.3f}  "
              f"prefetch={m.prefetch_count}")


if __name__ == "__main__":
    main()
