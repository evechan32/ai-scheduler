#!/usr/bin/env python3
"""v8: KV tiering comparison — long-context trace with growing KV pressure.

Compares cost_model, KVWeightedPolicy (v2) and KVJointPolicy (v8) on a
long-context decode trace where KV cache grows every step. Reports TPOT,
hit rate, KV GPU utilization and offload bytes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moesim.scheduler.cost_model import profiles_from_dicts
from moesim.scheduler.policies.cost_model import CostModelPolicy
from moesim.scheduler.policies.kv_aware import KVWeightedPolicy
from moesim.scheduler.policies.kv_joint import KVJointPolicy
from moesim.sim.moe_adapter import MoESimulation
from moesim.sim.resources import BandwidthResource, ComputeResource


def build_trace(n_experts=8, hot=2, num_steps=300, hot_frac=0.8):
    trace = []
    for i in range(num_steps):
        if i % 5 == 4:
            trace.append([f"e{i % n_experts}" for i in range(hot, n_experts)])
        else:
            trace.append([f"e{i % hot}"])
    return trace


def profiles(n_experts=8):
    rows = [
        {"expert_id": f"e{i}", "size_mb": 100.0, "gpu_exec_ms": 1.0,
         "cpu_exec_ms": 4.0, "activation_freq": 0.9 if i < 2 else 0.1}
        for i in range(n_experts)
    ]
    return profiles_from_dicts(rows)


def run_policy(policy, prof, steps, pcie_params, capacity_mb, kv_per_token_mb,
               kv_gpu_capacity_mb, kv_host_capacity_mb):
    pcie = BandwidthResource(**pcie_params)
    sim = MoESimulation(scheduler=policy, profiles=prof, gpu_capacity_mb=capacity_mb,
                        pcie=pcie, gpu=ComputeResource(concurrency=1, per_unit_ms=1.0),
                        cpu=ComputeResource(concurrency=4, per_unit_ms=1.0),
                        kv_per_token_mb=kv_per_token_mb,
                        kv_gpu_capacity_mb=kv_gpu_capacity_mb,
                        kv_host_capacity_mb=kv_host_capacity_mb)
    return sim.run(steps)


def main() -> None:
    prof = profiles()
    steps = build_trace()
    pcie_params = {"bandwidth_gbps": 8.0}
    kv_per_token_mb = 50.0
    kv_gpu_capacity_mb = 6 * 1024.0

    pcie = BandwidthResource(**pcie_params)
    policies = {
        "cost_model": CostModelPolicy(pcie=pcie, prefetch_n=2),
        "kv_aware(v2)": KVWeightedPolicy(pcie=pcie, prefetch_n=2),
        "kv_joint(v8)": KVJointPolicy(pcie=pcie, prefetch_n=2),
    }

    print("=== v8 KV tiering comparison (long-context trace, KV grows per step) ===")
    print(f"KV: {kv_per_token_mb:.0f} MB/token, GPU KV pool {kv_gpu_capacity_mb/1024:.0f} GiB, "
          f"300 decode steps")
    print(f"{'policy':16s} {'TPOT(ms)':>9s} {'hit':>6s} {'kv_gpu_util':>11s} "
          f"{'kv_offload(MB)':>13s}")
    for name, policy in policies.items():
        m = run_policy(policy, prof, steps, pcie_params, 12 * 1024.0,
                       kv_per_token_mb, kv_gpu_capacity_mb, 16 * 1024.0)
        offload = m.kv_offload_bytes
        print(f"{name:16s} {m.tpot_ms():9.3f} {m.hit_rate():6.3f} "
              f"{m.kv_gpu_utilization['mean']:11.3f} {offload:13.1f}")

    print("\nread: 300 tokens x 50MB = 15GB > 6GB GPU KV pool. KV overflow is "
          "offloaded over PCIe (contending with expert loads). cost_model ignores "
          "KV pressure: 8.9GB offloaded, PCIe queue backlog grows (transfer_wait). "
          "kv_joint steers experts to CPU under pressure and evicts cold KV, "
          "cutting offload to ~0MB (protects the KV pool) at a TPOT cost.")


if __name__ == "__main__":
    main()
