#!/usr/bin/env python3
"""2.0: expert disk-tier demo — cold experts demoted to SSD save DRAM.

Three-tier expert placement (GPU/DRAM/disk) vs two-tier (GPU/DRAM): the disk
tier lets expert weights exceed DRAM by demoting the coldest experts to SSD.
Trade-off: activating a disk expert pays a slow SSD read.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.cost_model import CostModelPolicy
from moesim.scheduler.policies.disk_tier import DiskTierPolicy
from moesim.sim.moe_adapter import MoESimulation
from moesim.sim.resources import BandwidthResource, ComputeResource


def _profiles(n=32, hot=8):
    return {f"e{i}": ExpertProfile(f"e{i}", 8.0, 0.076, 0.639,
                                    activation_freq=0.9 if i < hot else 0.05)
            for i in range(n)}


def _trace(n=32, hot=8, steps=500, hot_frac=0.9):
    import random
    rng = random.Random(42)
    out = []
    for _ in range(steps):
        if rng.random() < hot_frac:
            out.append([f"e{rng.randrange(hot)}"])
        else:
            out.append([f"e{hot + rng.randrange(n - hot)}"])
    return out


def _run(policy, profiles, steps, pcie):
    return MoESimulation(
        scheduler=policy, profiles=profiles, gpu_capacity_mb=256.0, pcie=pcie,
        gpu=ComputeResource(8, 1.0), cpu=ComputeResource(2, 1.0),
        disk_read_gbps=2.0, disk_latency_ms=5.0,
    ).run(steps)


def main() -> None:
    profiles = _profiles()
    steps = _trace()
    total_w = sum(p.size_mb for p in profiles.values())

    print(f"=== 2.0 expert disk tier ({len(profiles)} experts x 8MB = "
          f"{total_w:.0f}MB weights) ===")
    print(f"{'policy':24s} {'TPOT(ms)':>9s} {'hit':>6s} {'disk experts':>12s}")

    pcie = BandwidthResource(bandwidth_gbps=4.3)
    m = _run(CostModelPolicy(pcie=pcie, prefetch_n=0), profiles, steps, pcie)
    print(f"{'two-tier (GPU/DRAM)':24s} {m.tpot_ms():9.3f} {m.hit_rate():6.3f} "
          f"{'n/a':>12s}")

    pcie = BandwidthResource(bandwidth_gbps=4.3)
    sim = MoESimulation(
        scheduler=DiskTierPolicy(pcie=pcie, prefetch_n=0, disk_budget_mb=128.0),
        profiles=profiles, gpu_capacity_mb=256.0, pcie=pcie,
        gpu=ComputeResource(8, 1.0), cpu=ComputeResource(2, 1.0),
        disk_read_gbps=2.0, disk_latency_ms=5.0,
    )
    m = sim.run(steps)
    print(f"{'three-tier (GPU/DRAM/disk)':24s} {m.tpot_ms():9.3f} "
          f"{m.hit_rate():6.3f} {len(sim._state.disk_experts):>12d}")

    print(f"\nread: disk tier demotes the {len(sim._state.disk_experts)} coldest "
          f"experts ({len(sim._state.disk_experts)*8:.0f}MB) to SSD, freeing DRAM. "
          f"Cold experts are rarely activated, so the slow SSD read (2GB/s) only "
          f"hurts the ~{100*(1-0.9):.0f}% cold steps. This is how a model larger "
          f"than DRAM fits on one machine (FlexGen / MoE-Infinity).")


if __name__ == "__main__":
    main()
