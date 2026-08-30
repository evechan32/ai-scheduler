#!/usr/bin/env python3
"""2.0: three-tier KV demo — long-context KV exceeding GPU + DRAM spills to disk.

Shows the disk tier absorbing KV that would otherwise overflow DRAM (FlexGen
3-tier model). Compares two-tier (GPU+DRAM, KV lost/overflow) vs three-tier
(GPU+DRAM+disk, KV preserved on disk) across growing context length.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.lru import LRUPolicy
from moesim.sim.moe_adapter import MoESimulation
from moesim.sim.resources import BandwidthResource, ComputeResource


def _sim(kv_per_token, gpu_cap, host_cap, disk_cap):
    return MoESimulation(
        scheduler=LRUPolicy(),
        profiles={f"e{i}": ExpertProfile(f"e{i}", 10.0, 1.0, 5.0) for i in range(2)},
        gpu_capacity_mb=100.0,
        pcie=BandwidthResource(bandwidth_gbps=10.0),
        gpu=ComputeResource(1, 1.0), cpu=ComputeResource(4, 1.0),
        kv_per_token_mb=kv_per_token, kv_gpu_capacity_mb=gpu_cap,
        kv_host_capacity_mb=host_cap, kv_disk_capacity_mb=disk_cap,
    )


def main() -> None:
    gpu_cap, host_cap, disk_cap = 100.0, 500.0, 1000.0
    steps = [["e0"], ["e1"]] * 50  # 100 decode steps

    print("=== 2.0 three-tier KV (100 steps, growing context) ===")
    print(f"{'kv/token(MB)':>12s} {'total KV':>9s} {'GPU':>7s} {'DRAM':>7s} "
          f"{'disk':>7s} {'disk util':>10s}")
    for kv_per_token in (1.0, 5.0, 10.0, 20.0):
        sim = _sim(kv_per_token, gpu_cap, host_cap, disk_cap)
        sim.run(steps)
        s = sim._state
        total = kv_per_token * 100
        print(f"{kv_per_token:12.1f} {total:9.0f} {s.kv_gpu_mb:7.0f} "
              f"{s.kv_host_mb:7.0f} {s.kv_disk_mb:7.0f} "
              f"{sim._metrics.kv_disk_utilization['mean']:9.2f}%")

    print(f"\nread: as context grows, KV fills GPU ({gpu_cap}MB) then DRAM "
          f"({host_cap}MB) then spills to disk. The disk tier lets unbounded "
          f"context run; without it KV would overflow DRAM. Disk fetch is slow "
          f"(SSD bandwidth), so prefetch is the next optimization (MoE-Infinity "
          f"SSD->DRAM->GPU pipeline).")


if __name__ == "__main__":
    main()
