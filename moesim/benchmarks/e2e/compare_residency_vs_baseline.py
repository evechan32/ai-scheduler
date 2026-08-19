#!/usr/bin/env python3
"""Compare ResidencyAwarePolicy against baseline policies on a hot-heavy trace."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from moesim.scheduler.cost_model import ExpertProfile
from moesim.sim.sweep import compare_policies

PCIe_PARAMS = {"bandwidth_gbps": 10.0, "latency_ms": 0.1}
GPU_CAPACITY_MB = 12000.0
HOT = ("e0", "e1")
COLD = ("e5", "e6")


def build_profiles(n=256) -> dict[str, ExpertProfile]:
    """Synthetic profiles so the script runs standalone without microbench outputs.

    Hot experts are worth loading (cheap on GPU, slow on CPU); cold experts are
    cheap on CPU and heavier on GPU, so keeping them resident wastes GPU slots.
    """
    profs = {}
    for i in range(n):
        eid = f"e{i}"
        if eid in HOT:
            profs[eid] = ExpertProfile(eid, size_mb=10.0, gpu_exec_ms=0.1,
                                       cpu_exec_ms=5.0, activation_freq=0.9)
        else:
            profs[eid] = ExpertProfile(eid, size_mb=10.0, gpu_exec_ms=0.5,
                                       cpu_exec_ms=0.05, activation_freq=0.05)
    return profs


def build_trace(steps=100) -> list[list[str]]:
    """Hot experts on 80 steps, hot + cold on 20 steps."""
    return [list(HOT) + list(COLD) if i % 5 == 4 else list(HOT) for i in range(steps)]


def main() -> None:
    profiles = build_profiles()
    steps = build_trace()
    hot_steps = sum(1 for s in steps if set(s) == set(HOT))
    cold_steps = len(steps) - hot_steps
    print(f"=== residency vs baseline on hot trace ({len(steps)} steps: "
          f"{hot_steps} hot-only, {cold_steps} hot+cold) ===")
    results = compare_policies(profiles=profiles, steps=steps,
                               pcie_params=PCIe_PARAMS, gpu_capacity_mb=GPU_CAPACITY_MB)
    for name, m in results.items():
        print(f"{name:16s} TPOT={m.tpot_ms():8.3f}ms  tput={m.throughput_tok_s():8.3f} tok/s  "
              f"hit={m.hit_rate():.3f}")
    residency = results["residency"].tpot_ms()
    cost_model = results["cost_model"].tpot_ms()
    margin = (cost_model - residency) / cost_model * 100
    assert residency <= cost_model, (
        f"residency TPOT {residency:.3f}ms > cost_model {cost_model:.3f}ms"
    )
    print(f"\nPASS: residency TPOT {residency:.3f}ms <= cost_model {cost_model:.3f}ms "
          f"({margin:.1f}% faster)")


if __name__ == "__main__":
    main()
