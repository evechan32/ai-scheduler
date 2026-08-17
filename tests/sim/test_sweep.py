from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.cost_model import CostModelPolicy
from moesim.scheduler.policies.lru import LRUPolicy
from moesim.sim.sweep import compare_policies


def _profiles(n=4, size_mb=10.0):
    return {
        f"e{i}": ExpertProfile(
            expert_id=f"e{i}", size_mb=size_mb, gpu_exec_ms=1.0, cpu_exec_ms=3.0,
            activation_freq=float(i),
        )
        for i in range(n)
    }


def test_compare_policies_returns_all_four():
    steps = [["e0", "e1"], ["e0", "e1"], ["e0", "e1"], ["e2", "e3"]]
    results = compare_policies(
        profiles=_profiles(),
        steps=steps,
        pcie_params={"bandwidth_gbps": 10.0},
        gpu_capacity_mb=20.0,
    )
    assert set(results.keys()) == {"lru", "activation_freq", "cost_model", "residency"}
    for metrics in results.values():
        assert metrics.total_tokens == 4
