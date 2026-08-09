from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.cost_model import CostModelPolicy
from moesim.scheduler.policies.lru import LRUPolicy
from moesim.sim.moe_adapter import MoESimulation
from moesim.sim.resources import BandwidthResource, ComputeResource


def _profiles(n=4, size_mb=10.0):
    return {
        f"e{i}": ExpertProfile(
            expert_id=f"e{i}", size_mb=size_mb, gpu_exec_ms=1.0, cpu_exec_ms=3.0,
            activation_freq=float(i),
        )
        for i in range(n)
    }


def test_lru_simulation_reuses_resident_experts():
    profiles = _profiles()
    pcie = BandwidthResource(bandwidth_gbps=10.0)   # 10MB => 1ms load
    gpu = ComputeResource(concurrency=1, per_unit_ms=1.0)
    sim = MoESimulation(
        scheduler=LRUPolicy(),
        profiles=profiles,
        gpu_capacity_mb=20.0,
        pcie=pcie,
        gpu=gpu,
        cpu=ComputeResource(concurrency=4, per_unit_ms=1.0),
    )
    metrics = sim.run(steps=[["e0", "e1"], ["e0", "e1"], ["e0", "e1"]])
    # step1: load e0,e1 (PCIe serialized: 1ms + 1ms), then GPU exec each
    #        (starts after its load; GPU serialized): e0 done@2, e1 done@3 -> 3ms
    # step2,3: pure GPU exec (1ms each, serialized) -> 2ms per step
    assert metrics.total_tokens == 3
    assert metrics.cache_hits == 4  # steps 2,3 hit e0,e1
    assert metrics.cache_misses == 2
    assert metrics.total_time_ms == 7.0  # 3 + 2 + 2
    assert metrics.hit_rate() == 4 / 6


def test_cost_model_policy_uses_cpu_for_expensive_loads():
    profiles = _profiles()
    pcie = BandwidthResource(bandwidth_gbps=1.0)    # 10MB => 10ms load: CPU (3ms) wins
    sim = MoESimulation(
        scheduler=CostModelPolicy(pcie=pcie, prefetch_n=0),
        profiles=profiles,
        gpu_capacity_mb=40.0,
        pcie=pcie,
        gpu=ComputeResource(concurrency=1, per_unit_ms=1.0),
        cpu=ComputeResource(concurrency=4, per_unit_ms=1.0),
    )
    metrics = sim.run(steps=[["e0"], ["e1"]])
    # e0,e1 computed on CPU (3ms each) — no PCIe loads
    assert metrics.total_time_ms == 6.0
