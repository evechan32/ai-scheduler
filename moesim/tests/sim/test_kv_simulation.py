import pytest

from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.kv_aware import KVWeightedPolicy
from moesim.sim.moe_adapter import MoESimulation
from moesim.sim.resources import BandwidthResource, ComputeResource


def _profiles() -> dict[str, ExpertProfile]:
    return {f"e{i}": ExpertProfile(expert_id=f"e{i}", size_mb=100.0,
            gpu_exec_ms=1.0, cpu_exec_ms=5.0) for i in range(4)}


def test_kv_policy_evict_increases_sim_time():
    """Under KV pressure, evict_kv triggers a PCIe transfer that adds to step time.

    BandwidthResource is a single-lane serialized channel, so a 100MB evict at
    10GB/s costs 100/10 = 10ms. KV pressure 0.99 forces KVWeightedPolicy to emit
    evict_kv (alongside a 5ms CPU execution of the requested expert). The step
    completes at max(5ms CPU, 10ms PCIe evict) = 10ms, so the evict transfer is
    the critical path and must show up in tpot. Without KV timing the step would
    end at 5ms.
    """
    profiles = _profiles()
    pcie = BandwidthResource(bandwidth_gbps=10.0)
    sim = MoESimulation(
        scheduler=KVWeightedPolicy(pcie=pcie, prefetch_n=0),
        profiles=profiles,
        gpu_capacity_mb=1000.0,
        pcie=pcie,
        gpu=ComputeResource(concurrency=1, per_unit_ms=1.0),
        cpu=ComputeResource(concurrency=4, per_unit_ms=1.0),
    )
    sim._state.kv_gpu_capacity_mb = 100.0
    sim._state.kv_gpu_mb = 99.0
    sim._state.access_history = ["e1", "e2", "e3"]
    metrics = sim.run([["e0"]])
    assert metrics.tpot_ms() == pytest.approx(10.0)
