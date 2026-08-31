"""v8: KV growth, overflow offload and pressure feedback in the simulator."""
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.lru import LRUPolicy
from moesim.sim.moe_adapter import MoESimulation
from moesim.sim.resources import BandwidthResource, ComputeResource


def _profiles(n=2):
    return {f"e{i}": ExpertProfile(f"e{i}", 10.0, 1.0, 5.0) for i in range(n)}


def _sim(kv_per_token_mb=0.0, kv_capacity_mb=0.0, kv_host_capacity_mb=0.0):
    return MoESimulation(
        scheduler=LRUPolicy(),
        profiles=_profiles(),
        gpu_capacity_mb=100.0,
        pcie=BandwidthResource(bandwidth_gbps=10.0),
        gpu=ComputeResource(1, 1.0),
        cpu=ComputeResource(4, 1.0),
        kv_per_token_mb=kv_per_token_mb,
        kv_gpu_capacity_mb=kv_capacity_mb,
        kv_host_capacity_mb=kv_host_capacity_mb,
    )


def test_kv_growth_consumes_gpu_pool():
    sim = _sim(kv_per_token_mb=1.0, kv_capacity_mb=100.0)
    sim.run([["e0"], ["e1"], ["e0"]])
    assert sim._state.kv_gpu_mb == 3.0
    assert sim._state.kv_host_mb == 0.0


def test_kv_overflow_offloads_to_host():
    sim = _sim(kv_per_token_mb=2.0, kv_capacity_mb=5.0, kv_host_capacity_mb=100.0)
    sim.run([["e0"], ["e1"], ["e0"], ["e1"]])  # 4 tokens x 2MB = 8MB
    assert sim._state.kv_gpu_mb == 5.0
    assert sim._state.kv_host_mb == 3.0
    assert sim._metrics.kv_offload_bytes == 3.0


def test_kv_pressure_feedback():
    sim = _sim(kv_per_token_mb=1.0, kv_capacity_mb=4.0)
    sim.run([["e0"], ["e1"], ["e0"], ["e1"]])
    assert sim._state.kv_pressure == 1.0


def test_zero_kv_per_token_no_change():
    sim = _sim(kv_per_token_mb=0.0, kv_capacity_mb=100.0)
    sim.run([["e0"], ["e1"], ["e0"]])
    assert sim._state.kv_gpu_mb == 0.0
    assert sim._metrics.kv_offload_bytes == 0.0


def test_kv_utilization_recorded():
    sim = _sim(kv_per_token_mb=1.0, kv_capacity_mb=4.0)
    sim.run([["e0"], ["e1"], ["e0"], ["e1"]])
    assert sim._metrics.kv_gpu_utilization["max"] == 1.0
    assert sim._metrics.kv_gpu_utilization["mean"] > 0.0
