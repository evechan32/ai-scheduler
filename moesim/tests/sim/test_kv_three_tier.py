"""2.0: KV cache three-tier — GPU overflow to DRAM, DRAM overflow to disk."""
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.lru import LRUPolicy
from moesim.sim.moe_adapter import MoESimulation
from moesim.sim.resources import BandwidthResource, ComputeResource


def _profiles(n=2):
    return {f"e{i}": ExpertProfile(f"e{i}", 10.0, 1.0, 5.0) for i in range(n)}


def _sim(kv_per_token=1.0, gpu_cap=10.0, host_cap=20.0, disk_cap=100.0):
    return MoESimulation(
        scheduler=LRUPolicy(), profiles=_profiles(), gpu_capacity_mb=100.0,
        pcie=BandwidthResource(bandwidth_gbps=10.0),
        gpu=ComputeResource(1, 1.0), cpu=ComputeResource(4, 1.0),
        kv_per_token_mb=kv_per_token, kv_gpu_capacity_mb=gpu_cap,
        kv_host_capacity_mb=host_cap, kv_disk_capacity_mb=disk_cap,
    )


def test_kv_overflows_gpu_then_host_then_disk():
    sim = _sim(kv_per_token=10.0, gpu_cap=10.0, host_cap=20.0)
    # 4 tokens x 10MB = 40MB: gpu 10 + host 20 + disk 10
    sim.run([["e0"], ["e1"], ["e0"], ["e1"]])
    assert sim._state.kv_gpu_mb == 10.0
    assert sim._state.kv_host_mb == 20.0
    assert sim._state.kv_disk_mb == 10.0


def test_kv_disk_growth_when_host_full():
    sim = _sim(kv_per_token=10.0, gpu_cap=10.0, host_cap=20.0)
    sim.run([["e0"], ["e1"], ["e0"], ["e1"], ["e0"], ["e1"]])
    # 60MB total: gpu 10 + host 20 + disk 30
    assert sim._state.kv_disk_mb == 30.0


def test_kv_disk_metrics_recorded():
    sim = _sim(kv_per_token=10.0, gpu_cap=10.0, host_cap=20.0)
    sim.run([["e0"], ["e1"], ["e0"], ["e1"]])
    assert sim._metrics.kv_offload_bytes == 30.0  # host(20 overflow) + disk(10)
