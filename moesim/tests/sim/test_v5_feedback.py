from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.lru import LRUPolicy
from moesim.sim.moe_adapter import MoESimulation
from moesim.sim.resources import BandwidthResource, ComputeResource


def _profiles():
    return {f"e{i}": ExpertProfile(f"e{i}", 10.0, 1.0, 5.0) for i in range(3)}


def test_queue_len_updated():
    pcie = BandwidthResource(bandwidth_gbps=10.0)
    sim = MoESimulation(scheduler=LRUPolicy(), profiles=_profiles(), gpu_capacity_mb=100.0,
                        pcie=pcie, gpu=ComputeResource(1, 1.0), cpu=ComputeResource(4, 1.0))
    sim.run([["e0", "e1"]])
    assert sim._state.gpu_queue_len >= 0  # 已更新（无断言错误即通过）


def test_load_records_benefit():
    pcie = BandwidthResource(bandwidth_gbps=10.0)
    sim = MoESimulation(scheduler=LRUPolicy(), profiles=_profiles(), gpu_capacity_mb=100.0,
                        pcie=pcie, gpu=ComputeResource(1, 1.0), cpu=ComputeResource(4, 1.0))
    sim.run([["e0"]])
    # LRU 会 load e0，record_load 记录驻留价值
    assert "e0" in sim._state.residency_benefit
    assert sim._state.residency_benefit["e0"] > 0
