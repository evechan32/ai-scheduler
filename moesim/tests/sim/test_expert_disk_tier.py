"""2.0: expert weight disk tier — cold experts on SSD, prefetch cost on activation."""
from moesim.scheduler.base import Action
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.lru import LRUPolicy
from moesim.sim.moe_adapter import MoESimulation
from moesim.sim.resources import BandwidthResource, ComputeResource


def _profiles(n=3):
    return {f"e{i}": ExpertProfile(f"e{i}", 100.0, 1.0, 5.0) for i in range(n)}


class DiskCpuScheduler(LRUPolicy):
    """Forces requested experts to CPU, so disk-tier cost is isolated."""

    def decide(self, state, clock):
        return [Action(kind="execute_cpu", expert_ids=(eid,)) for eid in state.requested]


def _sim(disk_gbps=2.0, disk_latency_ms=5.0):
    return MoESimulation(
        scheduler=DiskCpuScheduler(), profiles=_profiles(), gpu_capacity_mb=1000.0,
        pcie=BandwidthResource(bandwidth_gbps=10.0),
        gpu=ComputeResource(1, 1.0), cpu=ComputeResource(4, 1.0),
        disk_read_gbps=disk_gbps, disk_latency_ms=disk_latency_ms,
    )


def test_disk_expert_cpu_exec_is_slower():
    sim = _sim()
    sim._state.disk_experts = {"e0"}  # e0 on disk
    m = sim.run([["e0"], ["e1"]])
    # e0 (disk, 100MB @2GB/s +5ms = 55ms + cpu 5ms = 60ms) + e1 (cpu 5ms)
    assert m.total_time_ms == 65.0


def test_host_expert_cpu_exec_no_disk_cost():
    sim = _sim()
    m = sim.run([["e0"], ["e1"]])
    # both on host: 5ms + 5ms
    assert m.total_time_ms == 10.0


def test_disk_read_uses_disk_bandwidth_not_pcie():
    fast_disk = _sim(disk_gbps=1000.0, disk_latency_ms=0.0)  # fast SSD
    fast_disk._state.disk_experts = {"e0"}
    m = fast_disk.run([["e0"]])
    # 100MB @1000GB/s = 0.1ms + cpu 5ms = 5.1ms
    assert m.total_time_ms == 5.1
