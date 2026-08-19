from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.residency import ResidencyAwarePolicy
from moesim.scheduler.state import ScheduleState
from moesim.sim.resources import BandwidthResource


def _profiles(hot=False):
    return {"e0": ExpertProfile("e0", 10.0, 1.0, 5.0, activation_freq=0.9 if hot else 0.1)}


def test_resident_stays_gpu():
    pcie = BandwidthResource(bandwidth_gbps=10.0)
    s = ScheduleState(profiles=_profiles(), resident={"e0"}, gpu_capacity_mb=100.0,
                      requested=("e0",))
    actions = ResidencyAwarePolicy(pcie=pcie).decide(s, 0.0)
    assert any(a.kind == "execute_gpu" for a in actions)


def test_gpu_queue_forces_cpu():
    pcie = BandwidthResource(bandwidth_gbps=10.0)
    s = ScheduleState(profiles=_profiles(), resident=set(), gpu_capacity_mb=100.0,
                      requested=("e0",), gpu_queue_len=3)
    actions = ResidencyAwarePolicy(pcie=pcie, gpu_concurrency=1).decide(s, 0.0)
    assert any(a.kind == "execute_cpu" for a in actions)


def test_hot_expert_loads_despite_cost():
    pcie = BandwidthResource(bandwidth_gbps=10.0)  # 10MB load = 1ms
    s = ScheduleState(profiles=_profiles(hot=True), resident=set(), gpu_capacity_mb=100.0,
                      requested=("e0",), migration_cost_ms=1.0)
    s.record_load("e0", 5.0)  # 驻留价值 5ms > load 1ms
    actions = ResidencyAwarePolicy(pcie=pcie).decide(s, 0.0)
    assert any(a.kind == "load" for a in actions)
