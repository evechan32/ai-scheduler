"""v6: OverlapAwarePolicy — EFT placement with queueing + gated prefetch."""
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.overlap import OverlapAwarePolicy
from moesim.scheduler.state import ScheduleState
from moesim.sim.resources import BandwidthResource


def _profiles(hot=False):
    return {"e0": ExpertProfile("e0", 10.0, 1.0, 5.0, activation_freq=0.9 if hot else 0.1),
            "e1": ExpertProfile("e1", 10.0, 1.0, 5.0, activation_freq=0.8 if hot else 0.1),
            "e2": ExpertProfile("e2", 10.0, 1.0, 5.0, activation_freq=0.2)}


def _state(profiles=None, requested=("e0",), resident=set(), **kw):
    return ScheduleState(
        profiles=profiles or _profiles(), resident=set(resident),
        gpu_capacity_mb=100.0, requested=requested, **kw)


def test_resident_stays_gpu():
    s = _state(resident={"e0"})
    actions = OverlapAwarePolicy(pcie=BandwidthResource(10.0)).decide(s, 0.0)
    assert any(a.kind == "execute_gpu" for a in actions)
    assert not any(a.kind == "execute_cpu" for a in actions)


def test_heavy_gpu_queue_sends_resident_to_cpu():
    # profile where CPU (1ms) is cheaper than GPU (5ms): pressure moves it to CPU
    profiles = {"e0": ExpertProfile("e0", 10.0, 5.0, 1.0)}
    s = _state(profiles=profiles, resident={"e0"}, gpu_queue_len=3, cpu_queue_len=0)
    actions = OverlapAwarePolicy(pcie=BandwidthResource(10.0), gpu_concurrency=1).decide(s, 0.0)
    # contention 3 > 1.5 and cpu cheaper (1 < 5*4) => execute_cpu
    assert any(a.kind == "execute_cpu" for a in actions)


def test_eft_chooses_cpu_when_cpu_wait_small():
    s = _state(cpu_wait_ms=1.0, pcie_wait_ms=50.0)
    actions = OverlapAwarePolicy(pcie=BandwidthResource(10.0)).decide(s, 0.0)
    # cpu_eft = 1 + 5 = 6; gpu_eft = 50 + 1 + 1 = 52 => CPU
    assert any(a.kind == "execute_cpu" for a in actions)


def test_eft_chooses_gpu_when_cpu_saturated():
    s = _state(cpu_wait_ms=100.0, cpu_queue_len=16, pcie_wait_ms=1.0)
    actions = OverlapAwarePolicy(pcie=BandwidthResource(10.0), cpu_concurrency=4).decide(s, 0.0)
    # cpu_eft = 100 + 5*5 = 125; gpu_eft = 1 + 1 + 1 = 3 => GPU load
    assert any(a.kind == "load" for a in actions)


def test_prefetch_emitted_when_slack():
    s = _state(requested=("e0",), profiles=_profiles(hot=True))
    actions = OverlapAwarePolicy(pcie=BandwidthResource(10.0), prefetch_n=2).decide(s, 0.0)
    assert any(a.kind == "prefetch" for a in actions)


def test_prefetch_gated_by_pcie_congestion():
    s = _state(requested=("e0",), profiles=_profiles(hot=True), pcie_queue_len=5)
    actions = OverlapAwarePolicy(pcie=BandwidthResource(10.0), max_pcie_queue=2).decide(s, 0.0)
    assert not any(a.kind == "prefetch" for a in actions)


def test_prefetch_gated_by_utilization():
    s = _state(requested=("e0",), profiles=_profiles(hot=True), pcie_utilization=0.9)
    actions = OverlapAwarePolicy(pcie=BandwidthResource(10.0), util_prefetch_threshold=0.7).decide(s, 0.0)
    assert not any(a.kind == "prefetch" for a in actions)


def test_hot_expert_loads_despite_single_step_cost():
    s = _state(profiles=_profiles(hot=True), cpu_wait_ms=0.0, pcie_wait_ms=0.0)
    s.record_load("e0", 5.0)  # benefit 5ms >= 1ms load_cost * 2
    actions = OverlapAwarePolicy(pcie=BandwidthResource(10.0)).decide(s, 0.0)
    assert any(a.kind == "load" for a in actions)


def test_prefetch_respects_capacity():
    s = ScheduleState(
        profiles=_profiles(hot=True), resident=set(), gpu_capacity_mb=15.0,
        requested=("e0",))
    actions = OverlapAwarePolicy(pcie=BandwidthResource(10.0), prefetch_n=2).decide(s, 0.0)
    # only 15MB free => at most one 10MB prefetch
    assert sum(1 for a in actions if a.kind == "prefetch") == 1
