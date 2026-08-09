from moesim.scheduler.base import Action, apply_actions
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.cost_model import CostModelPolicy
from moesim.scheduler.state import ScheduleState
from moesim.sim.resources import BandwidthResource


def _state(capacity_mb=20.0, resident=(), used=0.0):
    profiles = {
        f"e{i}": ExpertProfile(
            expert_id=f"e{i}", size_mb=10.0, gpu_exec_ms=1.0, cpu_exec_ms=3.0,
            activation_freq=float(i),
        )
        for i in range(4)
    }
    return ScheduleState(
        profiles=profiles, resident=set(resident), gpu_capacity_mb=capacity_mb,
        used_gpu_mb=used, requested=("e2",),
    )


def test_resident_expert_gets_execute_gpu():
    st = _state(resident=("e2",), used=10.0)
    policy = CostModelPolicy()
    actions = policy.decide(st, 0.0)
    assert Action(kind="execute_gpu", expert_ids=("e2",)) in actions


def test_cpu_cheaper_than_load():
    # 10MB at 1 GB/s => 10ms load; cpu 3ms < 10+1 => execute_cpu
    pcie = BandwidthResource(bandwidth_gbps=1.0)
    st = _state(resident=(), used=0.0)  # request e2, not resident
    policy = CostModelPolicy(pcie=pcie, prefetch_n=0)
    actions = policy.decide(st, 0.0)
    assert Action(kind="execute_cpu", expert_ids=("e2",)) in actions


def test_load_when_cpu_slower_than_fetch_and_compute():
    # 10MB at 10 GB/s => 1ms load; cpu 3ms > 1+1 => load
    pcie = BandwidthResource(bandwidth_gbps=10.0)
    st = _state(resident=(), used=0.0)
    policy = CostModelPolicy(pcie=pcie, prefetch_n=0)
    actions = policy.decide(st, 0.0)
    assert Action(kind="load", expert_ids=("e2",)) in actions


def test_prefetch_hottest():
    st = _state(resident=(), used=0.0, capacity_mb=40.0)
    policy = CostModelPolicy(pcie=BandwidthResource(bandwidth_gbps=10.0), prefetch_n=1)
    actions = policy.decide(st, 0.0)
    assert Action(kind="load", expert_ids=("e3",)) in actions  # hottest non-resident


def test_load_evicts_lru_when_over_capacity():
    st = _state(resident=("e0", "e1"), used=20.0)  # request e2, needs 10MB, no room
    st.access_history = ["e1", "e0"]  # e1 most recent, e0 least
    policy = CostModelPolicy(pcie=BandwidthResource(bandwidth_gbps=10.0), prefetch_n=0)
    actions = policy.decide(st, 0.0)
    assert Action(kind="load", expert_ids=("e2",)) in actions
    assert Action(kind="unload", expert_ids=("e0",)) in actions  # e0 evicted (LRU)
