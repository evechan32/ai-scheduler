"""v8: KVJointPolicy behavior under KV pressure."""
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.kv_joint import KVJointPolicy
from moesim.scheduler.policies.overlap import OverlapAwarePolicy
from moesim.scheduler.state import ScheduleState
from moesim.sim.resources import BandwidthResource


def _profiles(hot=False):
    return {"e0": ExpertProfile("e0", 10.0, 1.0, 5.0, activation_freq=0.9 if hot else 0.1),
            "e1": ExpertProfile("e1", 10.0, 1.0, 5.0, activation_freq=0.8 if hot else 0.1),
            "e2": ExpertProfile("e2", 10.0, 1.0, 5.0, activation_freq=0.2)}


def _state(requested=("e0",), resident=set(), pressure=0.0, **kw):
    return ScheduleState(profiles=_profiles(hot=True), resident=set(resident),
                         gpu_capacity_mb=100.0, requested=requested,
                         kv_pressure=pressure, **kw)


def test_low_pressure_behaves_like_overlap():
    s = _state(pressure=0.2)
    kv_actions = KVJointPolicy(pcie=BandwidthResource(10.0), prefetch_n=1).decide(s, 0.0)
    ov_actions = OverlapAwarePolicy(pcie=BandwidthResource(10.0), prefetch_n=1).decide(s, 0.0)
    assert [a.kind for a in kv_actions] == [a.kind for a in ov_actions]


def test_high_pressure_forces_nonresident_to_cpu():
    s = _state(pressure=0.95, resident=set())
    actions = KVJointPolicy(pcie=BandwidthResource(10.0)).decide(s, 0.0)
    assert any(a.kind == "execute_cpu" for a in actions)
    assert not any(a.kind == "load" for a in actions)
    assert not any(a.kind == "prefetch" for a in actions)


def test_high_pressure_keeps_resident_on_gpu():
    s = _state(pressure=0.95, resident={"e0"})
    actions = KVJointPolicy(pcie=BandwidthResource(10.0)).decide(s, 0.0)
    assert any(a.kind == "execute_gpu" for a in actions)
    assert not any(a.kind == "execute_cpu" for a in actions)


def test_full_pressure_evicts_cold_kv():
    s = _state(pressure=1.0, resident={"e0"})
    s.access_history = ["e1", "e2", "e0"]
    actions = KVJointPolicy(pcie=BandwidthResource(10.0)).decide(s, 0.0)
    assert any(a.kind == "evict_kv" for a in actions)
    evicted = [e for a in actions if a.kind == "evict_kv" for e in a.expert_ids]
    assert evicted and evicted[0] != "e0"  # requested expert protected


def test_high_pressure_no_prefetch_even_when_hot():
    s = _state(pressure=0.95, resident={"e0"})
    actions = KVJointPolicy(pcie=BandwidthResource(10.0), prefetch_n=5).decide(s, 0.0)
    assert not any(a.kind == "prefetch" for a in actions)
