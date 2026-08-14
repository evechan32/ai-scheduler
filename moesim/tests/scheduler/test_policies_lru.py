from moesim.scheduler.base import Action, apply_actions
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.lru import LRUPolicy
from moesim.scheduler.state import ScheduleState


def _state(capacity_mb=20.0, resident=(), used=0.0, requested=("e2",)):
    profiles = {
        f"e{i}": ExpertProfile(expert_id=f"e{i}", size_mb=10.0, gpu_exec_ms=1.0, cpu_exec_ms=3.0)
        for i in range(4)
    }
    return ScheduleState(
        profiles=profiles, resident=set(resident), gpu_capacity_mb=capacity_mb,
        used_gpu_mb=used, requested=tuple(requested),
    )


def test_lru_loads_requested_and_evicts_lru():
    st = _state(resident=("e0", "e1"), used=20.0)
    st.access_history = ["e1", "e0"]  # e1 most recent, e0 least
    policy = LRUPolicy()
    actions = policy.decide(st, 0.0)  # requested=("e2",): not resident, needs 10MB
    apply_actions(st, actions)
    assert "e2" in st.resident
    assert "e0" not in st.resident  # e0 evicted (LRU)
    assert st.used_gpu_mb == 20.0


def test_lru_no_eviction_when_space_available():
    st = _state(resident=("e0",), used=10.0)
    policy = LRUPolicy()
    apply_actions(st, policy.decide(st, 0.0))  # requested=("e2",): fits, no eviction
    assert st.resident == {"e0", "e2"}
    assert st.used_gpu_mb == 20.0
