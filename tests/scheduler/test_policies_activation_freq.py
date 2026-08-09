from moesim.scheduler.base import apply_actions
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.activation_freq import ActivationFreqPolicy
from moesim.scheduler.state import ScheduleState


def _state(capacity_mb=20.0, resident=(), used=0.0, requested=("e2",)):
    profiles = {
        f"e{i}": ExpertProfile(
            expert_id=f"e{i}", size_mb=10.0, gpu_exec_ms=1.0, cpu_exec_ms=3.0,
            activation_freq=float(i),  # e3 hottest
        )
        for i in range(4)
    }
    return ScheduleState(
        profiles=profiles, resident=set(resident), gpu_capacity_mb=capacity_mb,
        used_gpu_mb=used, requested=tuple(requested),
    )


def test_freq_policy_evicts_coldest():
    st = _state(capacity_mb=20.0, resident=("e0", "e1"), used=20.0)
    policy = ActivationFreqPolicy()
    apply_actions(st, policy.decide(st, 0.0))  # request e2 -> load overflows 20MB
    assert "e2" in st.resident
    assert "e0" not in st.resident  # coldest (freq=0) evicted
    assert st.resident == {"e1", "e2"}
    assert st.used_gpu_mb == 20.0


def test_freq_policy_prefetches_hottest():
    st = _state(capacity_mb=40.0, resident=(), used=0.0)
    policy = ActivationFreqPolicy()
    apply_actions(st, policy.decide(st, 0.0))  # request e2, no overflow
    assert "e2" in st.resident
    assert "e3" in st.resident  # hottest non-resident prefetched into free 30MB
    assert st.used_gpu_mb == 20.0
