from moesim.scheduler.cost_model import ExpertProfile, profiles_from_dicts
from moesim.scheduler.state import ScheduleState


def test_profiles_from_dicts():
    rows = [
        {"expert_id": "e0", "size_mb": 340.0, "gpu_exec_ms": 1.2, "cpu_exec_ms": 4.5},
        {"expert_id": "e1", "size_mb": 340.0, "gpu_exec_ms": 1.1, "cpu_exec_ms": 4.2},
    ]
    profiles = profiles_from_dicts(rows)
    assert set(profiles) == {"e0", "e1"}
    assert profiles["e0"].size_mb == 340.0


def test_mark_access_hit_and_miss():
    profiles = profiles_from_dicts(
        [{"expert_id": "e0", "size_mb": 10.0, "gpu_exec_ms": 1.0, "cpu_exec_ms": 3.0}]
    )
    st = ScheduleState(profiles=profiles, resident={"e0"}, gpu_capacity_mb=100.0)
    assert st.mark_access("e0") is True
    assert st.mark_access("e1") is False
    assert st.cache_hits == 1 and st.cache_misses == 1
    assert st.access_history == ["e0", "e1"]
