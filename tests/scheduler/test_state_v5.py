from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.state import ScheduleState


def _state():
    profiles = {f"e{i}": ExpertProfile(f"e{i}", 10.0, 1.0, 5.0, activation_freq=0.5) for i in range(3)}
    return ScheduleState(profiles=profiles, resident=set(), gpu_capacity_mb=100.0)


def test_v5_defaults():
    s = _state()
    assert s.gpu_queue_len == 0
    assert s.cpu_queue_len == 0
    assert s.residency_benefit == {}
    assert s.migration_cost_ms == 0.0


def test_record_load_updates_benefit():
    s = _state()
    s.record_load("e0", 79.0)
    assert s.residency_benefit["e0"] == 79.0
    # 再次 load 累计（多次命中价值叠加）
    s.record_load("e0", 79.0)
    assert s.residency_benefit["e0"] == 158.0
