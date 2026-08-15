from moesim.scheduler.base import Action, apply_actions
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.state import ScheduleState


def _state() -> ScheduleState:
    profiles = {f"e{i}": ExpertProfile(expert_id=f"e{i}", size_mb=10.0,
                 gpu_exec_ms=1.0, cpu_exec_ms=5.0) for i in range(3)}
    return ScheduleState(profiles=profiles, resident=set(),
                         gpu_capacity_mb=100.0, kv_gpu_mb=30.0, kv_host_mb=0.0)


def test_evict_kv_moves_gpu_to_host():
    s = _state()
    apply_actions(s, [Action(kind="evict_kv", expert_ids=("e0",))])
    assert s.kv_gpu_mb == 20.0
    assert s.kv_host_mb == 10.0


def test_fetch_kv_moves_host_to_gpu():
    s = _state()
    s.kv_gpu_mb = 0.0
    s.kv_host_mb = 30.0
    apply_actions(s, [Action(kind="fetch_kv", expert_ids=("e0",))])
    assert s.kv_gpu_mb == 10.0
    assert s.kv_host_mb == 20.0
