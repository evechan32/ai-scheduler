"""v6: prefetch action and resource feedback fields in state."""
from moesim.scheduler.base import Action, apply_actions
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.state import ScheduleState


def _state():
    profiles = {f"e{i}": ExpertProfile(f"e{i}", 10.0, 1.0, 5.0) for i in range(3)}
    return ScheduleState(profiles=profiles, resident=set(), gpu_capacity_mb=100.0)


def test_prefetch_action_is_valid():
    a = Action(kind="prefetch", expert_ids=("e0",))
    assert a.kind == "prefetch"


def test_prefetch_applies_like_load():
    s = _state()
    apply_actions(s, [Action(kind="prefetch", expert_ids=("e0", "e1"))])
    assert s.resident == {"e0", "e1"}
    assert s.used_gpu_mb == 20.0


def test_prefetch_respects_capacity():
    s = ScheduleState(profiles=_state().profiles, resident={"e2"}, gpu_capacity_mb=15.0)
    s.used_gpu_mb = 10.0
    try:
        apply_actions(s, [Action(kind="prefetch", expert_ids=("e0",))])
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_v6_state_defaults():
    s = _state()
    assert s.pcie_queue_len == 0
    assert s.pcie_utilization == 0.0
    assert s.gpu_utilization == 0.0
    assert s.cpu_utilization == 0.0
    assert s.gpu_wait_ms == 0.0
    assert s.cpu_wait_ms == 0.0
    assert s.pcie_wait_ms == 0.0
    assert s.pending_loads == {}
