import pytest
from moesim.scheduler.base import Action, Scheduler
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.state import ScheduleState


def test_action_validation():
    with pytest.raises(ValueError):
        Action(kind="fly_to_moon")
    a = Action(kind="load", expert_ids=("e0",))
    assert a.target == "gpu"


def test_scheduler_abstract():
    s = Scheduler()
    st = ScheduleState(profiles={}, resident=set(), gpu_capacity_mb=100.0)
    with pytest.raises(NotImplementedError):
        s.decide(st, 0.0)
