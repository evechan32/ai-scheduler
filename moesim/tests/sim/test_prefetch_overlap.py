"""v6: prefetch overlap in the simulator - transfers hidden behind compute."""
from moesim.scheduler.base import Action, Scheduler
from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.overlap import OverlapAwarePolicy
from moesim.sim.moe_adapter import MoESimulation
from moesim.sim.resources import BandwidthResource, ComputeResource


def _profiles():
    return {
        "e0": ExpertProfile("e0", 10.0, 3.0, 6.0, activation_freq=0.1),
        "e1": ExpertProfile("e1", 10.0, 3.0, 6.0, activation_freq=0.5),
        "e2": ExpertProfile("e2", 10.0, 3.0, 6.0, activation_freq=0.9),
    }


def _sim(prefetch_n=1, scheduler=None):
    pcie = BandwidthResource(bandwidth_gbps=10.0)  # 10MB => 1ms
    policy = scheduler or OverlapAwarePolicy(pcie=pcie, prefetch_n=prefetch_n)
    return MoESimulation(
        scheduler=policy,
        profiles=_profiles(),
        gpu_capacity_mb=100.0,
        pcie=pcie,
        gpu=ComputeResource(concurrency=1, per_unit_ms=1.0),
        cpu=ComputeResource(concurrency=4, per_unit_ms=1.0),
    )


def test_prefetch_overlaps_compute():
    sim = _sim(prefetch_n=1)
    m = sim.run([["e0"], ["e2"]])
    assert m.total_time_ms == 7.0
    assert m.prefetch_count == 2
    assert m.hidden_transfer_ms == 2.0
    assert m.total_transfer_ms == 3.0
    assert m.overlap_ratio() == 2 / 3


def test_prefetch_saves_next_step_pcie():
    no_pf = _sim(prefetch_n=0).run([["e0"], ["e2"]]).total_time_ms
    with_pf = _sim(prefetch_n=1).run([["e0"], ["e2"]]).total_time_ms
    assert no_pf == 8.0
    assert with_pf == 7.0
    assert with_pf < no_pf


def _slow_pcie_profiles():
    return {
        "e0": ExpertProfile("e0", 10.0, 0.5, 20.0, activation_freq=0.1),
        "e1": ExpertProfile("e1", 10.0, 0.5, 20.0, activation_freq=0.5),
        "e2": ExpertProfile("e2", 10.0, 0.5, 20.0, activation_freq=0.9),
    }


def test_in_flight_prefetch_waits_but_feedback_tracks_backlog():
    pcie = BandwidthResource(bandwidth_gbps=10.0)

    class Probe(OverlapAwarePolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.seen = []

        def decide(self, state, clock):
            self.seen.append(state.pcie_queue_len)
            return super().decide(state, clock)

    probe = Probe(pcie=pcie, prefetch_n=2)
    sim = MoESimulation(scheduler=probe, profiles=_slow_pcie_profiles(),
                        gpu_capacity_mb=100.0, pcie=pcie,
                        gpu=ComputeResource(1, 1.0), cpu=ComputeResource(4, 1.0))
    m = sim.run([["e0"], ["e1"]])
    assert m.total_time_ms == 3.5
    assert probe.seen[0] == 0
    assert probe.seen[1] == 2


class PrefetchOtherThenLoadStub(Scheduler):
    def __init__(self):
        self.step = 0

    def decide(self, state, clock):
        self.step += 1
        if self.step == 1:
            return [Action(kind="prefetch", expert_ids=("e2",))]
        return [Action(kind="load", expert_ids=("e2",)),
                Action(kind="execute_gpu", expert_ids=("e2",))]


def test_inflight_load_does_not_double_book_pcie():
    pcie = BandwidthResource(bandwidth_gbps=2.0)  # 10MB => 5ms
    sim = MoESimulation(scheduler=PrefetchOtherThenLoadStub(), profiles=_profiles(),
                        gpu_capacity_mb=100.0, pcie=pcie,
                        gpu=ComputeResource(1, 1.0))
    m = sim.run([["e1"], ["e2"]])
    assert m.total_transfer_ms == 5.0
    assert m.total_time_ms == 8.0


def test_deterministic_across_runs():
    a = _sim(prefetch_n=2).run([["e0"], ["e2"], ["e1"], ["e0"]]).total_time_ms
    b = _sim(prefetch_n=2).run([["e0"], ["e2"], ["e1"], ["e0"]]).total_time_ms
    assert a == b
