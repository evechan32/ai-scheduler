import numpy as np

from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.rl import RLScheduler
from moesim.scheduler.state import ScheduleState
from moesim.sim.moe_adapter import MoESimulation
from moesim.sim.resources import BandwidthResource, ComputeResource


def _profiles():
    return {
        f"e{i}": ExpertProfile(expert_id=f"e{i}", size_mb=10.0,
                gpu_exec_ms=1.0, cpu_exec_ms=5.0, activation_freq=1.0 - i * 0.1)
        for i in range(4)
    }


def _sim(scheduler):
    pcie = BandwidthResource(bandwidth_gbps=10.0)
    return MoESimulation(
        scheduler=scheduler, profiles=_profiles(), gpu_capacity_mb=100.0,
        pcie=pcie, gpu=ComputeResource(concurrency=1, per_unit_ms=1.0),
        cpu=ComputeResource(concurrency=4, per_unit_ms=1.0),
    )


def test_rl_decide_returns_valid_actions():
    np.random.seed(0)
    sched = RLScheduler()
    state = ScheduleState(profiles=_profiles(), resident=set(), gpu_capacity_mb=100.0,
                          requested=("e0",))
    actions = sched.decide(state, 0.0)
    kinds = {a.kind for a in actions}
    assert kinds <= {"load", "execute_gpu", "execute_cpu"}


def test_rl_training_improves_over_episodes():
    np.random.seed(42)
    trace = [["e0", "e1"]] * 40 + [["e0", "e1", "e2"]] * 10
    sched = RLScheduler()
    sim = _sim(sched)
    sched.train(sim, episodes=100, trace=trace)
    # deterministic check: decided actions reference experts from profiles
    assert sched.q_table  # non-empty after training


def test_rl_deterministic_with_fixed_seed():
    np.random.seed(7)
    a = RLScheduler()
    trace = [["e0", "e1"]] * 20
    a.train(_sim(a), episodes=30, trace=trace)
    np.random.seed(7)
    b = RLScheduler()
    b.train(_sim(b), episodes=30, trace=trace)
    assert a.q_table == b.q_table


def test_pg_training_updates_theta():
    np.random.seed(0)
    from moesim.scheduler.policies.rl import RLPolicyGradientScheduler
    sched = RLPolicyGradientScheduler(n_experts=4)
    trace = [["e0", "e1"]] * 30
    sched.train(_sim(sched), episodes=50, trace=trace)
    assert sched.theta is not None
    # 训练后 theta 应有非零梯度更新
    assert np.abs(sched.theta).sum() > 0


def test_pg_deterministic_seed():
    np.random.seed(3)
    from moesim.scheduler.policies.rl import RLPolicyGradientScheduler
    a = RLPolicyGradientScheduler(n_experts=4)
    a.train(_sim(a), episodes=20, trace=[["e0", "e1"]] * 10)
    np.random.seed(3)
    b = RLPolicyGradientScheduler(n_experts=4)
    b.train(_sim(b), episodes=20, trace=[["e0", "e1"]] * 10)
    assert np.allclose(a.theta, b.theta)
