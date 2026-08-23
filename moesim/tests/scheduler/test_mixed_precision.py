"""v7: mixed-precision expert placement — quantized CPU exec + low-precision prefetch.

Grounding: HOBBIT (arXiv:2411.01433, mixed-precision expert offloading),
QuantMoE-Bench (arXiv:2406.08155, frequency-aware bit allocation),
ktransformers INT4 CPU gemm (arXiv:2410.06410).
"""
import pytest

from moesim.scheduler.base import Action
from moesim.scheduler.cost_model import ExpertProfile, profiles_from_dicts
from moesim.scheduler.policies.overlap import OverlapAwarePolicy
from moesim.scheduler.state import ScheduleState
from moesim.sim.moe_adapter import MoESimulation
from moesim.sim.resources import BandwidthResource, ComputeResource


def _qp(expert_id, size, gpu, cpu, q_size=None, q_cpu=None, freq=0.1):
    return ExpertProfile(expert_id, size, gpu, cpu, activation_freq=freq,
                         q_size_mb=q_size, q_cpu_exec_ms=q_cpu)


def _state(profiles, requested=("e0",), **kw):
    return ScheduleState(profiles=profiles, resident=set(), gpu_capacity_mb=1000.0,
                         requested=requested, **kw)


def test_quantized_cpu_changes_placement_decision():
    pcie = BandwidthResource(bandwidth_gbps=100.0)  # 100MB => 1ms load
    plain = _qp("e0", 100.0, 1.0, 20.0)
    quant = _qp("e0", 100.0, 1.0, 20.0, q_size=25.0, q_cpu=1.5)
    # without quantized CPU time: gpu_eft = 1+1 = 2 < cpu 20 => load
    s = _state({"e0": plain})
    assert any(a.kind == "load" for a in OverlapAwarePolicy(pcie=pcie).decide(s, 0.0))
    # with quantized CPU time (1.5 < 2): => execute_cpu
    s2 = _state({"e0": quant})
    actions = OverlapAwarePolicy(pcie=pcie).decide(s2, 0.0)
    assert any(a.kind == "execute_cpu" for a in actions)


def test_plain_profile_falls_back_to_full_precision():
    p = _qp("e0", 100.0, 1.0, 20.0)
    assert p.quantized_size_mb() == 100.0
    assert p.quantized_cpu_exec_ms() == 20.0


def test_quantized_profile_fields():
    p = _qp("e0", 100.0, 1.0, 20.0, q_size=25.0, q_cpu=2.0)
    assert p.quantized_size_mb() == 25.0
    assert p.quantized_cpu_exec_ms() == 2.0


def test_profiles_from_dicts_parses_quantized_fields():
    rows = [{"expert_id": "e0", "size_mb": 100.0, "gpu_exec_ms": 1.0,
             "cpu_exec_ms": 20.0, "q_size_mb": 25.0, "q_cpu_exec_ms": 2.0}]
    p = profiles_from_dicts(rows)["e0"]
    assert p.quantized_size_mb() == 25.0
    assert p.quantized_cpu_exec_ms() == 2.0


def test_cpu_execution_uses_quantized_time():
    pcie = BandwidthResource(bandwidth_gbps=100.0)

    class CpuScheduler:
        def decide(self, state, clock):
            return [Action(kind="execute_cpu", expert_ids=(eid,))
                    for eid in state.requested]

    profiles = {"e0": _qp("e0", 100.0, 1.0, 20.0, q_size=25.0, q_cpu=2.0, freq=0.9),
                "e1": _qp("e1", 100.0, 1.0, 20.0)}
    sim = MoESimulation(scheduler=CpuScheduler(),
                        profiles=profiles, gpu_capacity_mb=1000.0, pcie=pcie,
                        gpu=ComputeResource(1, 1.0),
                        cpu=ComputeResource(4, 1.0))
    m = sim.run([["e0"], ["e1"]])
    # e0 executes CPU with quantized 2ms; e1 (no quantized variant) 20ms
    assert m.total_time_ms == 22.0


def test_prefetch_transfer_uses_quantized_size():
    pcie = BandwidthResource(bandwidth_gbps=100.0)  # 25MB=>0.25ms, 100MB=>1ms
    plain = {"e0": _qp("e0", 100.0, 0.5, 50.0),
             "e1": _qp("e1", 100.0, 0.5, 50.0, q_size=25.0, q_cpu=2.0, freq=0.9)}
    sim_plain = MoESimulation(scheduler=OverlapAwarePolicy(pcie=pcie, prefetch_n=1),
                              profiles={"e0": _qp("e0", 100.0, 0.5, 50.0),
                                        "e1": _qp("e1", 100.0, 0.5, 50.0,
                                                  freq=0.9)},
                              gpu_capacity_mb=1000.0, pcie=pcie,
                              gpu=ComputeResource(1, 1.0),
                              cpu=ComputeResource(4, 1.0))
    m_plain = sim_plain.run([["e0"], ["e1"]])
    sim_quant = MoESimulation(scheduler=OverlapAwarePolicy(pcie=pcie, prefetch_n=1),
                              profiles=plain, gpu_capacity_mb=1000.0, pcie=pcie,
                              gpu=ComputeResource(1, 1.0),
                              cpu=ComputeResource(4, 1.0))
    m_quant = sim_quant.run([["e0"], ["e1"]])
    # step1: e0 loads at full precision (1ms); e1 prefetched — 100MB (1ms) plain
    # vs 25MB (0.25ms) quantized
    assert m_plain.total_transfer_ms == 2.0
    assert m_quant.total_transfer_ms == 1.25
    assert m_quant.prefetch_count == 1


def test_load_still_uses_full_precision_size():
    pcie = BandwidthResource(bandwidth_gbps=100.0)
    profiles = {"e0": _qp("e0", 100.0, 0.5, 50.0, q_size=25.0, q_cpu=2.0)}
    sim = MoESimulation(scheduler=OverlapAwarePolicy(pcie=pcie),
                        profiles=profiles, gpu_capacity_mb=1000.0, pcie=pcie,
                        gpu=ComputeResource(1, 1.0),
                        cpu=ComputeResource(4, 1.0))
    sim.run([["e0"]])
    # e0 has no residency benefit yet -> EFT: cpu 2 (quantized) vs gpu 1.5
    # => load at full precision 100MB => 1ms transfer + 0.5ms exec
    assert sim._state.resident == {"e0"}
    assert sim._metrics.total_transfer_ms == 1.0
