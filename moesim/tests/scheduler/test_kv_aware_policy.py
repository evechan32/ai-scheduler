from moesim.scheduler.cost_model import ExpertProfile
from moesim.scheduler.policies.kv_aware import KVWeightedPolicy
from moesim.scheduler.state import ScheduleState


def _profiles() -> dict[str, ExpertProfile]:
    return {
        f"e{i}": ExpertProfile(
            expert_id=f"e{i}", size_mb=10.0,
            gpu_exec_ms=1.0, cpu_exec_ms=5.0, activation_freq=1.0 - i * 0.1,
        )
        for i in range(4)
    }


def test_low_kv_pressure_behaves_like_cost_model():
    profiles = _profiles()
    state = ScheduleState(
        profiles=profiles, resident=set(), gpu_capacity_mb=1000.0,
        requested=("e0",), kv_gpu_capacity_mb=1000.0, kv_gpu_mb=10.0,
    )
    actions = KVWeightedPolicy().decide(state, 0.0)
    # GPU 空间充足、KV 压力低 → 加载到 GPU
    assert any(a.kind == "load" for a in actions)


def test_high_kv_pressure_prefers_cpu_and_evicts_kv():
    profiles = _profiles()
    state = ScheduleState(
        profiles=profiles, resident=set(), gpu_capacity_mb=1000.0,
        requested=("e0",), kv_gpu_capacity_mb=100.0, kv_gpu_mb=99.0,
        access_history=["e1", "e2", "e3"],
    )
    actions = KVWeightedPolicy().decide(state, 0.0)
    # KV 压力 0.99 → 专家放 CPU，且驱逐冷 KV
    assert any(a.kind == "execute_cpu" for a in actions)
    assert any(a.kind == "evict_kv" for a in actions)
