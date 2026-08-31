"""v8: KV-joint policy — KV pressure steers expert placement and prefetch."""
from moesim.scheduler.base import Action
from moesim.scheduler.policies.overlap import OverlapAwarePolicy
from moesim.scheduler.state import ScheduleState


class KVJointPolicy(OverlapAwarePolicy):
    def __init__(self, pcie=None, gpu_concurrency: int = 1, cpu_concurrency: int = 4,
                 prefetch_n: int = 2, max_pcie_queue: int = 2,
                 util_prefetch_threshold: float = 0.7,
                 kv_pressure_threshold: float = 0.8) -> None:
        super().__init__(pcie=pcie, gpu_concurrency=gpu_concurrency,
                         cpu_concurrency=cpu_concurrency, prefetch_n=prefetch_n,
                         max_pcie_queue=max_pcie_queue,
                         util_prefetch_threshold=util_prefetch_threshold)
        self.kv_pressure_threshold = kv_pressure_threshold

    def decide(self, state: ScheduleState, clock: float) -> list[Action]:
        if state.kv_pressure <= self.kv_pressure_threshold:
            return super().decide(state, clock)
        actions: list[Action] = []
        for eid in state.requested:
            if eid in state.resident:
                actions.append(Action(kind="execute_gpu", expert_ids=(eid,)))
            else:
                actions.append(Action(kind="execute_cpu", expert_ids=(eid,)))
        if state.kv_pressure >= 1.0:
            for eid in reversed(state.access_history):
                if eid not in state.requested:
                    actions.append(Action(kind="evict_kv", expert_ids=(eid,)))
                    break
        return actions
