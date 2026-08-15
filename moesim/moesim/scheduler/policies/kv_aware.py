"""KV-pressure-aware policy: extends CostModelPolicy with KV tier pressure."""
from __future__ import annotations

from copy import deepcopy

from moesim.scheduler.base import Action
from moesim.scheduler.policies.cost_model import CostModelPolicy
from moesim.scheduler.state import ScheduleState


class KVWeightedPolicy(CostModelPolicy):
    def __init__(self, pcie=None, cpu_concurrency: int = 1,
                 prefetch_n: int = 1, kv_pressure_threshold: float = 0.9) -> None:
        super().__init__(pcie=pcie, cpu_concurrency=cpu_concurrency,
                         prefetch_n=prefetch_n)
        self.kv_pressure_threshold = kv_pressure_threshold

    def _kv_pressure(self, state: ScheduleState) -> float:
        if state.kv_gpu_capacity_mb <= 0:
            return 0.0
        return state.kv_gpu_mb / state.kv_gpu_capacity_mb

    def decide(self, state: ScheduleState, clock: float) -> list[Action]:
        actions = super().decide(state, clock)
        pressure = self._kv_pressure(state)
        if pressure <= self.kv_pressure_threshold:
            return actions

        # KV pressure high: force requested non-resident experts to CPU and evict cold KV
        sim = deepcopy(state)
        requested = list(state.requested)
        result: list[Action] = []
        for a in actions:
            if a.kind == "load" and a.expert_ids[0] in requested:
                result.append(Action(kind="execute_cpu", expert_ids=a.expert_ids))
            else:
                result.append(a)
        # evict one cold KV entry (most recently touched, non-requested)
        for eid in reversed(state.access_history):
            if eid not in requested:
                result.append(Action(kind="evict_kv", expert_ids=(eid,)))
                break
        return result
