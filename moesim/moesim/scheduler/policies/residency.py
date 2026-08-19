"""Residency-aware policy: queue-aware, migration-cost-aware, stability-first."""
from __future__ import annotations

from copy import deepcopy

from moesim.scheduler.base import Action, Scheduler
from moesim.scheduler.state import ScheduleState
from moesim.sim.resources import BandwidthResource


class ResidencyAwarePolicy(Scheduler):
    def __init__(self, pcie, gpu_concurrency: int = 1, cpu_concurrency: int = 4,
                 residency_bonus_factor: float = 2.0) -> None:
        self.pcie = pcie
        self.gpu_concurrency = gpu_concurrency
        self.cpu_concurrency = cpu_concurrency
        self.residency_bonus_factor = residency_bonus_factor

    def _gpu_contention(self, state: ScheduleState) -> float:
        if self.gpu_concurrency <= 0:
            return 0.0
        return state.gpu_queue_len / self.gpu_concurrency

    def _cpu_contention(self, state: ScheduleState) -> float:
        if self.cpu_concurrency <= 0:
            return 0.0
        return state.cpu_queue_len / self.cpu_concurrency

    def decide(self, state: ScheduleState, clock: float) -> list[Action]:
        actions: list[Action] = []
        for eid in state.requested:
            profile = state.profiles[eid]
            load_cost = self.pcie.transfer_time_ms(profile.size_mb)
            gpu_eff = profile.gpu_exec_ms * (1.0 + self._gpu_contention(state))
            cpu_eff = profile.cpu_exec_ms * (1.0 + self._cpu_contention(state))

            if eid in state.resident:
                # 稳定性：驻留专家留在 GPU，除非 GPU 排队严重且 CPU 实际更便宜
                if self._gpu_contention(state) > 1.5 and cpu_eff < gpu_eff:
                    actions.append(Action(kind="execute_cpu", expert_ids=(eid,)))
                else:
                    actions.append(Action(kind="execute_gpu", expert_ids=(eid,)))
                continue

            # 排队感知：GPU 严重排队且 CPU 几乎空闲 → 直接走 CPU（非驻留同样适用）
            if self._gpu_contention(state) > 1.5 and self._cpu_contention(state) < 0.25:
                actions.append(Action(kind="execute_cpu", expert_ids=(eid,)))
                continue

            # 驻留收益：累计价值超过 load_cost → load（长期收益）
            benefit = state.residency_benefit.get(eid, 0.0)
            if benefit >= load_cost * self.residency_bonus_factor:
                actions.append(Action(kind="load", expert_ids=(eid,)))
            elif cpu_eff < load_cost + gpu_eff:
                actions.append(Action(kind="execute_cpu", expert_ids=(eid,)))
            else:
                actions.append(Action(kind="load", expert_ids=(eid,)))
        return actions
