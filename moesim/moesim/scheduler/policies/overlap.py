"""v6: OverlapAwarePolicy — EFT placement with queueing + gated prefetch."""
from __future__ import annotations

from moesim.scheduler.base import Action, Scheduler
from moesim.scheduler.state import ScheduleState
from moesim.sim.resources import BandwidthResource


class OverlapAwarePolicy(Scheduler):
    def __init__(
        self,
        pcie: BandwidthResource | None = None,
        gpu_concurrency: int = 1,
        cpu_concurrency: int = 4,
        prefetch_n: int = 2,
        max_pcie_queue: int = 2,
        util_prefetch_threshold: float = 0.7,
        residency_bonus_factor: float = 2.0,
    ) -> None:
        self.pcie = pcie or BandwidthResource(bandwidth_gbps=10.0)
        self.gpu_concurrency = gpu_concurrency
        self.cpu_concurrency = cpu_concurrency
        self.prefetch_n = prefetch_n
        self.max_pcie_queue = max_pcie_queue
        self.util_prefetch_threshold = util_prefetch_threshold
        self.residency_bonus_factor = residency_bonus_factor

    def _gpu_contention(self, state: ScheduleState) -> float:
        if self.gpu_concurrency <= 0:
            return 0.0
        return state.gpu_queue_len / self.gpu_concurrency

    def _cpu_contention(self, state: ScheduleState) -> float:
        if self.cpu_concurrency <= 0:
            return 0.0
        return state.cpu_queue_len / self.cpu_concurrency

    def _cpu_eft(self, state: ScheduleState, profile) -> float:
        return state.cpu_wait_ms + profile.quantized_cpu_exec_ms() * (
            1.0 + self._cpu_contention(state)
        )

    def _gpu_eft(self, state: ScheduleState, profile) -> float:
        load_cost = self.pcie.transfer_time_ms(profile.size_mb)
        return (state.pcie_wait_ms + load_cost) + profile.gpu_exec_ms * (
            1.0 + self._gpu_contention(state)
        )

    def decide(self, state: ScheduleState, clock: float) -> list[Action]:
        actions: list[Action] = []
        for eid in state.requested:
            profile = state.profiles[eid]
            if eid in state.resident:
                gpu_cont = self._gpu_contention(state)
                cpu_eff = profile.cpu_exec_ms * (1.0 + self._cpu_contention(state))
                gpu_eff = profile.gpu_exec_ms * (1.0 + gpu_cont)
                if gpu_cont > 1.5 and cpu_eff < gpu_eff:
                    actions.append(Action(kind="execute_cpu", expert_ids=(eid,)))
                else:
                    actions.append(Action(kind="execute_gpu", expert_ids=(eid,)))
                continue

            load_cost = self.pcie.transfer_time_ms(profile.size_mb)
            benefit = state.residency_benefit.get(eid, 0.0)
            if benefit >= load_cost * self.residency_bonus_factor:
                actions.append(Action(kind="load", expert_ids=(eid,)))
            elif self._cpu_eft(state, profile) < self._gpu_eft(state, profile):
                actions.append(Action(kind="execute_cpu", expert_ids=(eid,)))
            else:
                actions.append(Action(kind="load", expert_ids=(eid,)))

        self._prefetch(state, actions)
        return actions

    def _prefetch(self, state: ScheduleState, actions: list[Action]) -> None:
        if self.prefetch_n <= 0:
            return
        if state.pcie_queue_len > self.max_pcie_queue:
            return
        if state.pcie_utilization > self.util_prefetch_threshold:
            return
        candidates = sorted(
            (p for p in state.profiles.values()
             if p.expert_id not in state.resident and p.expert_id not in state.requested),
            key=lambda p: p.activation_freq,
            reverse=True,
        )
        used = state.used_gpu_mb
        for profile in candidates[: self.prefetch_n]:
            if used + profile.size_mb <= state.gpu_capacity_mb + 1e-9:
                actions.append(Action(kind="prefetch", expert_ids=(profile.expert_id,)))
                used += profile.size_mb
