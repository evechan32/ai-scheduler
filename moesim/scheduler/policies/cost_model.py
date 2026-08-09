"""Heterogeneous-compute-aware policy: price CPU execution vs fetch+GPU execution."""
from __future__ import annotations

from copy import deepcopy

from moesim.scheduler.base import Action, Scheduler
from moesim.scheduler.state import ScheduleState
from moesim.sim.resources import BandwidthResource


class CostModelPolicy(Scheduler):
    def __init__(
        self,
        pcie: BandwidthResource | None = None,
        cpu_concurrency: int = 1,
        prefetch_n: int = 1,
    ) -> None:
        self.pcie = pcie or BandwidthResource(bandwidth_gbps=10.0)
        self.cpu_concurrency = cpu_concurrency
        self.prefetch_n = prefetch_n

    def decide(self, state: ScheduleState, clock: float) -> list[Action]:
        requested = list(state.requested)
        actions: list[Action] = []
        sim = deepcopy(state)

        for eid in requested:
            profile = sim.profiles[eid]
            if eid in sim.resident:
                actions.append(Action(kind="execute_gpu", expert_ids=(eid,)))
                continue
            load_cost = self.pcie.transfer_time_ms(profile.size_mb)
            if profile.cpu_exec_ms <= load_cost + profile.gpu_exec_ms:
                actions.append(Action(kind="execute_cpu", expert_ids=(eid,)))
            else:
                actions.append(Action(kind="load", expert_ids=(eid,)))
                sim.resident.add(eid)
                sim.used_gpu_mb += profile.size_mb

        # capacity discipline: evict LRU victims for the loads we decided
        for load in [a for a in actions if a.kind == "load"]:
            for eid in load.expert_ids:
                while sim.used_gpu_mb > sim.gpu_capacity_mb + 1e-9:
                    victim = self._lru_victim(sim, set(requested))
                    if victim is None:
                        break
                    sim.resident.remove(victim)
                    sim.used_gpu_mb -= sim.profiles[victim].size_mb
                    actions.append(Action(kind="unload", expert_ids=(victim,)))

        # prefetch top-N hottest non-resident experts
        if self.prefetch_n > 0:
            candidates = sorted(
                (p for p in sim.profiles.values()
                 if p.expert_id not in sim.resident and p.expert_id not in requested),
                key=lambda p: p.activation_freq,
                reverse=True,
            )
            for profile in candidates[: self.prefetch_n]:
                if sim.used_gpu_mb + profile.size_mb <= sim.gpu_capacity_mb + 1e-9:
                    actions.append(Action(kind="load", expert_ids=(profile.expert_id,)))
                    sim.resident.add(profile.expert_id)
                    sim.used_gpu_mb += profile.size_mb
                else:
                    break
        return actions

    def _lru_victim(self, state: ScheduleState, protected: set[str]) -> str | None:
        for eid in reversed(state.access_history):
            if eid in state.resident and eid not in protected:
                return eid
        return None
