"""Activation-frequency-aware caching with hot-expert prefetch."""
from __future__ import annotations

from copy import deepcopy

from moesim.scheduler.base import Action, Scheduler
from moesim.scheduler.state import ScheduleState


class ActivationFreqPolicy(Scheduler):
    def decide(self, state: ScheduleState, clock: float) -> list[Action]:
        requested = list(state.requested)
        actions: list[Action] = []
        sim = deepcopy(state)
        for eid in requested:
            if eid not in sim.resident:
                actions.append(Action(kind="load", expert_ids=(eid,)))
                sim.resident.add(eid)
                sim.used_gpu_mb += sim.profiles[eid].size_mb
        for load in [a for a in actions if a.kind == "load"]:
            for eid in load.expert_ids:
                while sim.used_gpu_mb > sim.gpu_capacity_mb + 1e-9:
                    victim = self._coldest(sim, set(requested))
                    if victim is None:
                        break
                    sim.resident.remove(victim)
                    sim.used_gpu_mb -= sim.profiles[victim].size_mb
                    actions.append(Action(kind="unload", expert_ids=(victim,)))
        # prefetch hottest non-resident expert if budget allows
        non_resident = [
            p for p in sim.profiles.values()
            if p.expert_id not in sim.resident and p.expert_id not in requested
        ]
        if non_resident:
            hottest = max(non_resident, key=lambda p: p.activation_freq)
            if sim.used_gpu_mb + hottest.size_mb <= sim.gpu_capacity_mb + 1e-9:
                actions.append(Action(kind="load", expert_ids=(hottest.expert_id,)))
        return actions

    def _coldest(self, state: ScheduleState, protected: set[str]) -> str | None:
        residents = [p for p in state.profiles.values()
                     if p.expert_id in state.resident and p.expert_id not in protected]
        if not residents:
            return None
        return min(residents, key=lambda p: p.activation_freq).expert_id
