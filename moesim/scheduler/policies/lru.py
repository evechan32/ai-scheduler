"""Least-recently-used expert caching baseline."""
from __future__ import annotations

from copy import deepcopy

from moesim.scheduler.base import Action, Scheduler
from moesim.scheduler.state import ScheduleState


class LRUPolicy(Scheduler):
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
                    victim = self._lru_victim(sim, set(requested))
                    if victim is None:
                        break
                    sim.resident.remove(victim)
                    sim.used_gpu_mb -= sim.profiles[victim].size_mb
                    actions.append(Action(kind="unload", expert_ids=(victim,)))
        return actions

    def _lru_victim(self, state: ScheduleState, protected: set[str]) -> str | None:
        for eid in reversed(state.access_history):
            if eid in state.resident and eid not in protected:
                return eid
        return None
