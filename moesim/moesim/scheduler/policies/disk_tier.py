"""2.0: DiskTierPolicy — three-tier expert placement (GPU/DRAM/disk)."""
from __future__ import annotations

from moesim.scheduler.base import Action
from moesim.scheduler.policies.cost_model import CostModelPolicy
from moesim.scheduler.state import ScheduleState


class DiskTierPolicy(CostModelPolicy):
    """Hot experts -> GPU, warm -> CPU (DRAM), coldest -> disk (SSD).

    Demotes the least-activated experts to disk to save DRAM, trading a slow
    disk read (SSD bandwidth) on the rare activation of a cold expert — the
    FlexGen / MoE-Infinity three-tier placement.
    """

    def __init__(self, pcie=None, cpu_concurrency: int = 1, prefetch_n: int = 1,
                 disk_budget_mb: float = 0.0) -> None:
        super().__init__(pcie=pcie, cpu_concurrency=cpu_concurrency,
                         prefetch_n=prefetch_n)
        self.disk_budget_mb = disk_budget_mb

    def _disk_candidates(self, state: ScheduleState) -> list[str]:
        if self.disk_budget_mb <= 0.0:
            return []
        # coldest (lowest activation_freq) experts first
        ranked = sorted(
            (p for p in state.profiles.values() if p.expert_id not in state.resident),
            key=lambda p: p.activation_freq,
        )
        budget = self.disk_budget_mb
        candidates = []
        for p in ranked:
            if budget < p.size_mb:
                break
            candidates.append(p.expert_id)
            budget -= p.size_mb
        return candidates

    def decide(self, state: ScheduleState, clock: float) -> list[Action]:
        actions = super().decide(state, clock)
        for eid in self._disk_candidates(state):
            actions.append(Action(kind="demote_to_disk", expert_ids=(eid,)))
        return actions
