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
                 disk_budget_mb: float = 0.0, dram_capacity_mb: float = 0.0,
                 prefetch_disk_n: int = 0) -> None:
        super().__init__(pcie=pcie, cpu_concurrency=cpu_concurrency,
                         prefetch_n=prefetch_n)
        self.disk_budget_mb = disk_budget_mb
        self.dram_capacity_mb = dram_capacity_mb
        self.prefetch_disk_n = prefetch_disk_n

    def _disk_candidates(self, state: ScheduleState) -> list[str]:
        # non-resident experts ranked coldest-first
        ranked = sorted(
            (p for p in state.profiles.values() if p.expert_id not in state.resident),
            key=lambda p: p.activation_freq,
        )
        if not ranked:
            return []

        # Passive: DRAM capacity constraint — experts exceeding DRAM must go to disk.
        required = 0.0
        if self.dram_capacity_mb > 0.0:
            total_nonresident = sum(p.size_mb for p in ranked)
            required = max(0.0, total_nonresident - self.dram_capacity_mb)

        # Active: disk_budget — demote even more to save DRAM (optional).
        budget = max(required, self.disk_budget_mb)
        candidates = []
        for p in ranked:
            if budget < p.size_mb:
                break
            candidates.append(p.expert_id)
            budget -= p.size_mb
        return candidates

    def decide(self, state: ScheduleState, clock: float) -> list[Action]:
        actions = super().decide(state, clock)
        demoted = set(self._disk_candidates(state))
        for eid in demoted:
            actions.append(Action(kind="demote_to_disk", expert_ids=(eid,)))
        # Predicted disk prefetch: hottest disk experts are most likely to be
        # activated next — prefetch them off SSD ahead of time.
        if self.prefetch_disk_n > 0:
            hot_disk = sorted(
                (p.expert_id for p in state.profiles.values()
                 if p.expert_id in demoted),
                key=lambda eid: -state.profiles[eid].activation_freq,
            )
            for eid in hot_disk[: self.prefetch_disk_n]:
                actions.append(Action(kind="prefetch_from_disk", expert_ids=(eid,)))
        return actions
