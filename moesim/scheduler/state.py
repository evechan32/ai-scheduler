"""Scheduler state — serializable, deterministic."""
from __future__ import annotations

from dataclasses import dataclass, field

from moesim.scheduler.cost_model import ExpertProfile


@dataclass
class ScheduleState:
    profiles: dict[str, ExpertProfile]
    resident: set[str]
    gpu_capacity_mb: float
    used_gpu_mb: float = 0.0
    requested: tuple[str, ...] = ()  # experts requested in the current step
    access_history: list[str] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0

    def mark_access(self, expert_id: str) -> bool:
        self.access_history.append(expert_id)
        hit = expert_id in self.resident
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        return hit
