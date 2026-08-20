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
    kv_gpu_mb: float = 0.0
    kv_host_mb: float = 0.0
    kv_gpu_capacity_mb: float = 0.0
    gpu_resident: dict[int, set[str]] = field(default_factory=lambda: {0: set()})
    pcie_queue_len: int = 0
    pcie_utilization: float = 0.0
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    gpu_wait_ms: float = 0.0
    cpu_wait_ms: float = 0.0
    pcie_wait_ms: float = 0.0
    pending_loads: dict[str, float] = field(default_factory=dict)
    gpu_queue_len: int = 0
    cpu_queue_len: int = 0
    residency_benefit: dict[str, float] = field(default_factory=dict)
    migration_cost_ms: float = 0.0

    def __post_init__(self) -> None:
        # Backward-compatible one-time sync: when per-GPU residency is provided,
        # reflect GPU 0's set into the legacy `resident` field. This is not a
        # live view (mutations after construction are not propagated).
        if self.gpu_resident.get(0):
            self.resident = set(self.gpu_resident[0])

    def mark_access(self, expert_id: str) -> bool:
        self.access_history.append(expert_id)
        hit = expert_id in self.resident
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        return hit

    def record_load(self, expert_id: str, cost_ms: float) -> None:
        self.residency_benefit[expert_id] = (
            self.residency_benefit.get(expert_id, 0.0) + cost_ms
        )
