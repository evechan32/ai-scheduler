"""Expert cost profiles and loading helpers."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertProfile:
    expert_id: str
    size_mb: float
    gpu_exec_ms: float
    cpu_exec_ms: float
    activation_freq: float = 0.0
    q_size_mb: float | None = None
    q_cpu_exec_ms: float | None = None

    def quantized_size_mb(self) -> float:
        return self.q_size_mb if self.q_size_mb is not None else self.size_mb

    def quantized_cpu_exec_ms(self) -> float:
        return self.q_cpu_exec_ms if self.q_cpu_exec_ms is not None else self.cpu_exec_ms


def profiles_from_dicts(rows: list[dict]) -> dict[str, ExpertProfile]:
    profiles: dict[str, ExpertProfile] = {}
    for row in rows:
        p = ExpertProfile(
            expert_id=row["expert_id"],
            size_mb=float(row["size_mb"]),
            gpu_exec_ms=float(row["gpu_exec_ms"]),
            cpu_exec_ms=float(row["cpu_exec_ms"]),
            activation_freq=float(row.get("activation_freq", 0.0)),
            q_size_mb=float(row["q_size_mb"]) if row.get("q_size_mb") is not None else None,
            q_cpu_exec_ms=float(row["q_cpu_exec_ms"]) if row.get("q_cpu_exec_ms") is not None else None,
        )
        profiles[p.expert_id] = p
    return profiles
