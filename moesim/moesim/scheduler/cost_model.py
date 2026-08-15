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


def profiles_from_dicts(rows: list[dict]) -> dict[str, ExpertProfile]:
    profiles: dict[str, ExpertProfile] = {}
    for row in rows:
        p = ExpertProfile(
            expert_id=row["expert_id"],
            size_mb=float(row["size_mb"]),
            gpu_exec_ms=float(row["gpu_exec_ms"]),
            cpu_exec_ms=float(row["cpu_exec_ms"]),
            activation_freq=float(row.get("activation_freq", 0.0)),
        )
        profiles[p.expert_id] = p
    return profiles
