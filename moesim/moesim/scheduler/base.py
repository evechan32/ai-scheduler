"""Scheduler abstraction: pure decide() contract."""
from __future__ import annotations

from dataclasses import dataclass

from moesim.scheduler.state import ScheduleState

_VALID_KINDS = {"load", "unload", "execute_gpu", "execute_cpu", "evict_kv", "fetch_kv"}


@dataclass(frozen=True)
class Action:
    kind: str
    expert_ids: tuple[str, ...] = ()
    target: str = "gpu"

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"invalid action kind: {self.kind}")


def apply_actions(state: ScheduleState, actions: list[Action]) -> None:
    """Apply load/unload actions to state. Deterministic; raises on capacity violation."""
    for action in actions:
        if action.kind == "unload":
            for eid in action.expert_ids:
                if eid in state.resident:
                    state.resident.remove(eid)
                    state.used_gpu_mb -= state.profiles[eid].size_mb
    for action in actions:
        if action.kind == "load":
            for eid in action.expert_ids:
                size = state.profiles[eid].size_mb
                if state.used_gpu_mb + size > state.gpu_capacity_mb + 1e-9:
                    raise ValueError(
                        f"load of {eid} would exceed capacity "
                        f"({state.used_gpu_mb}+{size}>{state.gpu_capacity_mb})"
                    )
                state.resident.add(eid)
                state.used_gpu_mb += size
    # execute_*/evict_kv/fetch_kv are informational at the state layer


class Scheduler:
    def decide(self, state: ScheduleState, clock: float) -> list[Action]:
        raise NotImplementedError
