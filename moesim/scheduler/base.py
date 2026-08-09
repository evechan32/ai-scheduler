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


class Scheduler:
    def decide(self, state: ScheduleState, clock: float) -> list[Action]:
        raise NotImplementedError
