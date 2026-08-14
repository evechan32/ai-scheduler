"""Expert executor abstraction — pluggable execution backends."""
from __future__ import annotations


class ExpertExecutor:
    def load(self, expert_ids: list[str]) -> None:
        raise NotImplementedError

    def unload(self, expert_ids: list[str]) -> None:
        raise NotImplementedError

    def execute_gpu(self, expert_id: str, hidden_states: object) -> object:
        raise NotImplementedError

    def execute_cpu(self, expert_id: str, hidden_states: object) -> object:
        raise NotImplementedError
