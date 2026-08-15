"""llama.cpp executor backend — forwards expert execution to a llama_cpp handle."""
from __future__ import annotations

from moesim.executor.base import ExpertExecutor


class LlamaCppExecutor(ExpertExecutor):
    """Wrap a llama-cpp-python Llama instance. load/unload record residency;
    execute_gpu forwards to eval().

    Requires llama-cpp-python: `pip install llama-cpp-python`. The import is
    checked lazily so the executor can be constructed (and unit-tested) with
    fake/duck-typed llama handles even when llama_cpp is not installed; the
    real forward path raises a clear ImportError with the install hint.
    """

    def __init__(self, llama) -> None:
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            self._available = False
        else:
            self._available = True
        self.llama = llama
        self.residency: dict[str, str] = {}

    def _require(self) -> None:
        if self._available or hasattr(self.llama, "eval"):
            return
        raise ImportError(
            "LlamaCppExecutor requires llama-cpp-python: "
            "pip install llama-cpp-python"
        )

    def load(self, expert_ids: list[str]) -> None:
        for eid in expert_ids:
            self.residency[eid] = "loaded"

    def unload(self, expert_ids: list[str]) -> None:
        for eid in expert_ids:
            self.residency[eid] = "unloaded"

    def execute_gpu(self, expert_id: str, hidden_states) -> object:
        self._require()
        return self.llama.eval(hidden_states)

    def execute_cpu(self, expert_id: str, hidden_states) -> object:
        self._require()
        raise NotImplementedError("llama.cpp backend executes on GPU only")
