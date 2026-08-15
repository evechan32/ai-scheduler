"""vLLM executor backend — forwards expert execution to a vLLM engine handle."""
from __future__ import annotations

from moesim.executor.base import ExpertExecutor


class VLLMExecutor(ExpertExecutor):
    """Wrap a vLLM engine. load/unload only record residency (vLLM manages its
    own device memory); execute_gpu forwards to the engine's generate().

    Requires vllm installed: `pip install vllm`. The import is checked lazily
    so the executor can be constructed (and unit-tested) with fake/duck-typed
    engine handles even when vllm is not installed; the real forward path
    raises a clear ImportError with the install hint.
    """

    def __init__(self, engine) -> None:
        try:
            import vllm  # noqa: F401
        except ImportError:
            self._available = False
        else:
            self._available = True
        self.engine = engine
        self.residency: dict[str, str] = {}

    def _require(self) -> None:
        if self._available or hasattr(self.engine, "generate"):
            return
        raise ImportError("VLLMExecutor requires vllm: pip install vllm")

    def load(self, expert_ids: list[str]) -> None:
        for eid in expert_ids:
            self.residency[eid] = "loaded"

    def unload(self, expert_ids: list[str]) -> None:
        for eid in expert_ids:
            self.residency[eid] = "unloaded"

    def execute_gpu(self, expert_id: str, hidden_states) -> object:
        self._require()
        return self.engine.generate(hidden_states)

    def execute_cpu(self, expert_id: str, hidden_states) -> object:
        self._require()
        raise NotImplementedError("vLLM backend executes on GPU only")
